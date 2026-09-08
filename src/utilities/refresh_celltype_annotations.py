#!/usr/bin/env python3
"""Propagate one lineage's cell-level reannotation through existing results.

The script preserves every clustering assignment, node ID, kNN/Markov edge, and
tree edge. It updates only aggregate cell-type compositions, dominant labels,
purity fields, branch purity, and cell-type-colored figures.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--metadata", type=Path, required=True,
                        help="Base metadata CSV used to retain labels absent from reannotation.")
    parser.add_argument("--reannotation", type=Path, required=True)
    parser.add_argument("--microcell-dir", type=Path, required=True)
    parser.add_argument("--node-dir", type=Path, required=True,
                        help="Leiden temporal-node result directory.")
    parser.add_argument("--tree-dir", type=Path, default=None,
                        help="Optional Markov-tree directory to refresh and replot.")
    parser.add_argument("--lineage", default="Liver")
    parser.add_argument("--csr-dir", type=Path,
                        default=Path("/scratch/amlt_code/traemb_csr_0829"))
    parser.add_argument("--metadata-cell-column", default="cell")
    parser.add_argument("--metadata-celltype-column", default="celltype")
    parser.add_argument("--metadata-lineage-column", default="lineage_final")
    parser.add_argument("--reannotation-cell-column", default="cell_name")
    parser.add_argument("--reannotation-celltype-column", default="celltype_new")
    parser.add_argument("--reannotation-lineage-column", default="lineage_19")
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--backup-tag", default=None,
                        help="Backup suffix. Default: before_<lineage>_reannotation_<timestamp>.")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate and aggregate everything without writing files or plots.")
    return parser.parse_args(argv)


def log(message):
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), message, flush=True)


def validate_unique(frame, column, description):
    if frame[column].isna().any():
        raise ValueError(f"{description} contains missing {column}")
    duplicate = frame[column].astype(str).duplicated(keep=False)
    if duplicate.any():
        examples = frame.loc[duplicate, column].astype(str).head(10).tolist()
        raise ValueError(f"{description} contains duplicate cell IDs: {examples}")


def load_combined_annotations(args):
    metadata_columns = [
        args.metadata_cell_column,
        args.metadata_celltype_column,
        args.metadata_lineage_column,
    ]
    parts = []
    log(f"streaming base {args.lineage} annotations from {args.metadata}")
    for chunk_index, chunk in enumerate(
            pd.read_csv(args.metadata, usecols=metadata_columns,
                        chunksize=args.chunk_size)):
        chosen = chunk[
            chunk[args.metadata_lineage_column].astype(str).eq(args.lineage)]
        if len(chosen):
            parts.append(chosen[
                [args.metadata_cell_column, args.metadata_celltype_column]].copy())
        if (chunk_index + 1) % 10 == 0:
            log(f"metadata chunks={chunk_index + 1}")
    if not parts:
        raise ValueError(f"No {args.lineage!r} rows found in base metadata")
    base = pd.concat(parts, ignore_index=True)
    base.columns = ["cell_id", "base_celltype"]
    base["cell_id"] = base.cell_id.astype(str)
    validate_unique(base, "cell_id", "base metadata lineage subset")

    reannotation_columns = [
        args.reannotation_cell_column,
        args.reannotation_celltype_column,
        args.reannotation_lineage_column,
    ]
    reannotation = pd.read_csv(args.reannotation, usecols=reannotation_columns)
    reannotation = reannotation[
        reannotation[args.reannotation_lineage_column].astype(str).eq(args.lineage)].copy()
    reannotation = reannotation[
        [args.reannotation_cell_column, args.reannotation_celltype_column]]
    reannotation.columns = ["cell_id", "reannotated_celltype"]
    reannotation["cell_id"] = reannotation.cell_id.astype(str)
    validate_unique(reannotation, "cell_id", "reannotation")
    if reannotation.reannotated_celltype.isna().any():
        raise ValueError("Reannotation contains missing cell types")
    reannotation["reannotated_celltype"] = (
        reannotation.reannotated_celltype.astype(str).str.strip())
    if reannotation.reannotated_celltype.eq("").any():
        raise ValueError("Reannotation contains blank cell types")

    missing_from_base = set(reannotation.cell_id) - set(base.cell_id)
    if missing_from_base:
        examples = sorted(missing_from_base)[:10]
        raise ValueError(
            f"{len(missing_from_base)} reannotated cells are absent from base "
            f"{args.lineage} metadata: {examples}")

    combined = base.merge(
        reannotation, on="cell_id", how="left", validate="one_to_one")
    combined["celltype"] = combined.reannotated_celltype.fillna(
        combined.base_celltype.fillna("Unknown").astype(str))
    combined["was_reannotated"] = combined.reannotated_celltype.notna()
    return combined, reannotation


def match_csr_rows(cell_ids_path, wanted, chunk_size):
    cell_ids = np.load(cell_ids_path, mmap_mode="r")
    n_cells = len(cell_ids)
    hashes = np.empty(n_cells, dtype=np.uint64)
    log(f"hashing {n_cells:,} CSR cell IDs")
    for start in range(0, n_cells, chunk_size):
        stop = min(start + chunk_size, n_cells)
        values = np.asarray(cell_ids[start:stop], dtype=object)
        hashes[start:stop] = pd.util.hash_array(
            values, categorize=False).astype(np.uint64)
        if stop % 1_000_000 < chunk_size or stop == n_cells:
            log(f"hashed {stop:,}/{n_cells:,}")
    order = np.argsort(hashes)
    sorted_hashes = hashes[order]
    target_ids = wanted.astype(str).to_numpy(dtype=object)
    target_hashes = pd.util.hash_array(
        target_ids, categorize=False).astype(np.uint64)
    positions = np.searchsorted(sorted_hashes, target_hashes)
    valid = positions < n_cells
    valid &= sorted_hashes[np.minimum(positions, n_cells - 1)] == target_hashes
    if not valid.all():
        examples = target_ids[~valid][:10].tolist()
        raise ValueError(
            f"{np.count_nonzero(~valid)} annotation cells do not match CSR IDs: {examples}")
    rows = order[positions]
    actual = np.asarray(cell_ids[rows]).astype(str)
    mismatch = actual != target_ids
    if mismatch.any():
        examples = list(zip(target_ids[mismatch][:5], actual[mismatch][:5]))
        raise RuntimeError(f"Cell-ID hash collision or mismatch: {examples}")
    return rows.astype(np.int64), n_cells


def make_composition(group_ids, celltypes, id_column):
    frame = pd.DataFrame({
        id_column: np.asarray(group_ids, dtype=np.int64),
        "celltype": np.asarray(celltypes, dtype=object),
    })
    composition = (frame.groupby([id_column, "celltype"], observed=True,
                                 sort=False).size()
                   .rename("cell_count").reset_index())
    totals = composition.groupby(id_column).cell_count.transform("sum")
    composition["fraction"] = composition.cell_count / totals
    composition.sort_values(
        [id_column, "cell_count", "celltype"],
        ascending=[True, False, True], inplace=True)
    composition["rank"] = composition.groupby(id_column).cumcount() + 1
    composition["rank"] = composition["rank"].astype(np.int64)
    composition.reset_index(drop=True, inplace=True)
    return composition


def summarize_composition(composition, id_column):
    top = composition[composition["rank"].eq(1)].copy()
    types = composition.groupby(id_column).size()
    matched = composition.groupby(id_column).cell_count.sum()
    summary = top[[id_column, "celltype", "cell_count", "fraction"]].rename(
        columns={
            "celltype": "dominant_celltype",
            "cell_count": "dominant_celltype_count",
            "fraction": "celltype_purity",
        })
    summary["celltype_types"] = summary[id_column].map(types).astype(np.int64)
    summary["celltype_matched_cells"] = summary[id_column].map(matched).astype(np.int64)
    return summary


def validate_group_coverage(table, lineage, id_column, composition):
    expected = set(
        table.loc[table.lineage.astype(str).eq(lineage), id_column].astype(int))
    observed = set(composition[id_column].astype(int))
    if expected != observed:
        raise RuntimeError(
            f"{id_column} coverage mismatch: missing={len(expected-observed)}, "
            f"unexpected={len(observed-expected)}")


def replace_composition(original, replacement, target_ids, id_column):
    kept = original[~original[id_column].isin(target_ids)]
    result = pd.concat([kept, replacement], ignore_index=True)
    return result.sort_values([id_column, "rank"]).reset_index(drop=True)


def update_summary_columns(table, summary, target_ids, id_column):
    result = table.copy()
    indexed = summary.set_index(id_column)
    ordered_ids = result.loc[result[id_column].isin(target_ids), id_column].astype(int)
    if set(ordered_ids) != set(indexed.index.astype(int)):
        raise RuntimeError(f"Summary rows do not match target {id_column}s")
    row_mask = result[id_column].isin(target_ids)
    mappings = {
        "dominant_celltype": "dominant_celltype",
        "celltype_purity": "celltype_purity",
        "dominant_celltype_count": "dominant_celltype_count",
        "dominant_celltype_cells": "dominant_celltype_count",
        "celltype_types": "celltype_types",
        "celltype_matched_cells": "celltype_matched_cells",
    }
    for output_column, summary_column in mappings.items():
        if output_column in result.columns:
            result.loc[row_mask, output_column] = (
                indexed.loc[ordered_ids.to_numpy(), summary_column].to_numpy())
    return result


def aggregate_nodes(microcell_composition, assignment, group_column):
    joined = microcell_composition.merge(
        assignment[["microcell_id", group_column]],
        on="microcell_id", how="left", validate="many_to_one")
    if joined[group_column].isna().any():
        raise RuntimeError(f"Some microcells have no {group_column} assignment")
    grouped = (joined.groupby([group_column, "celltype"], observed=True,
                              as_index=False).cell_count.sum())
    totals = grouped.groupby(group_column).cell_count.transform("sum")
    grouped["fraction"] = grouped.cell_count / totals
    grouped.sort_values(
        [group_column, "cell_count", "celltype"],
        ascending=[True, False, True], inplace=True)
    grouped["rank"] = grouped.groupby(group_column).cumcount() + 1
    grouped["rank"] = grouped["rank"].astype(np.int64)
    return grouped.reset_index(drop=True)


def refresh_tables(args, combined, csr_rows, n_csr_cells):
    microcell_dir = args.microcell_dir.resolve()
    node_dir = args.node_dir.resolve()

    cell_to_microcell = np.load(
        microcell_dir / "cell_to_microcell.npy", mmap_mode="r")
    if len(cell_to_microcell) != n_csr_cells:
        raise ValueError("cell_to_microcell.npy length does not match CSR cell_ids.npy")
    microcell_ids = np.asarray(cell_to_microcell[csr_rows], dtype=np.int64)
    new_microcell_composition = make_composition(
        microcell_ids, combined.celltype.to_numpy(), "microcell_id")

    microcells_path = microcell_dir / "microcells.parquet"
    microcell_composition_path = (
        microcell_dir / "microcell_celltype_composition.parquet")
    microcells = pd.read_parquet(microcells_path)
    old_microcell_composition = pd.read_parquet(microcell_composition_path)
    validate_group_coverage(
        microcells, args.lineage, "microcell_id", new_microcell_composition)
    target_microcells = microcells.loc[
        microcells.lineage.astype(str).eq(args.lineage), "microcell_id"]
    microcell_summary = summarize_composition(
        new_microcell_composition, "microcell_id")
    updated_microcells = update_summary_columns(
        microcells, microcell_summary, target_microcells, "microcell_id")
    updated_microcell_composition = replace_composition(
        old_microcell_composition, new_microcell_composition,
        target_microcells, "microcell_id")

    assignment_path = node_dir / "microcell_node_assignments.parquet"
    assignment = pd.read_parquet(
        assignment_path,
        columns=["microcell_id", "lineage", "state_id", "node_id"])
    liver_assignment = assignment[
        assignment.lineage.astype(str).eq(args.lineage)].copy()
    if set(liver_assignment.microcell_id.astype(int)) != set(target_microcells.astype(int)):
        raise RuntimeError("Leiden assignment does not cover exactly the refreshed microcells")

    state_composition = aggregate_nodes(
        new_microcell_composition, liver_assignment, "state_id")
    node_composition = aggregate_nodes(
        new_microcell_composition, liver_assignment, "node_id")

    states_path = node_dir / "states.parquet"
    nodes_path = node_dir / "nodes.parquet"
    state_composition_path = node_dir / "state_celltype_composition.parquet"
    node_composition_path = node_dir / "node_celltype_composition.parquet"
    states = pd.read_parquet(states_path)
    nodes = pd.read_parquet(nodes_path)
    old_state_composition = pd.read_parquet(state_composition_path)
    old_node_composition = pd.read_parquet(node_composition_path)
    validate_group_coverage(states, args.lineage, "state_id", state_composition)
    validate_group_coverage(nodes, args.lineage, "node_id", node_composition)
    target_states = states.loc[
        states.lineage.astype(str).eq(args.lineage), "state_id"]
    target_nodes = nodes.loc[
        nodes.lineage.astype(str).eq(args.lineage), "node_id"]
    updated_states = update_summary_columns(
        states, summarize_composition(state_composition, "state_id"),
        target_states, "state_id")
    updated_nodes = update_summary_columns(
        nodes, summarize_composition(node_composition, "node_id"),
        target_nodes, "node_id")
    updated_state_composition = replace_composition(
        old_state_composition, state_composition, target_states, "state_id")
    updated_node_composition = replace_composition(
        old_node_composition, node_composition, target_nodes, "node_id")

    updates = {
        microcells_path: updated_microcells,
        microcell_composition_path: updated_microcell_composition,
        states_path: updated_states,
        nodes_path: updated_nodes,
        state_composition_path: updated_state_composition,
        node_composition_path: updated_node_composition,
    }
    tree_objects = None
    if args.tree_dir is not None:
        tree_dir = args.tree_dir.resolve()
        tree_nodes_path = tree_dir / "tree_nodes.parquet"
        segments_path = tree_dir / "branch_segments.parquet"
        tree_nodes = pd.read_parquet(tree_nodes_path)
        segments = pd.read_parquet(segments_path)
        node_summary = updated_nodes.set_index("node_id")
        tree_target = (
            tree_nodes.node_type.eq("temporal_node") &
            tree_nodes.lineage.astype(str).eq(args.lineage))
        ids = tree_nodes.loc[tree_target, "node_id"].astype(int)
        for column in (
                "dominant_celltype", "dominant_celltype_cells",
                "celltype_purity"):
            if column in tree_nodes and column in node_summary:
                tree_nodes.loc[tree_target, column] = (
                    node_summary.loc[ids.to_numpy(), column].to_numpy())

        node_lookup = tree_nodes.set_index("node_id")
        new_segment_purity = []
        for row in segments.itertuples(index=False):
            path_ids = json.loads(row.path_node_ids)
            path = node_lookup.loc[path_ids]
            real = path[path.node_type.eq("temporal_node")]
            valid = real.celltype_purity.notna()
            if valid.any():
                purity = float(np.average(
                    real.loc[valid, "celltype_purity"],
                    weights=real.loc[valid, "n_cells"]))
            else:
                purity = np.nan
            new_segment_purity.append(purity)
        segments["cell_weighted_celltype_purity"] = new_segment_purity
        updates[tree_nodes_path] = tree_nodes
        updates[segments_path] = segments
        tree_objects = (tree_nodes, pd.read_parquet(tree_dir / "tree_edges.parquet"),
                        pd.read_parquet(tree_dir / "candidate_markov_edges.parquet"))

    return updates, {
        "microcell_composition": new_microcell_composition,
        "state_composition": state_composition,
        "node_composition": node_composition,
        "updated_nodes": updated_nodes,
        "tree_objects": tree_objects,
    }


def backup_path(path, tag):
    return path.with_name(f"{path.stem}.{tag}{path.suffix}")


def commit_parquet_updates(updates, backup_tag):
    temporary = {}
    for path, frame in updates.items():
        if not path.exists():
            raise FileNotFoundError(path)
        backup = backup_path(path, backup_tag)
        if backup.exists():
            raise FileExistsError(f"Backup already exists: {backup}")
        temp = path.with_name(f".{path.name}.{os.getpid()}.refresh_tmp")
        if temp.exists():
            raise FileExistsError(temp)
        frame.to_parquet(temp, index=False)
        temporary[path] = temp
    try:
        for path in updates:
            shutil.copy2(path, backup_path(path, backup_tag))
        for path, temp in temporary.items():
            temp.replace(path)
    except Exception:
        for temp in temporary.values():
            if temp.exists():
                temp.unlink()
        raise


def render_tree_plots(args, tree_objects):
    if args.tree_dir is None or args.skip_plots or tree_objects is None:
        return
    import importlib.util

    module_path = (
        Path(__file__).resolve().parents[1] /
        "workflow" / "build_markov_node_tree.py")
    spec = importlib.util.spec_from_file_location(
        "build_markov_node_tree", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    tree_dir = args.tree_dir.resolve()
    tree_nodes, tree_edges, candidate_edges = tree_objects
    with tempfile.TemporaryDirectory(
            prefix=".celltype_refresh_plots_", dir=tree_dir.parent) as temp:
        plot_dir = Path(temp)
        module.plot_outputs(
            tree_nodes, tree_edges, candidate_edges, plot_dir, args.dpi)
        shutil.copytree(plot_dir, tree_dir, dirs_exist_ok=True)


def main(argv=None):
    args = parse_args(argv)
    if args.chunk_size < 1 or args.dpi < 1:
        raise ValueError("--chunk-size and --dpi must be positive")
    for left, right in (
            (args.microcell_dir.resolve(), args.node_dir.resolve()),
            (args.microcell_dir.resolve(),
             args.tree_dir.resolve() if args.tree_dir else None),
            (args.node_dir.resolve(),
             args.tree_dir.resolve() if args.tree_dir else None)):
        if right is not None and left == right:
            raise ValueError("microcell, node, and tree directories must differ")

    combined, reannotation = load_combined_annotations(args)
    log(f"base {args.lineage} cells={len(combined):,}; "
        f"reannotated={combined.was_reannotated.sum():,}; "
        f"preserved base labels={(~combined.was_reannotated).sum():,}")
    log(f"final cell labels: {combined.celltype.value_counts().to_dict()}")
    csr_rows, n_csr_cells = match_csr_rows(
        args.csr_dir.resolve() / "cell_ids.npy",
        combined.cell_id, args.chunk_size)
    updates, results = refresh_tables(
        args, combined, csr_rows, n_csr_cells)

    node_composition = results["node_composition"]
    node_top = node_composition[node_composition["rank"].eq(1)]
    summary = {
        "lineage": args.lineage,
        "base_metadata": str(args.metadata.resolve()),
        "reannotation": str(args.reannotation.resolve()),
        "n_base_lineage_cells": int(len(combined)),
        "n_reannotated_cells": int(combined.was_reannotated.sum()),
        "n_preserved_base_label_cells": int((~combined.was_reannotated).sum()),
        "final_celltype_counts": {
            str(key): int(value)
            for key, value in combined.celltype.value_counts().items()
        },
        "dominant_temporal_node_counts": {
            str(key): int(value)
            for key, value in node_top.celltype.value_counts().items()
        },
        "n_temporal_nodes": int(node_top.node_id.nunique()),
        "assignments_changed": False,
        "tree_edges_changed": False,
    }
    log(f"dominant temporal-node labels: {summary['dominant_temporal_node_counts']}")
    if args.dry_run:
        log("dry run complete; no files changed")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    backup_tag = args.backup_tag
    if backup_tag is None:
        safe_lineage = "".join(
            character.lower() if character.isalnum() else "_"
            for character in args.lineage).strip("_")
        backup_tag = (
            f"before_{safe_lineage}_reannotation_"
            f"{time.strftime('%Y%m%d_%H%M%S')}")
    log(f"writing {len(updates)} refreshed tables; backup tag={backup_tag}")
    commit_parquet_updates(updates, backup_tag)
    render_tree_plots(args, results["tree_objects"])
    summary["backup_tag"] = backup_tag
    summary["plots_refreshed"] = bool(
        args.tree_dir is not None and not args.skip_plots)
    summary_path = (
        args.tree_dir.resolve() if args.tree_dir is not None
        else args.node_dir.resolve()) / "celltype_refresh_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    log(f"done; summary: {summary_path}")


if __name__ == "__main__":
    main()
