#!/usr/bin/env python3
"""Plot an existing Markov node tree using sampled underlying cells.

The tree topology and branch coordinates are read unchanged from an existing
Markov-tree run. Cells are sampled independently within every temporal node.
Each sampled cell is drawn at its own predicted stage and around its node's
branch coordinate, so the plots show within-node temporal variation without
using cell type or stage to rebuild the tree.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("/mnt/input/sc_cz/Concord/eval/2026_08_19/TraEmbTree")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--tree-dir", type=Path,
        default=DEFAULT_ROOT / "markov_tree_0903_nodeknn_k10")
    parser.add_argument(
        "--node-dir", type=Path,
        default=DEFAULT_ROOT / "leiden_nodes_0903_by_lineage_r05_stage2")
    parser.add_argument(
        "--microcell-dir", type=Path,
        default=DEFAULT_ROOT / "microcells_0903_predicted_stage")
    parser.add_argument(
        "--predicted-stage", type=Path,
        default=Path("/mnt/input/sc_cz/Concord/eval/2026_09_03/predicted_stage.csv"))
    parser.add_argument(
        "--metadata", type=Path,
        default=Path("/mnt/input/sc_cz/Concord/data/all_lineage_260829_liver_reanno.csv"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cells-per-node", type=int, default=150)
    parser.add_argument(
        "--candidate-factor", type=float, default=3.0,
        help="Temporary oversampling factor used to obtain an exact per-node cap efficiently.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--branch-jitter", type=float, default=0.012)
    parser.add_argument("--point-size", type=float, default=1.2)
    parser.add_argument("--point-alpha", type=float, default=0.65)
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--metadata-cell-column", default="cell")
    parser.add_argument("--metadata-celltype-column", default="celltype")
    return parser.parse_args(argv)


def log(message):
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), message, flush=True)


def validate_args(args):
    if args.cells_per_node < 1:
        raise ValueError("--cells-per-node must be positive")
    if args.candidate_factor <= 1 or not np.isfinite(args.candidate_factor):
        raise ValueError("--candidate-factor must be finite and greater than 1")
    if args.branch_jitter < 0 or not np.isfinite(args.branch_jitter):
        raise ValueError("--branch-jitter must be finite and nonnegative")
    if args.point_size <= 0 or not np.isfinite(args.point_size):
        raise ValueError("--point-size must be positive and finite")
    if not 0 < args.point_alpha <= 1:
        raise ValueError("--point-alpha must lie in (0, 1]")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be positive")


def load_tree_and_mapping(args):
    nodes = pd.read_parquet(args.tree_dir / "tree_nodes.parquet")
    edges = pd.read_parquet(args.tree_dir / "tree_edges.parquet")
    assignment = pd.read_parquet(
        args.node_dir / "microcell_node_assignments.parquet",
        columns=["microcell_id", "node_id"])
    required_nodes = {"node_id", "node_type", "lineage", "tree_stage", "tree_x"}
    required_edges = {"parent_id", "child_id", "lineage", "edge_kind"}
    if not required_nodes.issubset(nodes):
        raise ValueError(f"tree_nodes.parquet is missing {sorted(required_nodes-set(nodes))}")
    if not required_edges.issubset(edges):
        raise ValueError(f"tree_edges.parquet is missing {sorted(required_edges-set(edges))}")
    real = nodes[nodes.node_type.eq("temporal_node")].copy()
    real_ids = real.node_id.to_numpy(dtype=np.int64)
    if not np.array_equal(np.sort(real_ids), np.arange(len(real))):
        raise ValueError("Temporal node IDs must be contiguous from zero")
    microcell_ids = assignment.microcell_id.to_numpy(dtype=np.int64)
    if (assignment.microcell_id.duplicated().any() or
            not np.array_equal(np.sort(microcell_ids), np.arange(len(assignment)))):
        raise ValueError("microcell_id must be unique and contiguous from zero")
    assignment_nodes = assignment.node_id.to_numpy(dtype=np.int64)
    if assignment_nodes.min() < -1 or assignment_nodes.max() >= len(real):
        raise ValueError("Assignment node_id must be -1 or a valid temporal node")
    maximum = int(assignment.microcell_id.max())
    microcell_to_node = np.full(maximum + 1, -1, dtype=np.int32)
    microcell_to_node[microcell_ids] = assignment_nodes.astype(np.int32)
    return nodes, edges, real, microcell_to_node


def sample_cell_indices(cell_to_microcell_path, microcell_to_node, real_nodes,
                        cap, factor, seed, chunk_size):
    """Return an exact uniform cap per node using oversampling then trimming."""
    cell_to_microcell = np.load(cell_to_microcell_path, mmap_mode="r")
    n_cells = len(cell_to_microcell)
    node_count = len(real_nodes)
    expected = np.zeros(node_count, dtype=np.int64)
    indexed = real_nodes.set_index("node_id")
    if "n_cells" in indexed:
        expected[indexed.index.to_numpy(dtype=np.int64)] = indexed.n_cells.to_numpy(np.int64)
    else:
        log("counting cells per temporal node")
        for start in range(0, n_cells, chunk_size):
            stop = min(start + chunk_size, n_cells)
            microcells = np.asarray(cell_to_microcell[start:stop], dtype=np.int64)
            valid = (microcells >= 0) & (microcells < len(microcell_to_node))
            mapped = microcell_to_node[microcells[valid]]
            mapped = mapped[(mapped >= 0) & (mapped < node_count)]
            expected += np.bincount(mapped, minlength=node_count)
    if (expected <= 0).any():
        raise ValueError("At least one temporal node has no underlying cells")

    probability = np.minimum(1.0, factor * cap / expected.astype(np.float64))
    rng = np.random.default_rng(seed)
    selected_indices, selected_nodes, selected_keys = [], [], []
    log(f"sampling at most {cap:,} cells from each of {node_count:,} nodes")
    for start in range(0, n_cells, chunk_size):
        stop = min(start + chunk_size, n_cells)
        microcells = np.asarray(cell_to_microcell[start:stop], dtype=np.int64)
        valid = (microcells >= 0) & (microcells < len(microcell_to_node))
        nodes = np.full(stop - start, -1, dtype=np.int32)
        nodes[valid] = microcell_to_node[microcells[valid]]
        valid &= (nodes >= 0) & (nodes < node_count)
        keys = rng.random(stop - start)
        chosen = valid & (keys < probability[np.maximum(nodes, 0)])
        local = np.flatnonzero(chosen)
        if len(local):
            selected_indices.append(local.astype(np.int64) + start)
            selected_nodes.append(nodes[local].copy())
            selected_keys.append(keys[local].copy())
        if stop % 5_000_000 < chunk_size or stop == n_cells:
            log(f"scanned {stop:,}/{n_cells:,} cells")

    candidates = pd.DataFrame({
        "cell_index": np.concatenate(selected_indices),
        "node_id": np.concatenate(selected_nodes).astype(np.int64),
        "_random_key": np.concatenate(selected_keys),
    })
    candidates.sort_values(["node_id", "_random_key"], inplace=True)
    candidates = candidates[candidates.groupby("node_id", sort=False).cumcount().lt(cap)]
    candidates.drop(columns="_random_key", inplace=True)
    candidates.sort_values("cell_index", inplace=True)
    candidates.reset_index(drop=True, inplace=True)

    observed = candidates.groupby("node_id").size().reindex(range(node_count), fill_value=0)
    targets = np.minimum(expected, cap)
    underfilled = observed.to_numpy() < targets
    if underfilled.any():
        bad = np.flatnonzero(underfilled)
        raise RuntimeError(
            f"Oversampling left {len(bad)} nodes below their target sample size. "
            f"Rerun with a larger --candidate-factor; examples: {bad[:10].tolist()}")
    return candidates, n_cells, expected


def attach_predicted_stage(sample, path, n_cells, chunk_size):
    wanted = np.zeros(n_cells, dtype=bool)
    wanted[sample.cell_index.to_numpy(dtype=np.int64)] = True
    parts = []
    log(f"streaming sampled stages and cell IDs from {path}")
    for number, chunk in enumerate(pd.read_csv(
            path, usecols=["idx", "cell_id", "predicted_stage"],
            dtype={"idx": "int64", "cell_id": str, "predicted_stage": "float64"},
            keep_default_na=False, chunksize=chunk_size), 1):
        indices = chunk.idx.to_numpy(dtype=np.int64)
        if len(indices) and (indices.min() < 0 or indices.max() >= n_cells):
            raise ValueError("predicted_stage.csv contains idx outside cell_to_microcell.npy")
        chosen = wanted[indices]
        if chosen.any():
            parts.append(chunk.loc[chosen].copy())
        if number % 10 == 0:
            log(f"stage chunks={number}")
    stages = pd.concat(parts, ignore_index=True)
    if stages.idx.duplicated().any() or len(stages) != len(sample):
        raise ValueError(
            f"Expected {len(sample):,} unique sampled stages, found {len(stages):,}")
    stages.rename(columns={"idx": "cell_index"}, inplace=True)
    result = sample.merge(stages, on="cell_index", how="left", validate="one_to_one")
    if result[["cell_id", "predicted_stage"]].isna().any().any():
        raise ValueError("Some sampled cells lack cell ID or predicted stage")
    if not np.isfinite(result.predicted_stage.to_numpy(dtype=float)).all():
        raise ValueError("Sampled predicted stages contain non-finite values")
    result["cell_id"] = result.cell_id.astype(str)
    return result


def attach_celltypes(sample, args):
    wanted = set(sample.cell_id)
    parts = []
    log(f"streaming sampled cell types from {args.metadata}")
    for number, chunk in enumerate(pd.read_csv(
            args.metadata,
            usecols=[args.metadata_cell_column, args.metadata_celltype_column],
            dtype={args.metadata_cell_column: str}, keep_default_na=False,
            chunksize=args.chunk_size), 1):
        chosen = chunk[args.metadata_cell_column].astype(str).isin(wanted)
        if chosen.any():
            parts.append(chunk.loc[chosen].copy())
        if number % 10 == 0:
            log(f"metadata chunks={number}")
    annotations = pd.concat(parts, ignore_index=True)
    annotations = annotations[[args.metadata_cell_column, args.metadata_celltype_column]]
    annotations.columns = ["cell_id", "celltype"]
    annotations["cell_id"] = annotations.cell_id.astype(str)
    if annotations.cell_id.duplicated().any():
        raise ValueError("Base metadata has duplicate sampled cell IDs")
    result = sample.merge(annotations, on="cell_id", how="left", validate="one_to_one")
    missing = result.celltype.isna()
    if missing.any():
        examples = result.loc[missing, "cell_id"].head(10).tolist()
        raise ValueError(f"{missing.sum():,} sampled cells lack base cell type: {examples}")
    result["celltype"] = result.celltype.astype(str)
    return result


def discrete_colors(size, matplotlib):
    colors, seen = [], set()
    for name in ("tab20", "tab20b", "tab20c", "Set1", "Set2", "Set3",
                 "Dark2", "Paired", "Accent"):
        for value in matplotlib.colormaps[name].colors:
            color = matplotlib.colors.to_hex(value).upper()
            if color not in seen and color not in {"#FFFFFF", "#000000", "#B0B0B0"}:
                seen.add(color)
                colors.append(color)
    golden = 0.618033988749895
    index = 0
    while len(colors) < size:
        hsv = ((index * golden) % 1.0, (0.62, 0.78, 0.9)[index % 3],
               (0.68, 0.82, 0.94)[(index // 3) % 3])
        color = matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb(hsv)).upper()
        if color not in seen:
            seen.add(color)
            colors.append(color)
        index += 1
    return colors[:size]


def make_color_map(sample, tree_dir, matplotlib):
    counts = sample.groupby("celltype", sort=False).size().sort_values(ascending=False)
    labels = counts.index.tolist()
    colors = {}
    existing_path = tree_dir / "celltype_colors.csv"
    if existing_path.exists():
        existing = pd.read_csv(existing_path)
        if {"celltype", "color"}.issubset(existing):
            colors.update(dict(zip(existing.celltype.astype(str), existing.color.astype(str))))
    missing = [label for label in labels if label not in colors]
    used = set(colors.values())
    for color in discrete_colors(len(missing) + len(used), matplotlib):
        if not missing:
            break
        if color not in used:
            colors[missing.pop(0)] = color
            used.add(color)
    return labels, counts, colors


def plot_outputs(sample, nodes, edges, output, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output.mkdir(parents=True, exist_ok=False)
    lookup = nodes.set_index("node_id")
    real_nodes = nodes[nodes.node_type.eq("temporal_node")]
    node_info = real_nodes.set_index("node_id")[["lineage", "tree_x"]]
    sample = sample.join(node_info, on="node_id", validate="many_to_one")
    rng = np.random.default_rng(args.seed + 1)
    sample["plot_branch"] = (
        sample.tree_x.to_numpy(dtype=float) +
        rng.normal(0.0, args.branch_jitter, len(sample)))
    lineages = sorted(real_nodes.lineage.astype(str).unique())
    labels, counts, colors = make_color_map(sample, args.tree_dir, matplotlib)

    def draw_edges(ax, local_edges, horizontal=True):
        for edge in local_edges.itertuples(index=False):
            x = [lookup.at[edge.parent_id, "tree_stage"],
                 lookup.at[edge.child_id, "tree_stage"]]
            y = [lookup.at[edge.parent_id, "tree_x"],
                 lookup.at[edge.child_id, "tree_x"]]
            ax.plot(x if horizontal else y, y if horizontal else x,
                    color="#9A9A9A", linewidth=.35, alpha=.35, zorder=1)

    def save(fig, stem):
        fig.savefig(output / f"{stem}.png", dpi=args.dpi, bbox_inches="tight")
        fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(17, 13))
    draw_edges(ax, edges)
    points = ax.scatter(sample.predicted_stage, sample.plot_branch,
                        c=sample.predicted_stage, cmap="viridis",
                        s=args.point_size, alpha=args.point_alpha,
                        linewidths=0, rasterized=True, zorder=2)
    fig.colorbar(points, ax=ax, label="Single-cell predicted stage")
    ax.set(xlabel="Single-cell predicted stage", ylabel="Lineage-separated branch layout",
           title=(f"Markov tree represented by {len(sample):,} sampled cells "
                  "(early to late, left to right)"))
    ax.set_yticks(np.arange(len(lineages)) + .5, lineages, fontsize=8)
    ax.grid(axis="x", color="#EEEEEE", linewidth=.5)
    save(fig, "tree_sampled_cells_by_stage")

    fig, ax = plt.subplots(figsize=(18, 14))
    draw_edges(ax, edges)
    ax.scatter(sample.predicted_stage, sample.plot_branch,
               c=sample.celltype.map(colors), s=args.point_size,
               alpha=args.point_alpha, linewidths=0, rasterized=True, zorder=2)
    handles = [Line2D([0], [0], marker="o", linestyle="", color=colors[label], label=label)
               for label in labels]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, -.08),
              fontsize=6, ncol=min(8, max(1, len(labels))), markerscale=2,
              columnspacing=.8, handletextpad=.3)
    ax.set(xlabel="Single-cell predicted stage", ylabel="Lineage-separated branch layout",
           title=(f"Markov tree represented by sampled cells — all {len(labels)} observed "
                  "cell types (stratified per node)"))
    ax.set_yticks(np.arange(len(lineages)) + .5, lineages, fontsize=8)
    ax.grid(axis="x", color="#EEEEEE", linewidth=.5)
    save(fig, "tree_sampled_cells_by_celltype")

    lineage_dir = output / "trees_by_lineage_cells"
    lineage_dir.mkdir()
    for position, lineage in enumerate(lineages, 1):
        local = sample[sample.lineage.eq(lineage)]
        local_edges = edges[edges.lineage.eq(lineage) & edges.edge_kind.ne("global_root")]
        local_labels = [label for label in labels if label in set(local.celltype)]
        legend_columns = max(1, math.ceil(len(local_labels) / 30))
        fig, ax = plt.subplots(figsize=(12 + 3 * legend_columns, 8))
        draw_edges(ax, local_edges)
        ax.scatter(local.predicted_stage, local.plot_branch,
                   c=local.celltype.map(colors), s=max(args.point_size, 2.0),
                   alpha=args.point_alpha, linewidths=0, rasterized=True, zorder=2)
        root = nodes[nodes.node_type.eq("lineage_root") & nodes.lineage.eq(lineage)]
        ax.scatter(root.tree_stage, root.tree_x, marker="*", s=80,
                   color="black", zorder=3)
        handles = [Line2D([0], [0], marker="o", linestyle="", color=colors[label], label=label)
                   for label in local_labels]
        handles.append(Line2D([0], [0], marker="*", linestyle="", color="black",
                              markersize=9, label="Lineage root"))
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1, .5),
                  fontsize=8, ncol=legend_columns, markerscale=1.5,
                  columnspacing=1.0, handletextpad=.35)
        ax.set(xlabel="Single-cell predicted stage", ylabel="Branch layout",
               title=(f"{lineage} — {len(local):,} sampled cells and "
                      f"{len(local_labels)} observed cell types"))
        ax.grid(axis="x", color="#EEEEEE", linewidth=.5)
        safe = re.sub(r"[^A-Za-z0-9]+", "_", lineage).strip("_").lower()
        stem = lineage_dir / f"{position:02d}_{safe}"
        fig.savefig(stem.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)

    sample.drop(columns=["tree_x", "plot_branch"]).to_parquet(
        output / "sampled_cells.parquet", index=False)
    pd.DataFrame({
        "celltype": labels,
        "color": [colors[label] for label in labels],
        "n_sampled_cells": [int(counts[label]) for label in labels],
    }).to_csv(output / "sampled_celltype_colors.csv", index=False)
    return sample


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    args.tree_dir = args.tree_dir.resolve()
    args.node_dir = args.node_dir.resolve()
    args.microcell_dir = args.microcell_dir.resolve()
    output = (args.output_dir.resolve() if args.output_dir is not None else
              args.tree_dir / "sampled_cell_tree_plots")
    if output.exists():
        raise FileExistsError(f"Output path already exists: {output}")

    log("loading the existing tree and microcell-to-node assignments")
    nodes, edges, real_nodes, microcell_to_node = load_tree_and_mapping(args)
    sample, n_input_cells, node_counts = sample_cell_indices(
        args.microcell_dir / "cell_to_microcell.npy", microcell_to_node,
        real_nodes, args.cells_per_node, args.candidate_factor,
        args.seed, args.chunk_size)
    sample = attach_predicted_stage(
        sample, args.predicted_stage, n_input_cells, args.chunk_size)
    sample = attach_celltypes(sample, args)
    plotted = plot_outputs(sample, nodes, edges, output, args)

    per_node = plotted.groupby("node_id").size()
    summary = {
        "n_input_cells": int(n_input_cells),
        "n_temporal_nodes": int(len(real_nodes)),
        "n_sampled_cells": int(len(plotted)),
        "cells_per_node_cap": int(args.cells_per_node),
        "min_sampled_cells_per_node": int(per_node.min()),
        "median_sampled_cells_per_node": float(per_node.median()),
        "max_sampled_cells_per_node": int(per_node.max()),
        "n_sampled_celltypes": int(plotted.celltype.nunique()),
        "sampling": "uniform within each temporal node; all cells retained below cap",
        "x_coordinate": "individual-cell predicted_stage",
        "y_coordinate": "assigned node tree_x plus display-only Gaussian jitter",
        "tree_topology_modified": False,
        "population_abundance_interpretation": (
            "Do not infer abundance from point density because sampling is stratified by node."),
        "seed": int(args.seed),
        "node_cell_count_sum": int(node_counts.sum()),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    (output / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    log(f"done: {output}")
    log(f"sampled cells={len(plotted):,}; observed cell types={plotted.celltype.nunique():,}")


if __name__ == "__main__":
    main()
