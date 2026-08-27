#!/usr/bin/env python3
# Full-data example (reuse existing 30,428 microcells; do not rerun MC2 passes):
#
#   OPENBLAS_NUM_THREADS=32 OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 \
#   /home/aiscuser/.conda/envs/train/bin/python \
#     /mnt/input/sc_cz/Concord/eval/2026_08_19/remerge_stagebin_microcells_to_nodes.py \
#     --input-dir /mnt/input/sc_cz/Concord/eval/2026_08_19/landmark_nodes_mc2_lineage_stagebin2_marginal \
#     --output-dir /mnt/input/sc_cz/Concord/eval/2026_08_19/landmark_nodes_mc2_lineage_stagebin2_marginal_2000 \
#     --target-nodes 2000 \
#     --allocation-knn 30 \
#     --allocation-density-power 1.0 \
#     --regroup-knn 64 \
#     --faiss-threads 32 \
#     --overwrite
#
"""Re-merge existing stage-bin microcells into a new landmark-node count.

This reuses final microcells and does not rerun either MC2-inspired microcell
pass. Landmark quotas are reallocated by equal-microcell-weight normalized
mean marginal cosine-coverage gain, followed by the original weighted
``graph_partition`` merge within each lineage x stage-bin stratum.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np

from select_landmark_nodes_mc2_lineage_stagebin_marginal import (
    allocate_by_microcell_gain,
    graph_partition,
    normalize,
)

HERE = Path(__file__).resolve().parent


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--target-nodes", type=int, default=2000)
    p.add_argument("--min-nodes-per-stratum", type=int, default=1)
    p.add_argument(
        "--min-cells-per-stratum", type=int, default=1,
        help="Exclude lineage x stage-bin strata representing fewer cells than this.",
    )
    p.add_argument("--allocation-knn", type=int, default=30)
    p.add_argument("--allocation-density-power", type=float, default=1.0)
    p.add_argument("--regroup-knn", type=int, default=64)
    p.add_argument("--assignment-chunk-size", type=int, default=1_000_000)
    p.add_argument("--faiss-threads", type=int, default=32)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    values = np.asarray(values)[order]
    weights = np.asarray(weights, dtype=np.float64)[order]
    cumulative = np.cumsum(weights)
    return float(values[min(np.searchsorted(cumulative, q * cumulative[-1]), len(values) - 1)])


def main() -> None:
    args = parse_args()
    if args.min_cells_per_stratum < 1:
        raise ValueError("min-cells-per-stratum must be at least 1")
    import faiss
    import pandas as pd

    faiss.omp_set_num_threads(args.faiss_threads)
    source = args.input_dir.resolve()
    output = args.output_dir.resolve()
    temporary = output.with_name(f".{output.name}.building-{os.getpid()}")
    if source == output:
        raise ValueError("output-dir must differ from input-dir")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; use --overwrite")
    required = (
        "microcells.parquet", "microcell_embeddings.npy",
        "microcell_celltype_composition.parquet", "cell_to_microcell.npy",
        "run_config.json",
    )
    missing = [name for name in required if not (source / name).exists()]
    if missing:
        raise FileNotFoundError(f"input directory is missing: {missing}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    try:
        micro = pd.read_parquet(source / "microcells.parquet").sort_values("microcell_id").reset_index(drop=True)
        micro_embeddings = np.load(source / "microcell_embeddings.npy", mmap_mode="r")
        expected_ids = np.arange(len(micro), dtype=np.int64)
        if not np.array_equal(micro["microcell_id"].to_numpy(), expected_ids):
            raise ValueError("microcell_id must be contiguous and aligned to microcell_embeddings.npy")
        if len(micro_embeddings) != len(micro):
            raise ValueError("microcell table and embedding rows are not aligned")
        group_columns = ["lineage_id", "stage_bin_id"]
        if not set(group_columns).issubset(micro.columns):
            raise ValueError(f"microcells.parquet lacks stage-bin columns: {group_columns}")

        all_grouped = list(micro.groupby(group_columns, sort=True, observed=True))
        grouped = []
        excluded_rows = []
        for key, frame in all_grouped:
            represented_cells = int(frame["n_cells"].sum())
            if represented_cells >= args.min_cells_per_stratum:
                grouped.append((key, frame))
                continue
            first = frame.iloc[0]
            excluded_rows.append(dict(
                lineage_id=int(first.lineage_id), lineage=str(first.lineage),
                stage_bin_id=int(first.stage_bin_id),
                stage_bin_left=float(first.stage_bin_left),
                stage_bin_right=float(first.stage_bin_right),
                stage_bin_label=str(first.stage_bin_label),
                n_microcells=len(frame), n_cells=represented_cells,
                exclusion_reason="below_min_cells_per_stratum",
            ))
        if not grouped:
            raise ValueError("no strata remain after min-cells-per-stratum filtering")
        center_sets, micro_ids_by_stratum = [], []
        for key, frame in grouped:
            ids = frame["microcell_id"].to_numpy(dtype=np.int64)
            micro_ids_by_stratum.append(ids)
            center_sets.append(np.asarray(micro_embeddings[ids], dtype=np.float32))
        if args.target_nodes < len(grouped):
            raise ValueError(f"target-nodes={args.target_nodes} is below {len(grouped)} eligible strata")
        eligible_microcells = sum(len(ids) for ids in micro_ids_by_stratum)
        if args.target_nodes > eligible_microcells:
            raise ValueError(f"target-nodes={args.target_nodes} exceeds {eligible_microcells} eligible microcells")

        log(
            f"allocating {args.target_nodes:,} nodes across {len(grouped):,} eligible strata "
            f"from {eligible_microcells:,} microcells; excluded strata={len(excluded_rows):,}"
        )
        quota = allocate_by_microcell_gain(
            center_sets, args.target_nodes, args.min_nodes_per_stratum,
            args.allocation_knn, args.allocation_density_power,
        )
        micro_to_node = np.full(len(micro), -1, dtype=np.int32)
        node_micro_ids: list[np.ndarray] = []
        node_lineage: list[tuple[int, str]] = []
        node_bins: list[tuple[int, float, float, str]] = []
        offset = 0

        for stratum_index, ((_, frame), ids, centers, q) in enumerate(
            zip(grouped, micro_ids_by_stratum, center_sets, quota)
        ):
            q = int(q)
            weights = frame.set_index("microcell_id").loc[ids, "n_cells"].to_numpy(dtype=np.int64)
            ideal = max(1, int(math.ceil(weights.sum() / q)))
            labels = graph_partition(
                centers, ideal, 0, 2 * ideal, weights=weights,
                target_groups=q, k=min(args.regroup_knn, max(1, len(ids) - 1)),
            )
            if len(np.unique(labels)) != q:
                raise RuntimeError(f"stratum {stratum_index} produced {len(np.unique(labels))}/{q} nodes")
            micro_to_node[ids] = offset + labels
            first = frame.iloc[0]
            lineage = (int(first.lineage_id), str(first.lineage))
            bin_info = (int(first.stage_bin_id), float(first.stage_bin_left),
                        float(first.stage_bin_right), str(first.stage_bin_label))
            for local_node in range(q):
                members = ids[labels == local_node]
                node_micro_ids.append(members)
                node_lineage.append(lineage)
                node_bins.append(bin_info)
            offset += q
            log(f"stratum {stratum_index + 1}/{len(grouped)}: {len(ids):,} microcells -> {q} nodes")
        eligible_ids = np.concatenate(micro_ids_by_stratum)
        if np.any(micro_to_node[eligible_ids] < 0) or offset != args.target_nodes:
            raise RuntimeError("eligible microcell-to-node assignment is incomplete")

        dimension = micro_embeddings.shape[1]
        node_embeddings = np.empty((args.target_nodes, dimension), dtype=np.float32)
        node_rows = []
        for node, ids in enumerate(node_micro_ids):
            frame = micro.set_index("microcell_id").loc[ids]
            weights = frame["n_cells"].to_numpy(dtype=np.float64)
            raw_centers = np.asarray(micro_embeddings[ids], dtype=np.float32)
            center = np.average(raw_centers, axis=0, weights=weights).astype(np.float32)
            node_embeddings[node] = center
            unit_center = normalize(center[None, :])[0]
            distance = 1.0 - normalize(raw_centers) @ unit_center
            total = int(weights.sum())
            mean_stage = float(np.average(frame["mean_stage"], weights=weights))
            second_moment = float(np.average(
                frame["stage_std"].to_numpy(dtype=float) ** 2
                + frame["mean_stage"].to_numpy(dtype=float) ** 2,
                weights=weights,
            ))
            stage_std = math.sqrt(max(0.0, second_moment - mean_stage ** 2))
            lineage_id, lineage = node_lineage[node]
            bin_id, bin_left, bin_right, bin_label = node_bins[node]
            node_rows.append(dict(
                node_id=node, lineage=lineage, lineage_id=lineage_id,
                constraint_stage_id=bin_id, constraint_stage=(bin_left + bin_right) / 2,
                stage_bin_id=bin_id, stage_bin_left=bin_left,
                stage_bin_right=bin_right, stage_bin_label=bin_label,
                n_cells=total, mean_stage=mean_stage, stage_std=stage_std,
                min_stage=float(frame["min_stage"].min()),
                max_stage=float(frame["max_stage"].max()),
                mean_cosine_distance=float(np.average(distance, weights=weights)),
                p95_cosine_distance=weighted_quantile(distance, weights, 0.95),
            ))

        composition = pd.read_parquet(source / "microcell_celltype_composition.parquet")
        composition["node_id"] = micro_to_node[composition["microcell_id"].to_numpy(dtype=np.int64)]
        composition = composition[composition["node_id"] >= 0].copy()
        node_composition = (
            composition.groupby(["node_id", "celltype"], as_index=False, observed=True)["cell_count"].sum()
        )
        totals = node_composition.groupby("node_id")["cell_count"].transform("sum")
        node_composition["fraction"] = node_composition["cell_count"] / totals
        node_composition = node_composition.sort_values(
            ["node_id", "cell_count", "celltype"], ascending=[True, False, True]
        )
        node_composition["rank"] = node_composition.groupby("node_id").cumcount() + 1
        summaries = {}
        for node, frame in node_composition.groupby("node_id", sort=False):
            fractions = frame["fraction"].to_numpy(dtype=float)
            summaries[int(node)] = dict(
                dominant_celltype=str(frame.iloc[0].celltype),
                celltype_purity=float(frame.iloc[0].fraction),
                dominant_celltype_count=int(frame.iloc[0].cell_count),
                celltype_types=len(frame),
                celltype_matched_cells=int(frame.cell_count.sum()),
                celltype_entropy=float(-(fractions * np.log(fractions)).sum()),
            )
        for row in node_rows:
            row.update(summaries.get(row["node_id"], dict(
                dominant_celltype="Unknown", celltype_purity=np.nan,
                dominant_celltype_count=0, celltype_types=0,
                celltype_matched_cells=0, celltype_entropy=np.nan,
            )))

        np.save(temporary / "node_embeddings.npy", node_embeddings)
        np.save(temporary / "microcell_to_node.npy", micro_to_node)
        cell_to_microcell = np.load(source / "cell_to_microcell.npy", mmap_mode="r")
        cell_to_node = np.lib.format.open_memmap(
            temporary / "cell_to_node.npy", mode="w+", dtype=np.int32,
            shape=(len(cell_to_microcell),),
        )
        for lo in range(0, len(cell_to_microcell), args.assignment_chunk_size):
            hi = min(len(cell_to_microcell), lo + args.assignment_chunk_size)
            cell_to_node[lo:hi] = micro_to_node[np.asarray(cell_to_microcell[lo:hi], dtype=np.int64)]
        del cell_to_node

        micro["landmark_node_id"] = micro_to_node
        micro.to_parquet(temporary / "microcells.parquet", index=False)
        pd.DataFrame(node_rows).to_parquet(temporary / "nodes.parquet", index=False)
        node_composition[["node_id", "celltype", "cell_count", "fraction", "rank"]].to_parquet(
            temporary / "node_celltype_composition.parquet", index=False)
        for name in ("microcell_embeddings.npy", "microcell_celltype_composition.parquet", "cell_to_microcell.npy"):
            shutil.copy2(source / name, temporary / name)
        pd.DataFrame(excluded_rows).to_parquet(
            temporary / "excluded_sparse_strata.parquet", index=False
        )

        source_config = json.loads((source / "run_config.json").read_text())
        config = dict(source_config)
        config.update(
            source_microcell_dir=str(source), output_dir=str(output),
            target_nodes=args.target_nodes,
            min_nodes_per_stratum=args.min_nodes_per_stratum,
            allocation_knn=args.allocation_knn,
            allocation_density_power=args.allocation_density_power,
            min_cells_per_stratum=args.min_cells_per_stratum,
            regroup_knn=args.regroup_knn,
            faiss_threads=args.faiss_threads,
            n_nodes=args.target_nodes, n_microcells=len(micro),
            nodes_per_stratum=[int(x) for x in quota],
            allocation_objective="equal_microcell_weight_normalized_mean_marginal_cosine_coverage_gain",
            n_eligible_strata=len(grouped),
            n_excluded_sparse_strata=len(excluded_rows),
            n_excluded_microcells=int(np.count_nonzero(micro_to_node < 0)),
            n_excluded_cells=int(sum(row["n_cells"] for row in excluded_rows)),
            excluded_cell_assignment=-1,
            stratum_filter="sum_microcell_n_cells_gte_min_cells_per_stratum",
            method="remerge_existing_stagebin_microcells_marginal_allocation_weighted_graph_partition",
            node_distance_qc="cell_count_weighted_microcell_center_to_node_center_cosine",
        )
        (temporary / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
        if output.exists():
            shutil.rmtree(output)
        temporary.rename(output)
        log(f"done: {output}")
    except Exception:
        log(f"failed; partial outputs kept: {temporary}")
        raise


if __name__ == "__main__":
    main()
