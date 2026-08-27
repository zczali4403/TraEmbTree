#!/usr/bin/env python3
"""Re-merge stage-bin microcells with a data-driven landmark count.

Instead of specifying ``--target-nodes``, stop global marginal allocation when
the best available next landmark would improve mean microcell cosine coverage
by less than ``--min-marginal-gain``.  Every non-empty lineage x stage-bin
stratum receives at least ``--min-nodes-per-stratum`` landmarks.

The final weighted graph merging is delegated to
``remerge_stagebin_microcells_to_nodes.py`` so that node construction is
identical to the fixed-count workflow.
"""

from __future__ import annotations

import argparse
import heapq
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from select_landmark_nodes_mc2_lineage_stagebin_marginal import normalize


HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--min-marginal-gain", type=float, required=True,
        help=("Stop when the largest available reduction in mean nearest-center "
              "cosine distance is below this value."),
    )
    parser.add_argument("--min-nodes-per-stratum", type=int, default=1)
    parser.add_argument(
        "--min-cells-per-stratum", type=int, default=50,
        help="Exclude lineage x stage-bin strata representing fewer cells than this.",
    )
    parser.add_argument("--allocation-knn", type=int, default=30)
    parser.add_argument("--allocation-density-power", type=float, default=1.0)
    parser.add_argument("--regroup-knn", type=int, default=64)
    parser.add_argument("--assignment-chunk-size", type=int, default=1_000_000)
    parser.add_argument("--faiss-threads", type=int, default=32)
    parser.add_argument(
        "--max-nodes", type=int, default=None,
        help="Optional safety cap; it is not a target and may never be reached.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def allocate_until_gain(
    center_sets: list[np.ndarray], minimum: int, knn: int,
    density_power: float, min_gain: float, max_nodes: int | None,
) -> tuple[np.ndarray, float | None, str]:
    """Return per-stratum quotas by thresholding the best marginal gain."""
    import faiss

    states: list[dict] = []
    heap: list[tuple[float, int, int]] = []

    def next_proposal(state: dict):
        if len(state["selected"]) >= len(state["z"]):
            return None
        score = state["nearest"].astype(np.float64) * np.power(
            state["density_rank"], density_power
        )
        score[np.asarray(state["selected"], dtype=np.int64)] = -np.inf
        candidate = int(np.argmax(score))
        distance = np.maximum(0.0, 1.0 - state["z"] @ state["z"][candidate]).astype(np.float32)
        gain = float(np.maximum(0.0, state["nearest"] - distance).mean())
        return candidate, distance, gain

    for centers in center_sets:
        z = normalize(centers)
        count = len(z)
        if count == 1:
            density = np.ones(1, dtype=np.float32)
        else:
            k = min(max(1, knn), count - 1)
            index = faiss.IndexFlatIP(z.shape[1])
            index.add(z)
            similarity, _ = index.search(z, k + 1)
            mean_distance = np.maximum(0.0, 1.0 - similarity[:, 1:]).mean(axis=1)
            density = (1.0 / np.maximum(mean_distance, 1e-8)).astype(np.float32)
        order = np.argsort(density, kind="stable")
        density_rank = np.empty(count, dtype=np.float32)
        density_rank[order] = (np.arange(count, dtype=np.float32) + 1) / count
        first = int(np.argmax(density))
        nearest = np.maximum(0.0, 1.0 - z @ z[first]).astype(np.float32)
        state = dict(z=z, density_rank=density_rank, selected=[first], nearest=nearest)
        required = min(max(1, minimum), count)
        while len(state["selected"]) < required:
            proposal = next_proposal(state)
            if proposal is None:
                break
            candidate, distance, _ = proposal
            state["selected"].append(candidate)
            state["nearest"] = np.minimum(state["nearest"], distance)
        states.append(state)

    allocated = sum(len(state["selected"]) for state in states)
    if max_nodes is not None and max_nodes < allocated:
        raise ValueError(
            f"max-nodes={max_nodes} is below the mandatory allocation of {allocated}"
        )
    for stratum, state in enumerate(states):
        proposal = next_proposal(state)
        state["proposal"] = proposal
        if proposal is not None:
            heapq.heappush(heap, (-proposal[2], stratum, len(state["selected"])))

    last_accepted_gain = None
    stop_reason = "all_microcells_selected"
    while heap:
        negative_gain, stratum, version = heapq.heappop(heap)
        state = states[stratum]
        if version != len(state["selected"]) or state["proposal"] is None:
            continue
        candidate, distance, gain = state["proposal"]
        if gain < min_gain:
            stop_reason = "marginal_gain_threshold"
            break
        if max_nodes is not None and allocated >= max_nodes:
            stop_reason = "max_nodes_safety_cap"
            break
        state["selected"].append(candidate)
        state["nearest"] = np.minimum(state["nearest"], distance)
        allocated += 1
        last_accepted_gain = gain
        if allocated % 25 == 0:
            print(f"adaptive allocation: {allocated:,} nodes; accepted gain={gain:.6g}", flush=True)
        state["proposal"] = next_proposal(state)
        if state["proposal"] is not None:
            heapq.heappush(
                heap, (-state["proposal"][2], stratum, len(state["selected"]))
            )

    quota = np.asarray([len(state["selected"]) for state in states], dtype=np.int32)
    return quota, last_accepted_gain, stop_reason


def main() -> None:
    args = parse_args()
    if not np.isfinite(args.min_marginal_gain) or args.min_marginal_gain < 0:
        raise ValueError("min-marginal-gain must be finite and non-negative")
    if args.min_nodes_per_stratum < 1:
        raise ValueError("min-nodes-per-stratum must be at least 1")
    if args.min_cells_per_stratum < 1:
        raise ValueError("min-cells-per-stratum must be at least 1")

    import faiss
    import pandas as pd

    faiss.omp_set_num_threads(args.faiss_threads)
    source = args.input_dir.resolve()
    output = args.output_dir.resolve()
    micro = pd.read_parquet(source / "microcells.parquet").sort_values("microcell_id")
    embeddings = np.load(source / "microcell_embeddings.npy", mmap_mode="r")
    if len(micro) != len(embeddings):
        raise ValueError("microcell table and embedding rows are not aligned")
    expected = np.arange(len(micro), dtype=np.int64)
    if not np.array_equal(micro["microcell_id"].to_numpy(dtype=np.int64), expected):
        raise ValueError("microcell_id must be contiguous and aligned to embeddings")

    all_grouped = list(micro.groupby(["lineage_id", "stage_bin_id"], sort=True, observed=True))
    grouped = [
        item for item in all_grouped
        if int(item[1]["n_cells"].sum()) >= args.min_cells_per_stratum
    ]
    if not grouped:
        raise ValueError("no strata remain after min-cells-per-stratum filtering")
    center_sets = [
        np.asarray(embeddings[frame["microcell_id"].to_numpy(dtype=np.int64)], dtype=np.float32)
        for _, frame in grouped
    ]
    quota, last_gain, stop_reason = allocate_until_gain(
        center_sets, args.min_nodes_per_stratum, args.allocation_knn,
        args.allocation_density_power, args.min_marginal_gain, args.max_nodes,
    )
    target_nodes = int(quota.sum())
    print(
        f"threshold selected {target_nodes:,} nodes across {len(grouped):,} eligible strata; "
        f"excluded_strata={len(all_grouped)-len(grouped):,}; "
        f"stop_reason={stop_reason}", flush=True,
    )

    command = [
        sys.executable, str(HERE / "remerge_stagebin_microcells_to_nodes.py"),
        "--input-dir", str(source), "--output-dir", str(output),
        "--target-nodes", str(target_nodes),
        "--min-nodes-per-stratum", str(args.min_nodes_per_stratum),
        "--min-cells-per-stratum", str(args.min_cells_per_stratum),
        "--allocation-knn", str(args.allocation_knn),
        "--allocation-density-power", str(args.allocation_density_power),
        "--regroup-knn", str(args.regroup_knn),
        "--assignment-chunk-size", str(args.assignment_chunk_size),
        "--faiss-threads", str(args.faiss_threads),
    ]
    if args.overwrite:
        command.append("--overwrite")
    subprocess.run(command, check=True)

    config_path = output / "run_config.json"
    config = json.loads(config_path.read_text())
    actual_quota = np.asarray(config["nodes_per_stratum"], dtype=np.int32)
    if not np.array_equal(actual_quota, quota):
        raise RuntimeError("fixed-count replay did not reproduce adaptive quotas")
    config.update(
        selection_mode="adaptive_marginal_gain_threshold",
        min_marginal_gain=args.min_marginal_gain,
        min_cells_per_stratum=args.min_cells_per_stratum,
        max_nodes_safety_cap=args.max_nodes,
        adaptive_last_accepted_gain=last_gain,
        adaptive_stop_reason=stop_reason,
        adaptive_selected_nodes=target_nodes,
        method="adaptive_gain_threshold_then_weighted_graph_partition",
    )
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")

    final_nodes = pd.read_parquet(output / "nodes.parquet")
    lineage_counts = final_nodes.groupby("lineage", observed=True).size().sort_index()
    print("\nFinal landmark-node summary", flush=True)
    print(f"total_nodes: {len(final_nodes):,}", flush=True)
    print("nodes_by_lineage:", flush=True)
    for lineage, count in lineage_counts.items():
        print(f"  {lineage}: {int(count):,}", flush=True)


if __name__ == "__main__":
    main()
