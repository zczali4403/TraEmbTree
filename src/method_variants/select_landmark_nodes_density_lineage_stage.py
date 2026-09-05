#!/usr/bin/env python3
# Full-data example: estimate density from every cell in each lineage x stage stratum.
#
#   OPENBLAS_NUM_THREADS=32 OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 \
#   /home/aiscuser/.conda/envs/train/bin/python \
#     /mnt/input/sc_cz/Concord/eval/2026_08_19/select_landmark_nodes_density_lineage_stage.py \
#     --target-nodes 2000 \
#     --allocation-strategy marginal_gain \
#     --min-cells-per-node 200 \
#     --density-use-all \
#     --density-index auto \
#     --faiss-threads 32 \
#     --output-dir /mnt/input/sc_cz/Concord/eval/2026_08_19/landmark_nodes_density_lineage_stage_all \
#     --overwrite
#
"""Select density-aware TraEmb landmark nodes within lineage x exact-stage strata.

The method is designed for the full 15M-cell dataset.  In every non-empty
lineage/stage stratum it:

1. estimates local density for all cells using cosine kNN;
2. gives each stratum one node, then allocates extra nodes by normalized mean
   marginal coverage gain rather than observed cell count;
3. chooses density-weighted, well-separated peaks;
4. assigns every cell in the stratum to its nearest peak in bounded batches;
5. stores the arithmetic mean of the original TraEmb vectors as the final node.

This is a density-peak landmark method, not an MC2/metacell implementation.
Outputs are compatible with ``build_landmark_tree.py``.

Example
-------
OPENBLAS_NUM_THREADS=32 OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 \
/home/aiscuser/.conda/envs/train/bin/python \
  select_landmark_nodes_density_lineage_stage.py \
  --target-nodes 2000 --output-dir landmark_nodes_density_lineage_stage \
  --overwrite
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "32")

import argparse
import heapq
import json
import math
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--embeddings", type=Path, default=Path("/mnt/input/sc_cz/Concord/eval/2026_09_03/embeddings.npy"))
    p.add_argument("--csr-dir", type=Path, default=Path("/scratch/amlt_code/traemb_csr_0829"))
    p.add_argument("--metadata", type=Path, default=HERE / "all_lineage_260811_liver_reanno.csv")
    p.add_argument("--output-dir", type=Path, default=HERE / "landmark_nodes_density_lineage_stage")
    p.add_argument("--target-nodes", type=int, default=2000)
    p.add_argument("--min-nodes-per-stratum", type=int, default=1)
    p.add_argument("--quota-power", type=float, default=0.5)
    p.add_argument("--allocation-strategy", choices=("marginal_gain", "cell_count"),
                   default="marginal_gain", help="Allocate extra nodes by normalized coverage gain or cell count.")
    p.add_argument("--allocation-log-every", type=int, default=25)
    p.add_argument("--min-cells-per-node", type=int, default=200,
                   help="Confidence/cap constraint for extra nodes; every nonempty stratum still gets one.")
    p.add_argument("--density-knn", type=int, default=30,
                   help="Neighbors used to estimate sample density.")
    p.add_argument("--density-sample-min", type=int, default=5000,
                   help="Preferred minimum density sample for a sufficiently large stratum.")
    p.add_argument("--density-sample-max", type=int, default=100000,
                   help="Hard sample cap per stratum; requested node count may exceed it.")
    p.add_argument("--density-use-all", action="store_true",
                   help="Estimate density for every cell in every stratum (no density sampling).")
    p.add_argument("--density-candidates-per-node", type=int, default=200,
                   help="Preferred sampled candidates per requested node.")
    p.add_argument("--density-power", type=float, default=1.0,
                   help="Strength of density relative to peak separation; 0 is pure farthest-point selection.")
    p.add_argument("--density-index", choices=("auto", "flat", "hnsw"), default="auto",
                   help="kNN index for density estimation; auto uses exact flat only on small inputs.")
    p.add_argument("--density-flat-max-cells", type=int, default=100000,
                   help="In auto mode, use exact flat kNN up to this many density candidates.")
    p.add_argument("--hnsw-m", type=int, default=32, help="HNSW graph degree in full-density mode.")
    p.add_argument("--hnsw-ef-construction", type=int, default=100)
    p.add_argument("--hnsw-ef-search", type=int, default=128)
    p.add_argument("--density-query-chunk-size", type=int, default=50000)
    p.add_argument("--assignment-chunk-size", type=int, default=100000)
    p.add_argument("--metadata-chunk-size", type=int, default=500000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--faiss-threads", type=int, default=32)
    p.add_argument("--max-cells", type=int, default=None, help="Testing only: use the first N aligned rows.")
    p.add_argument("--skip-celltype", action="store_true",
                   help="Skip the expensive metadata join (useful for a quick test).")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def normalize(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float32).copy()
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    return out


def allocate_capped(counts: np.ndarray, total: int, minimum: int, power: float) -> np.ndarray:
    """Allocate exactly total nodes, never assigning more nodes than cells."""
    counts = np.asarray(counts, dtype=np.int64)
    active = counts > 0
    n_active = int(active.sum())
    if total < n_active:
        raise ValueError(f"target-nodes={total} is below {n_active} nonempty strata")
    if total > int(counts.sum()):
        raise ValueError(f"target-nodes={total} exceeds the number of cells ({int(counts.sum())})")
    quota = np.zeros(len(counts), dtype=np.int64)
    quota[active] = np.minimum(counts[active], max(1, int(minimum)))
    if int(quota.sum()) > total:
        quota[active] = 1
    remaining = total - int(quota.sum())
    weights = np.where(active, counts.astype(np.float64) ** power, 0.0)
    while remaining:
        room = counts - quota
        eligible = room > 0
        if not eligible.any():
            raise RuntimeError("node quota allocation exhausted all strata")
        w = np.where(eligible, weights, 0.0)
        raw = remaining * w / w.sum()
        add = np.minimum(room, np.floor(raw).astype(np.int64))
        gained = int(add.sum())
        quota += add
        remaining -= gained
        if remaining and gained == 0:
            fractions = raw - np.floor(raw)
            candidates = np.flatnonzero(eligible)
            order = candidates[np.argsort(-fractions[candidates], kind="stable")]
            take = min(remaining, len(order))
            quota[order[:take]] += 1
            remaining -= take
    return quota.astype(np.int32)


def sample_size(n_cells: int, n_nodes: int, args: argparse.Namespace) -> int:
    if args.density_use_all:
        return n_cells
    preferred = max(args.density_sample_min, n_nodes * args.density_candidates_per_node)
    return min(n_cells, max(n_nodes, min(args.density_sample_max, preferred)))


def density_peak_indices(vectors: np.ndarray, n_peaks: int, knn: int,
                         density_power: float, index_kind: str = "flat",
                         query_chunk_size: int = 50000, hnsw_m: int = 32,
                         hnsw_ef_construction: int = 100,
                         hnsw_ef_search: int = 128) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sample-row peaks, kNN densities, and selected peak densities."""
    import faiss

    z = normalize(vectors)
    n = len(z)
    if not 1 <= n_peaks <= n:
        raise ValueError(f"invalid n_peaks={n_peaks} for sample size {n}")
    if n == 1:
        return np.array([0], np.int64), np.ones(1), np.ones(1)

    k = min(max(1, knn), n - 1)
    if index_kind == "flat":
        index = faiss.IndexFlatIP(z.shape[1])
    elif index_kind == "hnsw":
        index = faiss.IndexHNSWFlat(z.shape[1], hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = hnsw_ef_construction
        index.hnsw.efSearch = hnsw_ef_search
    else:
        raise ValueError(f"unsupported density index: {index_kind}")
    index.add(z)
    density = np.empty(n, dtype=np.float64)
    for lo in range(0, n, query_chunk_size):
        hi = min(n, lo + query_chunk_size)
        similarity, neighbors = index.search(z[lo:hi], k + 1)
        row_ids = np.arange(lo, hi, dtype=np.int64)
        local_mean = np.empty(hi - lo, dtype=np.float64)
        for row in range(hi - lo):
            keep = neighbors[row] != row_ids[row]
            valid_similarity = similarity[row, keep][:k]
            if len(valid_similarity) < k:
                valid_similarity = similarity[row, :k]
            local_mean[row] = np.maximum(0.0, 1.0 - valid_similarity).mean()
        density[lo:hi] = 1.0 / np.maximum(local_mean, 1e-8)
    # Rank scaling prevents a few near-duplicate points from dominating selection.
    order = np.argsort(density, kind="stable")
    density_rank = np.empty(n, dtype=np.float64)
    density_rank[order] = (np.arange(n, dtype=np.float64) + 1.0) / n

    selected = np.empty(n_peaks, dtype=np.int64)
    selected[0] = int(np.argmax(density))
    nearest_distance = np.maximum(0.0, 1.0 - z @ z[selected[0]])
    chosen = np.zeros(n, dtype=bool)
    chosen[selected[0]] = True
    for j in range(1, n_peaks):
        score = nearest_distance * np.power(density_rank, density_power)
        score[chosen] = -np.inf
        selected[j] = int(np.argmax(score))
        chosen[selected[j]] = True
        distance = np.maximum(0.0, 1.0 - z @ z[selected[j]])
        nearest_distance = np.minimum(nearest_distance, distance)
    return selected, density, density[selected]

def allocate_peaks_by_marginal_gain(embeddings, stratum_order, unique_keys, starts,
                                    counts, total_nodes, args):
    """Allocate nodes by per-cell mean coverage improvement, not population size."""
    n_strata = len(unique_keys)
    if total_nodes < n_strata:
        raise ValueError(f"target-nodes={total_nodes} is below {n_strata} nonempty strata")
    if total_nodes > int(np.sum(counts)):
        raise ValueError(f"target-nodes={total_nodes} exceeds the number of cells")

    states = []
    heap = []

    def propose(stratum_index):
        state = states[stratum_index]
        capacity = max(1, len(state["ids"]) // args.min_cells_per_node)
        if len(state["peaks"]) >= min(len(state["ids"]), capacity):
            state["candidate"] = None
            state["gain"] = -np.inf
            return
        z = normalize(embeddings[state["ids"]])
        score = state["nearest"].astype(np.float64) * np.power(
            state["density_rank"], args.density_power)
        score[np.asarray(state["peaks"], dtype=np.int64)] = -np.inf
        candidate = int(np.argmax(score))
        candidate_distance = np.maximum(0.0, 1.0 - z @ z[candidate]).astype(np.float32)
        improvement = np.maximum(0.0, state["nearest"] - candidate_distance)
        gain = float(improvement.mean())
        state["candidate"] = candidate
        state["candidate_distance"] = candidate_distance
        state["gain"] = gain
        heapq.heappush(heap, (-gain, stratum_index, len(state["peaks"])))

    log(f"preparing all-cell densities for marginal-gain allocation across {n_strata:,} strata")
    for stratum_index, (start, count) in enumerate(zip(starts, counts)):
        ids = stratum_order[int(start):int(start + count)]
        index_kind = args.density_index
        if index_kind == "auto":
            index_kind = "flat" if len(ids) <= args.density_flat_max_cells else "hnsw"
        log(f"  density {stratum_index + 1}/{n_strata}: cells={len(ids):,}; index={index_kind}")
        _, density, _ = density_peak_indices(
            embeddings[ids], 1, args.density_knn, args.density_power,
            index_kind=index_kind, query_chunk_size=args.density_query_chunk_size,
            hnsw_m=args.hnsw_m, hnsw_ef_construction=args.hnsw_ef_construction,
            hnsw_ef_search=args.hnsw_ef_search)
        density = np.asarray(density, dtype=np.float32)
        density_order = np.argsort(density, kind="stable")
        density_rank = np.empty(len(ids), dtype=np.float32)
        density_rank[density_order] = (
            (np.arange(len(ids), dtype=np.float32) + 1.0) / len(ids))
        first = int(np.argmax(density))
        z = normalize(embeddings[ids])
        nearest = np.maximum(0.0, 1.0 - z @ z[first]).astype(np.float32)
        states.append(dict(ids=ids, density=density, density_rank=density_rank,
                           nearest=nearest, peaks=[first],
                           peak_density=[float(density[first])],
                           candidate=None, candidate_distance=None, gain=-np.inf))
        propose(stratum_index)

    allocated = n_strata
    while allocated < total_nodes:
        while heap:
            negative_gain, stratum_index, version = heapq.heappop(heap)
            state = states[stratum_index]
            if version == len(state["peaks"]) and state["candidate"] is not None:
                break
        else:
            raise RuntimeError("no stratum can accept the remaining landmark nodes")
        candidate = int(state["candidate"])
        gain = -float(negative_gain)
        state["peaks"].append(candidate)
        state["peak_density"].append(float(state["density"][candidate]))
        state["nearest"] = np.minimum(
            state["nearest"], state["candidate_distance"]).astype(np.float32)
        state["candidate"] = None
        state["candidate_distance"] = None
        allocated += 1
        if allocated % args.allocation_log_every == 0 or allocated == total_nodes:
            log(f"marginal allocation {allocated:,}/{total_nodes:,}: "
                f"stratum={stratum_index}; nodes_in_stratum={len(state['peaks'])}; "
                f"mean_gain={gain:.6g}")
        propose(stratum_index)

    peak_rows = [
        np.asarray(state["ids"][np.asarray(state["peaks"], dtype=np.int64)], dtype=np.int64)
        for state in states
    ]
    peak_densities = [
        np.asarray(state["peak_density"], dtype=np.float64) for state in states
    ]
    quota = np.asarray([len(state["peaks"]) for state in states], dtype=np.int32)
    return quota, peak_rows, peak_densities



def hash_strings(values: np.ndarray) -> np.ndarray:
    import pandas as pd
    return pd.util.hash_array(np.asarray(values, dtype=object), categorize=False).astype(np.uint64, copy=False)


def aggregate_celltypes(metadata: Path, cell_ids: np.ndarray, assignment: np.ndarray,
                        n_nodes: int, chunk_size: int):
    import pandas as pd

    n = len(cell_ids)
    log(f"building cell-ID hash index for {n:,} embedding rows")
    hashes = np.empty(n, np.uint64)
    for lo in range(0, n, chunk_size):
        hi = min(n, lo + chunk_size)
        hashes[lo:hi] = hash_strings(cell_ids[lo:hi])
    order = np.argsort(hashes)
    sorted_hashes = hashes[order]
    counts = defaultdict(int)
    matched = rows_seen = 0
    for frame in pd.read_csv(metadata, usecols=["cell", "celltype"], chunksize=chunk_size):
        rows_seen += len(frame)
        h = hash_strings(frame["cell"].astype(str).to_numpy())
        pos = np.searchsorted(sorted_hashes, h)
        ok = pos < n
        ok &= sorted_hashes[np.minimum(pos, n - 1)] == h
        rows = order[pos[ok]]
        celltypes = frame.loc[ok, "celltype"].fillna("Unknown").astype(str).to_numpy()
        for node, celltype in zip(np.asarray(assignment[rows]), celltypes):
            counts[(int(node), celltype)] += 1
        matched += len(rows)
        log(f"celltype matched {matched:,}/{rows_seen:,}")

    by_node = defaultdict(list)
    for (node, celltype), count in counts.items():
        by_node[node].append((celltype, count))
    summaries, composition = [], []
    for node in range(n_nodes):
        values = sorted(by_node[node], key=lambda x: (-x[1], x[0]))
        total = sum(count for _, count in values)
        for rank, (celltype, count) in enumerate(values, 1):
            composition.append(dict(node_id=node, celltype=celltype, cell_count=count,
                                    fraction=count / total, rank=rank))
        dominant, dominant_count = values[0] if values else ("Unknown", 0)
        second = values[1][0] if len(values) > 1 else ""
        entropy = -sum((count / total) * math.log(count / total) for _, count in values) if total else np.nan
        summaries.append(dict(node_id=node, dominant_celltype=dominant,
                              celltype_purity=dominant_count / total if total else np.nan,
                              dominant_celltype_count=dominant_count, second_celltype=second,
                              celltype_entropy=entropy, celltype_types=len(values),
                              celltype_matched_cells=total))
    return summaries, composition, matched, rows_seen


def main() -> None:
    args = parse_args()
    import faiss
    import pandas as pd

    if args.target_nodes <= 0 or args.density_knn <= 0 or args.assignment_chunk_size <= 0:
        raise ValueError("target-nodes, density-knn, and assignment-chunk-size must be positive")
    if args.density_sample_min <= 0 or args.density_sample_max <= 0:
        raise ValueError("density sample sizes must be positive")
    if args.min_cells_per_node <= 0:
        raise ValueError("min-cells-per-node must be positive")
    faiss.omp_set_num_threads(args.faiss_threads)
    output = args.output_dir.resolve()
    temporary = output.with_name(f".{output.name}.building-{os.getpid()}")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; use --overwrite")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    try:
        embeddings = np.load(args.embeddings, mmap_mode="r")
        full_n, dimension = embeddings.shape
        n = min(full_n, args.max_cells) if args.max_cells else full_n
        metadata_info = json.loads((args.csr_dir / "metadata.json").read_text())
        lineage_names = list(metadata_info["lineages"])
        stages = np.asarray(metadata_info["stages"], dtype=np.float32)
        lineage_ids = np.load(args.csr_dir / "lineage_ids.npy", mmap_mode="r")[:n]
        stage_ids = np.load(args.csr_dir / "stage_ids.npy", mmap_mode="r")[:n]
        cell_ids = np.load(args.csr_dir / "cell_ids.npy", mmap_mode="r")[:n]
        if not (len(lineage_ids) == len(stage_ids) == len(cell_ids) == n):
            raise ValueError("CSR labels, cell IDs, and embeddings are not row-aligned")
        if n == 0:
            raise ValueError("no input cells")

        n_stages = len(stages)
        keys = np.asarray(lineage_ids, np.int64) * n_stages + np.asarray(stage_ids, np.int64)
        stratum_order = np.argsort(keys, kind="stable")
        sorted_keys = keys[stratum_order]
        unique_keys, starts, counts = np.unique(sorted_keys, return_index=True, return_counts=True)
        preselected_rows = preselected_density = None
        if args.allocation_strategy == "marginal_gain":
            quota, preselected_rows, preselected_density = allocate_peaks_by_marginal_gain(
                embeddings, stratum_order, unique_keys, starts, counts,
                args.target_nodes, args)
        else:
            quota = allocate_capped(counts, args.target_nodes, args.min_nodes_per_stratum, args.quota_power)
        offsets = np.r_[0, np.cumsum(quota, dtype=np.int64)]
        assignment = np.lib.format.open_memmap(temporary / "cell_to_node.npy", mode="w+",
                                                dtype=np.int32, shape=(n,))
        distances = np.lib.format.open_memmap(temporary / "cell_to_node_distance.npy", mode="w+",
                                               dtype=np.float32, shape=(n,))
        assignment[:] = -1
        distances[:] = np.nan

        sums = np.zeros((args.target_nodes, dimension), np.float64)
        sizes = np.zeros(args.target_nodes, np.int64)
        distance_sum = np.zeros(args.target_nodes, np.float64)
        distance_p95_parts = [[] for _ in range(args.target_nodes)]
        representatives = np.full(args.target_nodes, -1, np.int64)
        representative_distance = np.full(args.target_nodes, np.inf)
        peak_density = np.empty(args.target_nodes, np.float64)
        rng = np.random.default_rng(args.seed)

        log(f"input={n:,} x {dimension}; strata={len(unique_keys):,}; final nodes={args.target_nodes:,}")
        for stratum_index, (key, start, count, q) in enumerate(zip(unique_keys, starts, counts, quota)):
            lineage_id = int(key // n_stages)
            stage_id = int(key % n_stages)
            ids = stratum_order[int(start):int(start + count)]
            q = int(q)
            if preselected_rows is not None:
                peak_rows = preselected_rows[stratum_index]
                selected_density = preselected_density[stratum_index]
                density_description = "all; allocated_by=marginal_gain"
            else:
                take = sample_size(len(ids), q, args)
                sample = ids if take == len(ids) else rng.choice(ids, size=take, replace=False)
                density_index = args.density_index
                if density_index == "auto":
                    density_index = "flat" if take <= args.density_flat_max_cells else "hnsw"
                selected, _, selected_density = density_peak_indices(
                    embeddings[sample], q, args.density_knn, args.density_power,
                    index_kind=density_index, query_chunk_size=args.density_query_chunk_size,
                    hnsw_m=args.hnsw_m, hnsw_ef_construction=args.hnsw_ef_construction,
                    hnsw_ef_search=args.hnsw_ef_search)
                peak_rows = np.asarray(sample[selected], dtype=np.int64)
                density_description = f"{take:,}; index={density_index}"
            log(f"stratum {stratum_index + 1}/{len(unique_keys)}: lineage={lineage_names[lineage_id]}; "
                f"stage={float(stages[stage_id]):g}; cells={len(ids):,}; "
                f"density_points={density_description}; nodes={q}")
            centers = normalize(embeddings[peak_rows])
            index = faiss.IndexFlatIP(dimension)
            index.add(centers)
            node_offset = int(offsets[stratum_index])
            peak_density[node_offset:node_offset + q] = selected_density

            for lo in range(0, len(ids), args.assignment_chunk_size):
                rows = ids[lo:lo + args.assignment_chunk_size]
                raw = np.asarray(embeddings[rows], dtype=np.float32)
                similarity, local = index.search(normalize(raw), 1)
                local = local[:, 0].astype(np.int32)
                node = local + node_offset
                distance = np.maximum(0.0, 1.0 - similarity[:, 0]).astype(np.float32)
                assignment[rows] = node
                distances[rows] = distance
                np.add.at(sizes, node, 1)
                np.add.at(sums, node, raw)
                np.add.at(distance_sum, node, distance)
                for local_node in np.unique(local):
                    mask = local == local_node
                    global_node = node_offset + int(local_node)
                    distance_p95_parts[global_node].append(distance[mask].copy())
                    candidate = int(np.argmin(distance[mask]))
                    candidate_rows = rows[mask]
                    candidate_distance = float(distance[mask][candidate])
                    if candidate_distance < representative_distance[global_node]:
                        representative_distance[global_node] = candidate_distance
                        representatives[global_node] = int(candidate_rows[candidate])

        if np.any(assignment < 0) or np.any(sizes == 0):
            raise RuntimeError(f"assignment failed: unassigned={np.count_nonzero(assignment < 0)}, empty_nodes={np.count_nonzero(sizes == 0)}")
        node_embeddings = (sums / sizes[:, None]).astype(np.float32)
        np.save(temporary / "node_embeddings.npy", node_embeddings)

        if args.skip_celltype:
            summaries = [dict(node_id=i, dominant_celltype="Unknown", celltype_purity=np.nan,
                              dominant_celltype_count=0, second_celltype="", celltype_entropy=np.nan,
                              celltype_types=0, celltype_matched_cells=0)
                         for i in range(args.target_nodes)]
            composition = []
            matched = metadata_rows = 0
        else:
            summaries, composition, matched, metadata_rows = aggregate_celltypes(
                args.metadata, cell_ids, assignment, args.target_nodes, args.metadata_chunk_size)
        summary_by_node = {row["node_id"]: row for row in summaries}

        node_rows = []
        for stratum_index, (key, q) in enumerate(zip(unique_keys, quota)):
            lineage_id = int(key // n_stages)
            stage_id = int(key % n_stages)
            stage = float(stages[stage_id])
            for node in range(int(offsets[stratum_index]), int(offsets[stratum_index + 1])):
                p95 = float(np.quantile(np.concatenate(distance_p95_parts[node]), 0.95))
                row = dict(node_id=node, lineage=lineage_names[lineage_id], lineage_id=lineage_id,
                           constraint_stage_id=stage_id, constraint_stage=stage,
                           n_cells=int(sizes[node]), mean_stage=stage, stage_std=0.0,
                           min_stage=stage, max_stage=stage,
                           representative_idx=int(representatives[node]),
                           representative_cell_id=str(cell_ids[representatives[node]]),
                           peak_sample_density=float(peak_density[node]),
                           mean_cosine_distance=float(distance_sum[node] / sizes[node]),
                           p95_cosine_distance=p95)
                row.update(summary_by_node[node])
                node_rows.append(row)
        pd.DataFrame(node_rows).to_parquet(temporary / "nodes.parquet", index=False)
        composition_columns = ["node_id", "celltype", "cell_count", "fraction", "rank"]
        pd.DataFrame(composition, columns=composition_columns).to_parquet(
            temporary / "node_celltype_composition.parquet", index=False)

        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        config.update(n_cells=n, n_nodes=len(node_rows), metadata_rows=metadata_rows,
                      celltype_matched=matched, n_nonempty_lineage_stage_strata=len(unique_keys),
                      stage_constraint="exact_stage_hard",
                      density_population="all_cells" if args.allocation_strategy == "marginal_gain" or args.density_use_all else "sampled_cells",
                      allocation_objective="normalized_mean_marginal_coverage_gain" if args.allocation_strategy == "marginal_gain" else "cell_count_power",
                      method="density_weighted_cosine_knn_peaks_marginal_allocation_lineage_exact_stage_hard")
        (temporary / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
        del assignment, distances
        if output.exists():
            shutil.rmtree(output)
        temporary.rename(output)
        log(f"done: {output}")
    except Exception:
        log(f"failed; partial outputs kept: {temporary}")
        raise
