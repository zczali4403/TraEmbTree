#!/usr/bin/env python3
"""Discover candidate trajectory nodes independently within known lineages.

Within each lineage, Scanpy cosine neighbors and Leiden use embeddings only.
Cell-type labels never enter clustering. After Leiden, predicted stage splits
each embedding state into temporal nodes using either fixed-width bins or
adaptive cell-weighted KDE valleys with broad-segment median splitting. This
script does not connect nodes and does not construct a tree.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--knn", type=int, default=30)
    parser.add_argument("--leiden-resolution", type=float, default=0.5)
    parser.add_argument("--leiden-iterations", type=int, default=2)
    parser.add_argument(
        "--temporal-split-method", choices=("fixed", "kde"), default="fixed",
        help="Fixed-width bins or adaptive weighted stage-density segmentation.")
    parser.add_argument("--stage-bin-width", type=float, default=2.0)
    parser.add_argument(
        "--stage-kde-bandwidth", type=float, default=0.5,
        help="Gaussian KDE bandwidth in predicted-stage units for adaptive splitting.")
    parser.add_argument(
        "--stage-min-peak-distance", type=float, default=1.0,
        help="Minimum stage separation between KDE peaks.")
    parser.add_argument(
        "--stage-valley-ratio", type=float, default=0.6,
        help="Accept a valley when its density is at most this fraction of the lower adjacent peak.")
    parser.add_argument(
        "--max-node-stage-span", type=float, default=3.0,
        help="KDE segments wider than this are recursively split at a cell-weighted median; 0 disables.")
    parser.add_argument(
        "--min-node-cells", type=int, default=50,
        help="Minimum cells for adaptive segments and the final node support filter.")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot-max-points", type=int, default=400_000)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args(argv)


def load_inputs(root: Path):
    metadata = (pd.read_parquet(root / "microcells.parquet")
                .sort_values("microcell_id").reset_index(drop=True))
    embedding = np.load(root / "microcell_embeddings.npy", mmap_mode="r")
    required = {"microcell_id", "n_cells", "mean_stage", "stage_source", "lineage"}
    if not required.issubset(metadata.columns):
        raise ValueError(f"Missing metadata columns: {sorted(required - set(metadata.columns))}")
    if embedding.ndim != 2 or len(metadata) != len(embedding):
        raise ValueError("microcell metadata and embedding rows are not aligned")
    if not np.array_equal(metadata.microcell_id.to_numpy(), np.arange(len(metadata))):
        raise ValueError("microcell_id must cover embedding rows 0..N-1")
    if not metadata.stage_source.eq("predicted_stage").all():
        raise ValueError("Every mean_stage must come from predicted_stage")
    if metadata.lineage.isna().any() or metadata.lineage.astype(str).str.strip().eq("").any():
        raise ValueError("Every metacell must have a nonempty lineage")
    if (metadata.n_cells.to_numpy() <= 0).any():
        raise ValueError("n_cells must be positive")
    if not np.isfinite(metadata.mean_stage.to_numpy(dtype=float)).all():
        raise ValueError("mean_stage must be finite")
    for start in range(0, len(embedding), 20_000):
        if not np.isfinite(embedding[start:start + 20_000]).all():
            raise ValueError("Embedding contains nonfinite values")
    return metadata, embedding


def scanpy_connectivities(vectors, k: int, threads: int, seed: int):
    import anndata as ad
    import scanpy as sc

    n = len(vectors)
    if n < 3:
        raise ValueError("Each lineage must contain at least three metacells")
    effective_k = min(k, n - 1)
    data = np.array(vectors, dtype=np.float32, order="C", copy=True)
    adata = ad.AnnData(X=data)
    sc.settings.n_jobs = threads
    sc.pp.neighbors(
        adata, n_neighbors=effective_k, use_rep="X", knn=True,
        method="umap", transformer=None, metric="cosine", random_state=seed)
    graph = adata.obsp["connectivities"].astype(np.float32).tocsr()
    directed_entries = int(adata.obsp["distances"].nnz)
    graph.setdiag(0)
    graph.eliminate_zeros()
    graph.sort_indices()
    if (graph - graph.T).nnz:
        raise ValueError("Scanpy fuzzy connectivities must be symmetric")
    components, component_id = connected_components(graph, directed=False)
    degrees = np.diff(graph.indptr)
    summary = {
        "n_metacells": n,
        "effective_knn": effective_k,
        "n_directed_knn_entries": directed_entries,
        "n_undirected_fuzzy_edges": int(graph.nnz // 2),
        "n_components": int(components),
        "n_isolated": int(np.count_nonzero(degrees == 0)),
        "largest_component": int(np.bincount(component_id).max()),
    }
    return graph, summary


def leiden_membership(adjacency, resolution: float, iterations: int, seed: int):
    import igraph as ig
    import leidenalg

    graph = ig.Graph.Weighted_Adjacency(
        adjacency, mode="undirected", attr="weight", loops=False)
    partition = leidenalg.find_partition(
        graph, leidenalg.RBConfigurationVertexPartition,
        weights="weight", resolution_parameter=resolution,
        n_iterations=iterations, seed=seed)
    return np.asarray(partition.membership, dtype=np.int32)


def discover_lineage_states(metadata, embedding, k, resolution, iterations,
                            threads, seed):
    lineage_values = metadata.lineage.astype(str).to_numpy()
    state_within = np.full(len(metadata), -1, dtype=np.int32)
    state_id = np.full(len(metadata), -1, dtype=np.int32)
    summaries = []
    offset = 0
    lineages = sorted(pd.unique(lineage_values))
    for position, lineage in enumerate(lineages, 1):
        members = np.flatnonzero(lineage_values == lineage)
        print(f"[{position}/{len(lineages)}] {lineage}: "
              f"Scanpy cosine kNN for {len(members):,} metacells", flush=True)
        adjacency, summary = scanpy_connectivities(
            embedding[members], k, threads, seed + position)
        local = leiden_membership(
            adjacency, resolution, iterations, seed + position)
        n_states = int(local.max()) + 1
        state_within[members] = local
        state_id[members] = local + offset
        offset += n_states
        summary.update(lineage=lineage, n_states=n_states)
        summaries.append(summary)
        print(f"    {n_states:,} embedding states; {summary['n_components']:,} graph components; "
              f"{summary['n_isolated']:,} isolated", flush=True)
    if (state_id < 0).any():
        raise RuntimeError("Some metacells were not assigned to a lineage state")
    return state_within, state_id, pd.DataFrame(summaries)


def make_temporal_nodes(state_id, stage, width):
    if not np.isfinite(width) or width <= 0:
        raise ValueError("--stage-bin-width must be positive and finite")
    stage_bin = np.floor(np.asarray(stage, dtype=float) / width).astype(np.int32)
    pairs = pd.MultiIndex.from_arrays([state_id, stage_bin])
    node_id, _ = pd.factorize(pairs, sort=True)
    return stage_bin, node_id.astype(np.int32)


def relabel_temporal_segments(labels, stage):
    """Renumber one-dimensional segments in increasing stage order."""
    labels = np.asarray(labels)
    order = sorted(
        np.unique(labels),
        key=lambda label: (
            float(np.min(stage[labels == label])), int(label)))
    mapping = {label: index for index, label in enumerate(order)}
    return np.asarray([mapping[label] for label in labels], dtype=np.int32)


def merge_small_temporal_segments(labels, stage, weights, min_cells):
    """Merge undersupported contiguous segments into the closest neighbor."""
    labels = relabel_temporal_segments(labels, stage)
    merges = 0
    while len(np.unique(labels)) > 1:
        rows = []
        for label in np.unique(labels):
            members = labels == label
            total = float(weights[members].sum())
            mean = float(np.average(stage[members], weights=weights[members]))
            rows.append((int(label), total, mean))
        deficient = [row for row in rows if row[1] < min_cells]
        if not deficient:
            break
        label, _, mean = min(
            deficient, key=lambda row: (row[1], row[0]))
        row_lookup = {row[0]: row for row in rows}
        neighbors = [
            neighbor for neighbor in (label - 1, label + 1)
            if neighbor in row_lookup
        ]
        target = min(
            neighbors,
            key=lambda neighbor: (
                abs(mean - row_lookup[neighbor][2]),
                -row_lookup[neighbor][1], neighbor))
        labels[labels == label] = target
        labels = relabel_temporal_segments(labels, stage)
        merges += 1
    return labels, merges


def split_wide_temporal_segments(labels, stage, weights, max_span,
                                 min_cells):
    """Recursively split broad segments at feasible cell-weighted medians."""
    labels = relabel_temporal_segments(labels, stage)
    splits = 0
    if max_span <= 0:
        return labels, splits
    while True:
        split_applied = False
        for label in np.unique(labels):
            members = np.flatnonzero(labels == label)
            local_stage = stage[members]
            if float(local_stage.max() - local_stage.min()) <= max_span:
                continue
            order = np.argsort(local_stage, kind="stable")
            ordered_members = members[order]
            ordered_stage = stage[ordered_members]
            ordered_weights = weights[ordered_members]
            cumulative_cells = np.cumsum(ordered_weights)
            total_cells = float(cumulative_cells[-1])
            candidates = np.flatnonzero(ordered_stage[:-1] < ordered_stage[1:])
            valid = candidates[
                (cumulative_cells[candidates] >= min_cells)
                & (total_cells - cumulative_cells[candidates] >= min_cells)
            ]
            if not len(valid):
                continue
            cut = int(min(
                valid,
                key=lambda position: (
                    abs(float(cumulative_cells[position]) - total_cells / 2),
                    float(ordered_stage[position]), position)))
            new_label = int(labels.max()) + 1
            labels[ordered_members[cut + 1:]] = new_label
            labels = relabel_temporal_segments(labels, stage)
            splits += 1
            split_applied = True
            break
        if not split_applied:
            break
    return labels, splits


def kde_temporal_segments(stage, weights, bandwidth, min_peak_distance,
                          valley_ratio, max_span, min_cells):
    """Split one Leiden state into contiguous weighted stage-density basins."""
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    stage = np.asarray(stage, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    stage_min, stage_max = float(stage.min()), float(stage.max())
    accepted_valleys = []
    peak_count = 1
    if len(np.unique(stage)) < 3 or stage_max <= stage_min:
        labels = np.zeros(len(stage), dtype=np.int32)
    else:
        target_step = min(0.1, max(0.01, bandwidth / 8))
        grid_size = int(np.clip(
            np.ceil((stage_max - stage_min) / target_step) + 1,
            32, 4096))
        edges = np.linspace(stage_min, stage_max, grid_size + 1)
        grid = (edges[:-1] + edges[1:]) / 2
        step = float(edges[1] - edges[0])
        histogram, _ = np.histogram(stage, bins=edges, weights=weights)
        density = gaussian_filter1d(
            histogram.astype(np.float64),
            sigma=max(bandwidth / step, 1e-6), mode="nearest")
        distance_bins = max(1, int(np.ceil(min_peak_distance / step)))
        peaks = find_peaks(density, distance=distance_bins)[0].tolist()
        if len(density) > 1 and density[0] > density[1]:
            peaks.append(0)
        if len(density) > 1 and density[-1] > density[-2]:
            peaks.append(len(density) - 1)
        peaks = sorted(set(peaks))
        peak_count = len(peaks)
        for left, right in zip(peaks[:-1], peaks[1:]):
            if grid[right] - grid[left] < min_peak_distance:
                continue
            valley = left + int(np.argmin(density[left:right + 1]))
            lower_peak = min(float(density[left]), float(density[right]))
            ratio = (float(density[valley]) / lower_peak
                     if lower_peak > 0 else 1.0)
            if ratio <= valley_ratio:
                accepted_valleys.append(float(grid[valley]))
        labels = np.searchsorted(
            np.asarray(accepted_valleys), stage, side="right").astype(np.int32)

    segments_before_merge = int(len(np.unique(labels)))
    labels, small_merges = merge_small_temporal_segments(
        labels, stage, weights, min_cells)
    labels, median_splits = split_wide_temporal_segments(
        labels, stage, weights, max_span, min_cells)
    diagnostics = {
        "stage_min": stage_min,
        "stage_max": stage_max,
        "stage_span": stage_max - stage_min,
        "n_unique_stage_values": int(len(np.unique(stage))),
        "n_kde_peaks": int(peak_count),
        "n_accepted_valleys": int(len(accepted_valleys)),
        "accepted_valley_stages": json.dumps(accepted_valleys),
        "n_segments_before_small_merge": segments_before_merge,
        "n_small_segment_merges": int(small_merges),
        "n_weighted_median_splits": int(median_splits),
        "n_segments_final": int(len(np.unique(labels))),
    }
    return labels, diagnostics


def make_kde_temporal_nodes(state_id, stage, weights, bandwidth,
                            min_peak_distance, valley_ratio, max_span,
                            min_cells):
    """Adaptively split every state and return globally contiguous node IDs."""
    state_id = np.asarray(state_id, dtype=np.int32)
    stage = np.asarray(stage, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    segment = np.full(len(stage), -1, dtype=np.int32)
    interval_left = np.full(len(stage), np.nan, dtype=np.float64)
    interval_right = np.full(len(stage), np.nan, dtype=np.float64)
    diagnostic_rows = []
    for state in np.unique(state_id):
        members = np.flatnonzero(state_id == state)
        local_labels, diagnostics = kde_temporal_segments(
            stage[members], weights[members], bandwidth,
            min_peak_distance, valley_ratio, max_span,
            min_cells)
        segment[members] = local_labels
        for label in np.unique(local_labels):
            local_members = members[local_labels == label]
            left = float(stage[local_members].min())
            right = float(stage[local_members].max())
            interval_left[local_members] = left
            interval_right[local_members] = right
        diagnostics.update(
            state_id=int(state),
            n_metacells=int(len(members)),
            n_cells=int(weights[members].sum()))
        diagnostic_rows.append(diagnostics)
    if np.any(segment < 0) or not np.isfinite(interval_left).all():
        raise RuntimeError("Adaptive temporal segmentation left metacells unassigned")
    pairs = pd.MultiIndex.from_arrays([state_id, segment])
    node_id, _ = pd.factorize(pairs, sort=True)
    return (segment, node_id.astype(np.int32), interval_left,
            interval_right, pd.DataFrame(diagnostic_rows))


def composition(group, category, weights, group_name, category_name):
    frame = pd.DataFrame({group_name: group, category_name: category,
                          "cell_count": weights})
    result = (frame.groupby([group_name, category_name], observed=True,
                            as_index=False).cell_count.sum())
    total = result.groupby(group_name).cell_count.transform("sum")
    result["fraction"] = result.cell_count / total
    result["rank"] = (result.groupby(group_name).cell_count
                      .rank(method="first", ascending=False).astype(int))
    return result.sort_values([group_name, "rank"]).reset_index(drop=True)


def summarize_groups(assignment, group_name):
    grouped = assignment.groupby(group_name, observed=True, sort=True)
    aggregation = {
        "lineage": ("lineage", "first"),
        "state_within_lineage": ("state_within_lineage", "first"),
        "n_metacells": ("microcell_id", "size"),
        "n_cells": ("n_cells", "sum"),
        "mean_stage": ("mean_stage", "mean"),
        "min_stage": ("mean_stage", "min"),
        "max_stage": ("mean_stage", "max"),
    }
    if group_name != "state_id":
        aggregation["state_id"] = ("state_id", "first")
    table = grouped.agg(**aggregation).reset_index()
    weighted = np.bincount(
        assignment[group_name].to_numpy(dtype=int),
        weights=assignment.mean_stage.to_numpy() * assignment.n_cells.to_numpy(),
        minlength=len(table)) / table.n_cells.to_numpy()
    table["cell_weighted_mean_stage"] = weighted
    return table


def attach_exact_celltype_purity(root, assignment, states, nodes):
    path = root / "microcell_celltype_composition.parquet"
    if not path.exists():
        return states, nodes, pd.DataFrame(), pd.DataFrame()
    comp = pd.read_parquet(path, columns=["microcell_id", "celltype", "cell_count"])
    ids = comp.microcell_id.to_numpy(dtype=int)
    comp["state_id"] = assignment.state_id.to_numpy()[ids]
    comp["node_id"] = assignment.node_id.to_numpy()[ids]
    state_comp = (comp.groupby(["state_id", "celltype"], observed=True,
                               as_index=False).cell_count.sum())
    node_comp = (comp.groupby(["node_id", "celltype"], observed=True,
                              as_index=False).cell_count.sum())
    for table, group_name in ((state_comp, "state_id"), (node_comp, "node_id")):
        total = table.groupby(group_name).cell_count.transform("sum")
        table["fraction"] = table.cell_count / total
        table["rank"] = (table.groupby(group_name).cell_count
                         .rank(method="first", ascending=False).astype(int))
        table.sort_values([group_name, "rank"], inplace=True)
    state_top = state_comp[state_comp["rank"] == 1].rename(columns={
        "celltype": "dominant_celltype", "cell_count": "dominant_celltype_cells",
        "fraction": "celltype_purity"})
    node_top = node_comp[node_comp["rank"] == 1].rename(columns={
        "celltype": "dominant_celltype", "cell_count": "dominant_celltype_cells",
        "fraction": "celltype_purity"})
    states = states.merge(
        state_top[["state_id", "dominant_celltype", "dominant_celltype_cells",
                   "celltype_purity"]], on="state_id", how="left", validate="one_to_one")
    nodes = nodes.merge(
        node_top[["node_id", "dominant_celltype", "dominant_celltype_cells",
                  "celltype_purity"]], on="node_id", how="left", validate="one_to_one")
    return states, nodes, state_comp, node_comp


def group_centroids(embedding, labels, n_groups):
    sums = np.zeros((n_groups, embedding.shape[1]), dtype=np.float64)
    counts = np.bincount(labels, minlength=n_groups).astype(float)
    for start in range(0, len(embedding), 20_000):
        stop = min(start + 20_000, len(embedding))
        np.add.at(sums, labels[start:stop], np.asarray(embedding[start:stop]))
    sums /= counts[:, None]
    norms = np.linalg.norm(sums, axis=1)
    return (sums / np.maximum(norms[:, None], 1e-12)).astype(np.float32)


def filter_temporal_nodes(assignment, nodes, node_comp, node_embeddings,
                          min_node_cells):
    """Filter low-support nodes while retaining complete audit information."""
    raw_ids = nodes.node_id.to_numpy(dtype=np.int64)
    if not np.array_equal(raw_ids, np.arange(len(nodes))):
        raise ValueError("Raw temporal node IDs must be contiguous from zero")
    keep = nodes.n_cells.to_numpy(dtype=np.int64) >= min_node_cells
    if not keep.any():
        raise ValueError(
            f"--min-node-cells {min_node_cells} removes every temporal node")
    mapping = np.full(len(nodes), -1, dtype=np.int64)
    mapping[keep] = np.arange(np.count_nonzero(keep), dtype=np.int64)

    result_assignment = assignment.copy()
    result_assignment.rename(columns={"node_id": "raw_node_id"}, inplace=True)
    result_assignment["node_id"] = mapping[
        result_assignment.raw_node_id.to_numpy(dtype=np.int64)]
    result_assignment["passes_support_filter"] = result_assignment.node_id.ge(0)

    retained = nodes.loc[keep].copy()
    retained.rename(columns={"node_id": "raw_node_id"}, inplace=True)
    retained.insert(0, "node_id", mapping[retained.raw_node_id.to_numpy(dtype=np.int64)])
    retained["passes_support_filter"] = True
    retained.reset_index(drop=True, inplace=True)

    filtered = nodes.loc[~keep].copy()
    filtered.rename(columns={"node_id": "raw_node_id"}, inplace=True)
    filtered.insert(0, "node_id", -1)
    filtered["passes_support_filter"] = False
    filtered["filter_reason"] = "n_cells_below_min_node_cells"
    filtered.reset_index(drop=True, inplace=True)

    retained_comp = node_comp[node_comp.node_id.isin(raw_ids[keep])].copy()
    filtered_comp = node_comp[node_comp.node_id.isin(raw_ids[~keep])].copy()
    for table in (retained_comp, filtered_comp):
        table.rename(columns={"node_id": "raw_node_id"}, inplace=True)
        table.insert(0, "node_id", mapping[
            table.raw_node_id.to_numpy(dtype=np.int64)])
    retained_comp.sort_values(["node_id", "rank"], inplace=True)
    retained_comp.reset_index(drop=True, inplace=True)
    filtered_comp.sort_values(["raw_node_id", "rank"], inplace=True)
    filtered_comp.reset_index(drop=True, inplace=True)

    retained_embeddings = np.asarray(node_embeddings[keep], dtype=np.float32)
    return (result_assignment, retained, retained_comp, retained_embeddings,
            filtered, filtered_comp)


def weighted_purity(comp):
    if comp.empty:
        return None
    return float(comp.loc[comp["rank"] == 1, "cell_count"].sum() /
                 comp.cell_count.sum())


def plot_outputs(root, output, assignment, states, nodes, max_points, dpi, seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coordinates = root / "microcell_umap_coordinates.parquet"
    if not coordinates.exists():
        print("No saved UMAP coordinates; skipping UMAP plots.", flush=True)
        return
    coords = pd.read_parquet(coordinates).sort_values("microcell_id").reset_index(drop=True)
    if not np.array_equal(coords.microcell_id.to_numpy(), assignment.microcell_id.to_numpy()):
        raise ValueError("Saved UMAP coordinates do not match current metacells")
    xy = coords[["UMAP1", "UMAP2"]].to_numpy()
    rng = np.random.default_rng(seed)
    selected = (np.arange(len(xy)) if len(xy) <= max_points else
                np.sort(rng.choice(len(xy), max_points, replace=False)))

    def save(fig, name):
        fig.savefig(output / f"{name}.png", dpi=dpi, bbox_inches="tight")
        fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    lineage = pd.Categorical(assignment.lineage)
    fig, ax = plt.subplots(figsize=(13, 10))
    palette = plt.get_cmap("tab20")(np.linspace(0, 1, len(lineage.categories)))
    codes = lineage.codes[selected]
    for code, name in enumerate(lineage.categories):
        mask = codes == code
        ax.scatter(xy[selected[mask], 0], xy[selected[mask], 1], s=.35,
                   color=palette[code], label=name, linewidths=0, rasterized=True)
    ax.legend(loc="center left", bbox_to_anchor=(1, .5), markerscale=7, fontsize=8)
    ax.set(xlabel="UMAP 1", ylabel="UMAP 2", title="Known lineage boundaries")
    save(fig, "umap_by_lineage")

    for column, title, cmap in (
            ("state_id", "Embedding-only Leiden states", "turbo"),
            ("node_id", "Leiden state × predicted-stage nodes", "turbo"),
            ("mean_stage", "Mean predicted stage", "viridis")):
        fig, ax = plt.subplots(figsize=(12, 10))
        points = ax.scatter(xy[selected, 0], xy[selected, 1],
                            c=assignment[column].to_numpy()[selected],
                            s=.35, cmap=cmap, linewidths=0, rasterized=True)
        fig.colorbar(points, ax=ax, label=column)
        ax.set(xlabel="UMAP 1", ylabel="UMAP 2", title=title)
        save(fig, f"umap_by_{column}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].hist(states.n_metacells, bins=50)
    axes[0, 0].set(xlabel="Metacells per Leiden state", ylabel="States")
    axes[0, 1].hist(nodes.n_metacells, bins=50)
    axes[0, 1].set(xlabel="Metacells per temporal node", ylabel="Nodes")
    if "celltype_purity" in states:
        axes[1, 0].hist(states.celltype_purity.dropna(), bins=np.linspace(0, 1, 41))
        axes[1, 0].set(xlabel="Leiden-state cell-type purity", ylabel="States")
        axes[1, 1].hist(nodes.celltype_purity.dropna(), bins=np.linspace(0, 1, 41))
        axes[1, 1].set(xlabel="Temporal-node cell-type purity", ylabel="Nodes")
    save(fig, "state_and_node_diagnostics")


def main(argv=None):
    args = parse_args(argv)
    if (args.knn < 2 or args.leiden_resolution <= 0 or args.leiden_iterations == 0
            or args.threads < 1 or args.plot_max_points < 1
            or args.min_node_cells < 1):
        raise ValueError(
            "Invalid kNN, Leiden, node-support, thread, or plotting parameter")
    if args.temporal_split_method == "kde":
        if (args.stage_kde_bandwidth <= 0
                or not np.isfinite(args.stage_kde_bandwidth)):
            raise ValueError("--stage-kde-bandwidth must be positive and finite")
        if (args.stage_min_peak_distance <= 0
                or not np.isfinite(args.stage_min_peak_distance)):
            raise ValueError(
                "--stage-min-peak-distance must be positive and finite")
        if (not 0 <= args.stage_valley_ratio <= 1
                or not np.isfinite(args.stage_valley_ratio)):
            raise ValueError("--stage-valley-ratio must be within [0, 1]")
        if (args.max_node_stage_span < 0
                or not np.isfinite(args.max_node_stage_span)):
            raise ValueError(
                "--max-node-stage-span must be finite and nonnegative")
    root, output = args.input_dir.resolve(), args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    metadata, embedding = load_inputs(root)
    state_local, state_id, graph_summary = discover_lineage_states(
        metadata, embedding, args.knn, args.leiden_resolution,
        args.leiden_iterations, args.threads, args.seed)
    stage_values = metadata.mean_stage.to_numpy(dtype=np.float64)
    split_diagnostics = pd.DataFrame()
    if args.temporal_split_method == "fixed":
        stage_bin, node_id = make_temporal_nodes(
            state_id, stage_values, args.stage_bin_width)
        temporal_segment = stage_bin
        interval_left = stage_bin * args.stage_bin_width
        interval_right = (stage_bin + 1) * args.stage_bin_width
    else:
        (temporal_segment, node_id, interval_left, interval_right,
         split_diagnostics) = make_kde_temporal_nodes(
             state_id, stage_values, metadata.n_cells.to_numpy(dtype=np.float64),
             args.stage_kde_bandwidth, args.stage_min_peak_distance,
             args.stage_valley_ratio, args.max_node_stage_span,
             args.min_node_cells)

    assignment = metadata.copy()
    assignment["state_within_lineage"] = state_local
    assignment["state_id"] = state_id
    assignment["temporal_segment_within_state"] = temporal_segment
    assignment["temporal_interval_left"] = interval_left
    assignment["temporal_interval_right"] = interval_right
    if args.temporal_split_method == "fixed":
        assignment["predicted_stage_bin"] = stage_bin
        assignment["stage_bin_left"] = interval_left
        assignment["stage_bin_right"] = interval_right
    assignment["node_id"] = node_id
    states = summarize_groups(assignment, "state_id")
    nodes = summarize_groups(assignment, "node_id")
    first_by_node = assignment.groupby("node_id", sort=True).first()
    nodes["temporal_segment_within_state"] = nodes.node_id.map(
        first_by_node.temporal_segment_within_state)
    nodes["temporal_interval_left"] = nodes.node_id.map(
        first_by_node.temporal_interval_left)
    nodes["temporal_interval_right"] = nodes.node_id.map(
        first_by_node.temporal_interval_right)
    if args.temporal_split_method == "fixed":
        nodes["predicted_stage_bin"] = nodes.node_id.map(
            first_by_node.predicted_stage_bin)
        nodes["stage_bin_left"] = nodes.temporal_interval_left
        nodes["stage_bin_right"] = nodes.temporal_interval_right
    else:
        state_annotations = (
            assignment.groupby("state_id", sort=True)
            .agg(lineage=("lineage", "first"),
                 state_within_lineage=("state_within_lineage", "first"))
            .reset_index())
        split_diagnostics = split_diagnostics.merge(
            state_annotations, on="state_id", how="left",
            validate="one_to_one")
    states, nodes, state_comp, node_comp = attach_exact_celltype_purity(
        root, assignment, states, nodes)
    raw_node_count = len(nodes)
    raw_node_purity = weighted_purity(node_comp)
    raw_node_embeddings = group_centroids(embedding, node_id, raw_node_count)
    (assignment, nodes, node_comp, node_embeddings,
     filtered_nodes, filtered_node_comp) = filter_temporal_nodes(
         assignment, nodes, node_comp, raw_node_embeddings,
         args.min_node_cells)

    assignment.to_parquet(output / "microcell_node_assignments.parquet", index=False)
    states.to_parquet(output / "states.parquet", index=False)
    nodes.to_parquet(output / "nodes.parquet", index=False)
    filtered_nodes.to_parquet(output / "filtered_small_nodes.parquet", index=False)
    graph_summary.to_csv(output / "lineage_knn_summary.csv", index=False)
    if not split_diagnostics.empty:
        split_diagnostics.to_parquet(
            output / "temporal_split_diagnostics.parquet", index=False)
    if not state_comp.empty:
        state_comp.to_parquet(output / "state_celltype_composition.parquet", index=False)
        node_comp.to_parquet(output / "node_celltype_composition.parquet", index=False)
        filtered_node_comp.to_parquet(
            output / "filtered_small_node_celltype_composition.parquet", index=False)
    np.save(output / "node_embeddings.npy", node_embeddings)
    config = {key: str(value.resolve()) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config["inputs"] = {
        name: {"size": (root / name).stat().st_size,
               "mtime_ns": (root / name).stat().st_mtime_ns}
        for name in ("microcells.parquet", "microcell_embeddings.npy")}
    (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    summary = {
        "n_metacells": int(len(metadata)), "n_cells": int(metadata.n_cells.sum()),
        "n_lineages": int(metadata.lineage.nunique()),
        "n_embedding_states": int(len(states)),
        "n_raw_temporal_nodes": int(raw_node_count),
        "n_temporal_nodes": int(len(nodes)),
        "min_node_cells": int(args.min_node_cells),
        "n_filtered_small_nodes": int(len(filtered_nodes)),
        "n_filtered_metacells": int(filtered_nodes.n_metacells.sum()),
        "n_filtered_cells": int(filtered_nodes.n_cells.sum()),
        "filtered_cell_fraction": float(
            filtered_nodes.n_cells.sum() / metadata.n_cells.sum()),
        "state_cell_weighted_celltype_purity": weighted_purity(state_comp),
        "raw_node_cell_weighted_celltype_purity": raw_node_purity,
        "node_cell_weighted_celltype_purity": weighted_purity(node_comp),
        "median_state_metacells": float(states.n_metacells.median()),
        "median_node_metacells": float(nodes.n_metacells.median()),
        "min_node_metacells": int(nodes.n_metacells.min()),
        "max_node_metacells": int(nodes.n_metacells.max()),
        "median_node_stage_span": float(
            (nodes.max_stage - nodes.min_stage).median()),
        "max_node_stage_span_observed": float(
            (nodes.max_stage - nodes.min_stage).max()),
        "clustering_inputs": "embedding only, independently within each known lineage",
        "temporal_split_method": args.temporal_split_method,
        "predicted_stage_usage": (
            "post-Leiden fixed-width temporal subdivision"
            if args.temporal_split_method == "fixed"
            else "post-Leiden cell-weighted KDE valleys plus broad-segment median splitting"),
        "node_support_filter": (
            "adaptive small segments are merged first; final n_cells >= min_node_cells"
            if args.temporal_split_method == "kde"
            else "n_cells >= min_node_cells; celltype-independent"),
        "filtered_assignment": "node_id=-1 with raw_node_id retained",
        "celltype_usage": "post-clustering evaluation only",
        "node_connections_constructed": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_outputs(root, output, assignment, states, nodes,
                 args.plot_max_points, args.dpi, args.seed)
    (output / "complete.json").write_text(json.dumps({"status": "complete"}) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Done: {output}", flush=True)


if __name__ == "__main__":
    main()
