#!/usr/bin/env python3
"""Discover candidate trajectory nodes independently within known lineages.

Within each lineage, Scanpy cosine neighbors and Leiden use embeddings only.
Cell-type labels never enter clustering. After Leiden, continuous predicted stage
splits each embedding state into temporal nodes. This script does not connect
nodes and does not construct a tree.
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
    parser.add_argument("--stage-bin-width", type=float, default=2.0)
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
            or args.threads < 1 or args.plot_max_points < 1):
        raise ValueError("Invalid kNN, Leiden, thread, or plotting parameter")
    root, output = args.input_dir.resolve(), args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    metadata, embedding = load_inputs(root)
    state_local, state_id, graph_summary = discover_lineage_states(
        metadata, embedding, args.knn, args.leiden_resolution,
        args.leiden_iterations, args.threads, args.seed)
    stage_bin, node_id = make_temporal_nodes(
        state_id, metadata.mean_stage.to_numpy(), args.stage_bin_width)
    assignment = metadata.copy()
    assignment["state_within_lineage"] = state_local
    assignment["state_id"] = state_id
    assignment["predicted_stage_bin"] = stage_bin
    assignment["stage_bin_left"] = stage_bin * args.stage_bin_width
    assignment["stage_bin_right"] = (stage_bin + 1) * args.stage_bin_width
    assignment["node_id"] = node_id
    states = summarize_groups(assignment, "state_id")
    nodes = summarize_groups(assignment, "node_id")
    nodes["predicted_stage_bin"] = nodes.node_id.map(
        assignment.groupby("node_id").predicted_stage_bin.first())
    nodes["stage_bin_left"] = nodes.predicted_stage_bin * args.stage_bin_width
    nodes["stage_bin_right"] = (nodes.predicted_stage_bin + 1) * args.stage_bin_width
    states, nodes, state_comp, node_comp = attach_exact_celltype_purity(
        root, assignment, states, nodes)
    assignment.to_parquet(output / "microcell_node_assignments.parquet", index=False)
    states.to_parquet(output / "states.parquet", index=False)
    nodes.to_parquet(output / "nodes.parquet", index=False)
    graph_summary.to_csv(output / "lineage_knn_summary.csv", index=False)
    if not state_comp.empty:
        state_comp.to_parquet(output / "state_celltype_composition.parquet", index=False)
        node_comp.to_parquet(output / "node_celltype_composition.parquet", index=False)
    np.save(output / "node_embeddings.npy",
            group_centroids(embedding, node_id, len(nodes)))
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
        "n_embedding_states": int(len(states)), "n_temporal_nodes": int(len(nodes)),
        "state_cell_weighted_celltype_purity": weighted_purity(state_comp),
        "node_cell_weighted_celltype_purity": weighted_purity(node_comp),
        "median_state_metacells": float(states.n_metacells.median()),
        "median_node_metacells": float(nodes.n_metacells.median()),
        "min_node_metacells": int(nodes.n_metacells.min()),
        "max_node_metacells": int(nodes.n_metacells.max()),
        "clustering_inputs": "embedding only, independently within each known lineage",
        "predicted_stage_usage": "post-Leiden temporal subdivision only",
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
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         