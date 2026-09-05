#!/usr/bin/env python3
"""Build and plot a stage-monotone tree from TraEmb landmark nodes.

The tree is constructed in increasing mean-stage order.  Within each lineage,
each node selects the closest earlier node of the same lineage as its parent.
The earliest node of each lineage selects the closest earlier node globally,
joining the lineage subtrees into one rooted tree.  This guarantees that every
edge points from an earlier node to a later node while preserving lineage
subtrees wherever possible.

Cell type is read from the node table as dominant_celltype, computed from
all member cells. It is used for coloring, not for constructing the tree.

Usage examples
--------------
Run with the default paths::

    /home/aiscuser/.conda/envs/train/bin/python build_landmark_tree.py \
      --output-dir landmark_tree

Use absolute paths and label every node ID::

    /home/aiscuser/.conda/envs/train/bin/python build_landmark_tree.py \
      --nodes /mnt/input/sc_cz/Concord/eval/2026_08_19/landmark_nodes/nodes.parquet \
      --embeddings /mnt/input/sc_cz/Concord/eval/2026_08_19/landmark_nodes/node_embeddings.npy \
      --metadata /mnt/input/sc_cz/Concord/data/all_lineage_260829.csv \
      --output-dir /mnt/input/sc_cz/Concord/eval/2026_08_19/landmark_tree \
      --label-nodes

Replace an existing output directory::

    /home/aiscuser/.conda/envs/train/bin/python build_landmark_tree.py \
      --output-dir landmark_tree --overwrite
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=Path, default=here / "landmark_nodes/nodes.parquet")
    parser.add_argument(
        "--embeddings", type=Path,
        default=here / "landmark_nodes/node_embeddings.npy",
    )
    parser.add_argument(
        "--metadata", type=Path,
        default=Path("/mnt/input/sc_cz/Concord/data/all_lineage_260829.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=here / "landmark_tree")
    parser.add_argument("--metadata-chunk-size", type=int, default=500_000)
    parser.add_argument(
        "--stage-column", default="mean_stage",
        help="Numeric node-table column used for temporal ordering and vertical position.",
    )
    parser.add_argument("--metric", choices=("cosine", "euclidean"), default="cosine",
                        help="Embedding distance used to select each node's earlier parent.")
    parser.add_argument("--figure-width", type=float, default=30.0)
    parser.add_argument("--figure-height", type=float, default=30.0)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--label-nodes", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def log(message: str) -> None:
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), message, flush=True)


def load_representative_metadata(
    metadata_path: Path, representative_ids: list[str], chunk_size: int
) -> pd.DataFrame:
    wanted = set(representative_ids)
    found: list[pd.DataFrame] = []
    usecols = ["cell", "celltype", "lineage_final", "stage", "dataset"]
    log(f"streaming representative annotations from {metadata_path}")
    for chunk_index, chunk in enumerate(
        pd.read_csv(metadata_path, usecols=usecols, chunksize=chunk_size)
    ):
        selected = chunk[chunk["cell"].isin(wanted)]
        if not selected.empty:
            found.append(selected.copy())
            wanted.difference_update(selected["cell"].astype(str))
        if (chunk_index + 1) % 10 == 0:
            log(f"metadata chunks={chunk_index + 1}; representatives found={len(representative_ids)-len(wanted)}/{len(representative_ids)}")
        if not wanted:
            break
    if found:
        result = pd.concat(found, ignore_index=True)
    else:
        result = pd.DataFrame(columns=usecols)
    duplicate = result["cell"].duplicated(keep=False)
    if duplicate.any():
        examples = result.loc[duplicate, "cell"].astype(str).head(10).tolist()
        raise ValueError(f"Representative cell IDs are duplicated in metadata: {examples}")
    if wanted:
        log(f"warning: {len(wanted)} representative cells have no metadata annotation")
    return result


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1)
    if not np.isfinite(values).all() or not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("Node embeddings contain NaN, Inf, or zero-norm rows")
    return values / norms[:, None]


def build_stage_monotone_tree(
    nodes: pd.DataFrame, embeddings: np.ndarray, stage_column: str, metric: str = "cosine"
) -> tuple[int, pd.DataFrame]:
    stages = nodes[stage_column].to_numpy(dtype=np.float64)
    if not np.isfinite(stages).all():
        raise ValueError(f"{stage_column!r} contains missing or non-finite values")
    node_ids = nodes["node_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(node_ids, np.arange(len(nodes))):
        raise ValueError("nodes.parquet must be ordered by contiguous node_id 0..N-1")
    lineages = nodes["lineage"].astype(str).to_numpy()
    if not np.isfinite(embeddings).all():
        raise ValueError("Node embeddings contain NaN or Inf")
    embedding_norms = np.linalg.norm(embeddings, axis=1)
    if metric == "cosine" and np.any(embedding_norms <= 0):
        raise ValueError("Cosine distance requires non-zero node embeddings")
    order = np.lexsort((node_ids, stages))
    rank = np.empty(len(nodes), dtype=np.int64)
    rank[order] = np.arange(len(nodes))
    root = int(order[0])
    rows = []
    for position in range(1, len(order)):
        child = int(order[position])
        earlier = order[:position]
        same_lineage = earlier[lineages[earlier] == lineages[child]]
        candidates = same_lineage if len(same_lineage) else earlier
        if metric == "cosine":
            similarities = (embeddings[candidates] @ embeddings[child]) / (
                embedding_norms[candidates] * embedding_norms[child]
            )
            selected = int(np.argmax(similarities))
            edge_distance = float(np.clip(1.0 - similarities[selected], 0, 2))
        elif metric == "euclidean":
            distances = np.linalg.norm(embeddings[candidates] - embeddings[child], axis=1)
            selected = int(np.argmin(distances))
            edge_distance = float(distances[selected])
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        parent = int(candidates[selected])
        rows.append({
            "parent_id": parent,
            "child_id": child,
            "edge_distance": edge_distance,
            "distance_metric": metric,
            "parent_stage": float(stages[parent]),
            "child_stage": float(stages[child]),
            "stage_delta": float(stages[child] - stages[parent]),
            "parent_lineage": str(lineages[parent]),
            "child_lineage": str(lineages[child]),
            "cross_lineage": bool(lineages[parent] != lineages[child]),
            "child_is_lineage_root": bool(len(same_lineage) == 0),
        })
    edges = pd.DataFrame(rows)
    if len(edges) != len(nodes) - 1:
        raise RuntimeError("Tree does not contain exactly N-1 edges")
    if (edges["stage_delta"] < -1e-12).any():
        raise RuntimeError("Tree contains a backward-stage edge")
    return root, edges


def tree_layout(root: int, edges: pd.DataFrame, stages: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    children: dict[int, list[int]] = {index: [] for index in range(len(stages))}
    for row in edges.itertuples(index=False):
        children[int(row.parent_id)].append(int(row.child_id))
    for parent in children:
        children[parent].sort(key=lambda child: (stages[child], child))
    x = np.full(len(stages), np.nan, dtype=np.float64)
    leaf_counter = 0

    def visit(node: int) -> float:
        nonlocal leaf_counter
        if not children[node]:
            x[node] = leaf_counter
            leaf_counter += 1
        else:
            child_x = [visit(child) for child in children[node]]
            x[node] = float(np.mean(child_x))
        return float(x[node])

    visit(root)
    if np.isnan(x).any():
        raise RuntimeError("Tree layout did not reach every node")
    if leaf_counter > 1:
        x /= leaf_counter - 1
    else:
        x[:] = 0.5
    return x, stages.copy()


def _discrete_palette(size: int) -> list[str]:
    """Return deterministic categorical colors without sampling a gradient cmap."""
    palette_names = (
        "tab20", "tab20b", "tab20c", "Set1", "Set2", "Set3",
        "Dark2", "Paired", "Accent",
    )
    rgb: list[tuple[float, float, float]] = []
    seen: set[str] = set()
    for name in palette_names:
        cmap = plt.get_cmap(name)
        for value in cmap.colors:
            color = matplotlib.colors.to_hex(value).upper()
            if color not in seen and color not in {"#FFFFFF", "#000000", "#B0B0B0"}:
                seen.add(color)
                rgb.append(tuple(matplotlib.colors.to_rgb(color)))

    # If more colors are needed, select them from a discrete RGB candidate grid.
    # Greedy maximin selection keeps each added color far from those already used.
    if len(rgb) < size:
        levels = np.linspace(0.12, 0.88, 7)
        candidates = np.asarray(
            [(r, g, b) for r in levels for g in levels for b in levels
             if 0.25 < (max(r, g, b) - min(r, g, b))
             and 0.18 < (0.2126*r + 0.7152*g + 0.0722*b) < 0.82],
            dtype=float,
        )
        chosen = np.asarray(rgb, dtype=float)
        while len(rgb) < size:
            distance = ((candidates[:, None, :] - chosen[None, :, :]) ** 2).sum(axis=2).min(axis=1)
            pick = int(np.argmax(distance))
            selected = tuple(candidates[pick])
            rgb.append(selected)
            chosen = np.vstack((chosen, selected))
            candidates = np.delete(candidates, pick, axis=0)
    return [matplotlib.colors.to_hex(color).upper() for color in rgb[:size]]


def categorical_colors(values: pd.Series) -> tuple[dict[str, str], list[str]]:
    labels = sorted(values.fillna("Unknown").astype(str).unique().tolist())
    known_labels = [label for label in labels if label != "Unknown"]
    palette = _discrete_palette(len(known_labels))
    colors = dict(zip(known_labels, palette))
    if "Unknown" in labels:
        colors["Unknown"] = "#B0B0B0"
    return colors, labels


def plot_tree(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    color_column: str,
    color_map: dict[str, str],
    labels: list[str],
    output_stem: Path,
    width: float,
    height: float,
    dpi: int,
    label_nodes: bool,
) -> None:
    x = nodes["tree_x"].to_numpy()
    y = nodes["tree_y_stage"].to_numpy()
    fig, ax = plt.subplots(figsize=(width, height))
    for edge in edges.itertuples(index=False):
        parent, child = int(edge.parent_id), int(edge.child_id)
        color = "#B8B8B8" if not edge.cross_lineage else "#555555"
        linewidth = 0.55 if not edge.cross_lineage else 1.2
        ax.plot([x[parent], x[child]], [y[parent], y[child]], color=color, linewidth=linewidth, alpha=0.72, zorder=1)
    categories = nodes[color_column].fillna("Unknown").astype(str)
    point_colors = categories.map(color_map).to_numpy()
    sizes = 28.0 + 110.0 * np.sqrt(nodes["n_cells"].to_numpy() / nodes["n_cells"].max())
    ax.scatter(x, y, s=sizes, c=point_colors, edgecolors="white", linewidths=0.35, zorder=2)
    if label_nodes:
        for node_id in range(len(nodes)):
            ax.text(x[node_id], y[node_id], str(node_id), fontsize=4.5, ha="center", va="bottom")
    ax.set_xlabel("Tree topology layout (horizontal position has no quantitative meaning)")
    ax.set_ylabel("Mean developmental stage")
    ax.set_title(f"TraEmb landmark tree colored by {color_column}")
    ax.invert_yaxis()
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.5)
    ax.set_xlim(-0.025, 1.025)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color_map[label],
               markeredgecolor="none", markersize=6, label=label)
        for label in labels
    ]
    legend_columns = max(1, math.ceil(len(labels) / 35))
    ax.legend(handles=handles, title=color_column, loc="upper left", bbox_to_anchor=(1.01, 1),
              frameon=False, fontsize=7, title_fontsize=8, ncol=legend_columns)
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_tree_continuous(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    color_column: str,
    output_stem: Path,
    width: float,
    height: float,
    dpi: int,
    label_nodes: bool,
) -> None:
    values = pd.to_numeric(nodes[color_column], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError(f"{color_column!r} contains no finite values")
    if ((values[finite] < 0) | (values[finite] > 1)).any():
        raise ValueError(f"{color_column!r} must be between 0 and 1")

    x = nodes["tree_x"].to_numpy()
    y = nodes["tree_y_stage"].to_numpy()
    fig, ax = plt.subplots(figsize=(width, height))
    for edge in edges.itertuples(index=False):
        parent, child = int(edge.parent_id), int(edge.child_id)
        color = "#B8B8B8" if not edge.cross_lineage else "#555555"
        linewidth = 0.55 if not edge.cross_lineage else 1.2
        ax.plot([x[parent], x[child]], [y[parent], y[child]], color=color,
                linewidth=linewidth, alpha=0.72, zorder=1)
    sizes = 28.0 + 110.0 * np.sqrt(nodes["n_cells"].to_numpy() / nodes["n_cells"].max())
    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_bad("#B0B0B0")
    points = ax.scatter(
        x, y, s=sizes, c=np.ma.masked_invalid(values), cmap=cmap, vmin=0.0, vmax=1.0,
        edgecolors="white", linewidths=0.35, zorder=2,
    )
    if label_nodes:
        for node_id in range(len(nodes)):
            ax.text(x[node_id], y[node_id], str(node_id), fontsize=4.5,
                    ha="center", va="bottom")
    ax.set_xlabel("Tree topology layout (horizontal position has no quantitative meaning)")
    ax.set_ylabel("Mean developmental stage")
    ax.set_title("TraEmb landmark tree colored by cell-type purity")
    ax.invert_yaxis()
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.5)
    ax.set_xlim(-0.025, 1.025)
    colorbar = fig.colorbar(points, ax=ax, pad=0.015, fraction=0.025)
    colorbar.set_label("Cell-type purity")
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    started = time.time()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory exists: {output_dir}; use --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    nodes = pd.read_parquet(args.nodes).sort_values("node_id").reset_index(drop=True)
    required = {
        "node_id", "lineage", "n_cells", "dominant_celltype", "celltype_purity",
        args.stage_column,
    }
    missing = required - set(nodes.columns)
    if missing:
        raise ValueError(f"Node table is missing columns: {sorted(missing)}")
    embedding_source = np.load(args.embeddings, mmap_mode="r")
    if embedding_source.ndim != 2 or embedding_source.shape[1] != 100:
        raise ValueError(f"Expected a (*, 100) embedding matrix, got {embedding_source.shape}")
    if embedding_source.shape[0] == len(nodes):
        embeddings = np.asarray(embedding_source, dtype=np.float32)
        embedding_rows = "node_rows"
    else:
        if "representative_idx" not in nodes.columns:
            raise ValueError(
                "Embedding row count differs from node count, so representative_idx "
                "is required to select rows from a full-cell embedding matrix"
            )
        representative_indices = nodes["representative_idx"].to_numpy(dtype=np.int64)
        if representative_indices.min() < 0 or representative_indices.max() >= embedding_source.shape[0]:
            raise ValueError("representative_idx is outside the raw embedding matrix")
        embeddings = np.asarray(embedding_source[representative_indices], dtype=np.float32)
        embedding_rows = "raw_representative_cell_rows"

    nodes["dominant_celltype"] = nodes["dominant_celltype"].fillna("Unknown").astype(str)

    root, edges = build_stage_monotone_tree(nodes, embeddings, args.stage_column, args.metric)
    tree_x, tree_y = tree_layout(root, edges, nodes[args.stage_column].to_numpy(dtype=float))
    nodes["tree_x"] = tree_x
    nodes["tree_y_stage"] = tree_y
    nodes["is_root"] = nodes["node_id"] == root
    child_to_parent = dict(zip(edges["child_id"], edges["parent_id"]))
    nodes["parent_id"] = nodes["node_id"].map(child_to_parent).fillna(-1).astype(np.int32)

    lineage_colors, lineage_labels = categorical_colors(nodes["lineage"])
    celltype_colors, celltype_labels = categorical_colors(nodes["dominant_celltype"])
    nodes.to_parquet(output_dir / "tree_nodes.parquet", index=False)
    edges.to_parquet(output_dir / "tree_edges.parquet", index=False)
    pd.DataFrame({"lineage": lineage_labels, "color": [lineage_colors[x] for x in lineage_labels]}).to_csv(
        output_dir / "lineage_colors.csv", index=False
    )
    pd.DataFrame({"celltype": celltype_labels, "color": [celltype_colors[x] for x in celltype_labels]}).to_csv(
        output_dir / "celltype_colors.csv", index=False
    )

    graph = nx.DiGraph()
    for row in nodes.itertuples(index=False):
        graph.add_node(int(row.node_id), lineage=str(row.lineage), celltype=str(row.dominant_celltype),
                       stage=float(row.tree_y_stage), n_cells=int(row.n_cells), tree_x=float(row.tree_x))
    for row in edges.itertuples(index=False):
        graph.add_edge(int(row.parent_id), int(row.child_id), edge_distance=float(row.edge_distance),
                       distance_metric=str(row.distance_metric), stage_delta=float(row.stage_delta),
                       cross_lineage=bool(row.cross_lineage))
    nx.write_graphml(graph, output_dir / "landmark_tree.graphml")

    plot_tree(nodes, edges, "lineage", lineage_colors, lineage_labels,
              output_dir / "tree_by_lineage", args.figure_width, args.figure_height, args.dpi, args.label_nodes)
    plot_tree(nodes, edges, "dominant_celltype", celltype_colors, celltype_labels,
              output_dir / "tree_by_celltype", args.figure_width, args.figure_height, args.dpi, args.label_nodes)
    plot_tree_continuous(
        nodes, edges, "celltype_purity", output_dir / "tree_by_celltype_purity",
        args.figure_width, args.figure_height, args.dpi, args.label_nodes,
    )

    summary = {
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "root_node_id": root,
        "root_stage": float(nodes.loc[root, args.stage_column]),
        "root_lineage": str(nodes.loc[root, "lineage"]),
        "n_lineages": int(nodes["lineage"].nunique()),
        "n_dominant_celltypes": int(nodes["dominant_celltype"].nunique()),
        "n_unknown_dominant_celltypes": int((nodes["dominant_celltype"] == "Unknown").sum()),
        "n_cross_lineage_edges": int(edges["cross_lineage"].sum()),
        "n_backward_stage_edges": int((edges["stage_delta"] < 0).sum()),
        "distance_metric": args.metric,
        "maximum_edge_distance": float(edges["edge_distance"].max()),
        "mean_edge_distance": float(edges["edge_distance"].mean()),
        "embedding_rows": embedding_rows,
        "elapsed_seconds": time.time() - started,
    }
    with (output_dir / "tree_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({"python": sys.version, "platform": platform.platform(),
                   "method": f"stage_monotone_raw_mean_node_{args.metric}_nearest_parent"})
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
    log(f"done: {output_dir}; root={root}; edges={len(edges)}; elapsed={time.time()-started:.1f}s")


if __name__ == "__main__":
    main()
