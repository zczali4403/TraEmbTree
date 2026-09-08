#!/usr/bin/env python3
"""Contract redundant degree-two nodes after a Markov tree has been built.

Only consecutive temporal nodes on nonbranching paths may be merged. Lineage
roots, branch points, and terminal nodes remain singleton anchors, preserving
the rooted branching topology. Merge decisions use node embeddings and stage
span only; cell-type labels are aggregated afterwards for evaluation and plots.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import defaultdict
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
        "--output-dir", type=Path,
        default=DEFAULT_ROOT / "markov_tree_0903_nodeknn_k10_contracted_d008_stage6")
    parser.add_argument(
        "--max-cosine-distance", type=float, default=0.08,
        help="Largest distance from a candidate node to the current weighted group center.")
    parser.add_argument(
        "--max-stage-span", type=float, default=6.0,
        help="Largest allowed max_stage-min_stage across a contracted node.")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args(argv)


def log(message):
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), message, flush=True)


def normalize_rows(values):
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("node_embeddings.npy must be a finite two-dimensional array")
    if np.any(norms <= 0):
        raise ValueError("node_embeddings.npy contains a zero-norm row")
    return values / norms[:, None]


def load_inputs(args):
    tree_nodes = pd.read_parquet(args.tree_dir / "tree_nodes.parquet")
    tree_edges = pd.read_parquet(args.tree_dir / "tree_edges.parquet")
    embeddings = np.load(args.node_dir / "node_embeddings.npy")
    composition = pd.read_parquet(args.node_dir / "node_celltype_composition.parquet")
    required_nodes = {
        "node_id", "node_type", "lineage", "n_metacells", "n_cells",
        "cell_weighted_mean_stage", "min_stage", "max_stage",
    }
    required_edges = {"parent_id", "child_id", "lineage", "virtual_edge"}
    required_composition = {"node_id", "celltype", "cell_count"}
    if not required_nodes.issubset(tree_nodes):
        raise ValueError(f"tree_nodes.parquet missing {sorted(required_nodes-set(tree_nodes))}")
    if not required_edges.issubset(tree_edges):
        raise ValueError(f"tree_edges.parquet missing {sorted(required_edges-set(tree_edges))}")
    if not required_composition.issubset(composition):
        raise ValueError(
            f"node_celltype_composition.parquet missing "
            f"{sorted(required_composition-set(composition))}")
    real = tree_nodes[tree_nodes.node_type.eq("temporal_node")].copy()
    real.sort_values("node_id", inplace=True)
    real.reset_index(drop=True, inplace=True)
    if not np.array_equal(real.node_id.to_numpy(dtype=np.int64), np.arange(len(real))):
        raise ValueError("Temporal node IDs must be contiguous from zero")
    if len(embeddings) != len(real):
        raise ValueError("node_embeddings.npy is not aligned to temporal nodes")
    if set(composition.node_id.unique()) != set(real.node_id):
        raise ValueError("Cell-type composition does not cover exactly the temporal nodes")
    return tree_nodes, tree_edges, real, np.asarray(embeddings), composition


def real_tree_adjacency(real_nodes, tree_edges):
    real_ids = set(real_nodes.node_id.astype(int))
    parents = defaultdict(list)
    children = defaultdict(list)
    real_edges = tree_edges[
        tree_edges.parent_id.isin(real_ids) & tree_edges.child_id.isin(real_ids)]
    for edge in real_edges.itertuples(index=False):
        parent, child = int(edge.parent_id), int(edge.child_id)
        parents[child].append(parent)
        children[parent].append(child)
    for node in real_ids:
        if len(parents[node]) > 1:
            raise ValueError(f"Temporal node {node} has more than one tree parent")
    return parents, children, real_edges


def eligible_chains(real_nodes, parents, children):
    """Find maximal paths containing only nonanchor degree-two nodes."""
    ids = set(real_nodes.node_id.astype(int))
    eligible = {
        node for node in ids
        if len(parents[node]) == 1 and len(children[node]) == 1
    }
    chains = []
    visited = set()
    starts = sorted(
        node for node in eligible
        if parents[node][0] not in eligible
    )
    for start in starts:
        chain = []
        node = start
        while node in eligible and node not in visited:
            chain.append(node)
            visited.add(node)
            node = children[node][0]
        chains.append(chain)
    if visited != eligible:
        raise RuntimeError("Failed to cover every eligible degree-two node")
    anchors = ids - eligible
    return chains, anchors


def partition_chain(chain, real_lookup, unit_embeddings, raw_embeddings,
                    max_distance, max_stage_span):
    groups = []
    current = []
    weighted_sum = None
    group_min_stage = np.inf
    group_max_stage = -np.inf
    for node in chain:
        row = real_lookup.loc[node]
        weight = float(row.n_cells)
        if not current:
            current = [node]
            weighted_sum = raw_embeddings[node].astype(np.float64) * weight
            group_min_stage = float(row.min_stage)
            group_max_stage = float(row.max_stage)
            continue
        center = weighted_sum / max(np.linalg.norm(weighted_sum), 1e-12)
        distance = max(0.0, 1.0 - float(center @ unit_embeddings[node]))
        candidate_min = min(group_min_stage, float(row.min_stage))
        candidate_max = max(group_max_stage, float(row.max_stage))
        if distance <= max_distance and candidate_max - candidate_min <= max_stage_span:
            current.append(node)
            weighted_sum += raw_embeddings[node] * weight
            group_min_stage = candidate_min
            group_max_stage = candidate_max
        else:
            groups.append(current)
            current = [node]
            weighted_sum = raw_embeddings[node].astype(np.float64) * weight
            group_min_stage = float(row.min_stage)
            group_max_stage = float(row.max_stage)
    if current:
        groups.append(current)
    return groups


def make_groups(real_nodes, embeddings, parents, children,
                max_distance, max_stage_span):
    chains, anchors = eligible_chains(real_nodes, parents, children)
    lookup = real_nodes.set_index("node_id")
    unit = normalize_rows(embeddings)
    groups = [[node] for node in sorted(anchors)]
    for chain in chains:
        groups.extend(partition_chain(
            chain, lookup, unit, embeddings, max_distance, max_stage_span))
    covered = [node for group in groups for node in group]
    if len(covered) != len(set(covered)) or set(covered) != set(real_nodes.node_id):
        raise RuntimeError("Contracted groups do not partition temporal nodes")
    groups.sort(key=lambda group: (
        str(lookup.at[group[0], "lineage"]),
        min(float(lookup.at[node, "cell_weighted_mean_stage"]) for node in group),
        min(group)))
    return groups, chains, anchors


def aggregate_composition(groups, composition):
    old_to_new = {
        old: new for new, group in enumerate(groups) for old in group
    }
    result = composition.copy()
    result["contracted_node_id"] = result.node_id.map(old_to_new).astype(np.int64)
    result = (result.groupby(["contracted_node_id", "celltype"], observed=True,
                             as_index=False).cell_count.sum())
    totals = result.groupby("contracted_node_id").cell_count.transform("sum")
    result["fraction"] = result.cell_count / totals
    result.sort_values(
        ["contracted_node_id", "cell_count", "celltype"],
        ascending=[True, False, True], inplace=True)
    result["rank"] = result.groupby("contracted_node_id").cumcount() + 1
    result["rank"] = result["rank"].astype(np.int64)
    result.rename(columns={"contracted_node_id": "node_id"}, inplace=True)
    return result.reset_index(drop=True), old_to_new


def aggregate_real_nodes(groups, real_nodes, embeddings, composition):
    lookup = real_nodes.set_index("node_id")
    top = composition[composition["rank"].eq(1)].set_index("node_id")
    rows, contracted_embeddings, membership = [], [], []
    for new_id, group in enumerate(groups):
        frame = lookup.loc[group]
        cell_weights = frame.n_cells.to_numpy(dtype=np.float64)
        metacell_weights = frame.n_metacells.to_numpy(dtype=np.float64)
        raw = np.average(embeddings[group], axis=0, weights=cell_weights)
        contracted_embeddings.append(raw.astype(np.float32))
        lineage_values = frame.lineage.astype(str).unique()
        if len(lineage_values) != 1:
            raise RuntimeError("A contracted node crosses lineage boundaries")
        celltype = top.loc[new_id]
        rows.append({
            "node_id": new_id,
            "lineage": lineage_values[0],
            "n_source_nodes": len(group),
            "source_node_ids": [int(value) for value in group],
            "n_metacells": int(frame.n_metacells.sum()),
            "n_cells": int(frame.n_cells.sum()),
            "mean_stage": float(np.average(frame.mean_stage, weights=metacell_weights)),
            "cell_weighted_mean_stage": float(np.average(
                frame.cell_weighted_mean_stage, weights=cell_weights)),
            "min_stage": float(frame.min_stage.min()),
            "max_stage": float(frame.max_stage.max()),
            "dominant_celltype": str(celltype.celltype),
            "dominant_celltype_cells": int(celltype.cell_count),
            "celltype_purity": float(celltype.fraction),
            "node_type": "temporal_node",
            "tree_stage": float(np.average(
                frame.cell_weighted_mean_stage, weights=cell_weights)),
        })
        for order, old_id in enumerate(group):
            membership.append({
                "source_node_id": int(old_id),
                "contracted_node_id": int(new_id),
                "position_within_contracted_node": int(order),
                "source_was_anchor": False,
            })
    return (pd.DataFrame(rows), np.stack(contracted_embeddings),
            pd.DataFrame(membership))


def rebuild_tree(real_nodes, old_tree_nodes, old_tree_edges, old_to_new, membership,
                 parents, children):
    old_virtual = old_tree_nodes[~old_tree_nodes.node_type.eq("temporal_node")].copy()
    next_id = len(real_nodes)
    virtual_map = {}
    virtual_rows = []
    for old in old_virtual.sort_values(["node_type", "lineage", "node_id"]).itertuples():
        virtual_map[int(old.node_id)] = next_id
        row = old._asdict()
        row["node_id"] = next_id
        row["n_source_nodes"] = 0
        row["source_node_ids"] = []
        virtual_rows.append(row)
        next_id += 1
    id_map = dict(old_to_new)
    id_map.update(virtual_map)

    edge_rows, seen = [], set()
    for edge in old_tree_edges.itertuples(index=False):
        parent = id_map[int(edge.parent_id)]
        child = id_map[int(edge.child_id)]
        if parent == child or (parent, child) in seen:
            continue
        seen.add((parent, child))
        edge_rows.append({
            "parent_id": int(parent),
            "child_id": int(child),
            "lineage": str(edge.lineage),
            "edge_kind": str(edge.edge_kind) if bool(edge.virtual_edge) else "contracted_tree",
            "virtual_edge": bool(edge.virtual_edge),
            "source_parent_id": int(edge.parent_id),
            "source_child_id": int(edge.child_id),
        })
    edges = pd.DataFrame(edge_rows)
    nodes = pd.concat([real_nodes, pd.DataFrame(virtual_rows)], ignore_index=True, sort=False)
    lookup = nodes.set_index("node_id")
    edges["parent_stage"] = edges.parent_id.map(lookup.tree_stage)
    edges["child_stage"] = edges.child_id.map(lookup.tree_stage)
    edges["stage_delta"] = edges.child_stage - edges.parent_stage
    if (edges.stage_delta <= 0).any():
        bad = edges[edges.stage_delta <= 0].head()
        raise RuntimeError(f"Contraction created a non-forward edge:\n{bad}")

    original_indegree = {node: len(parents[node]) for node in parents}
    original_outdegree = {node: len(children[node]) for node in children}
    anchor_nodes = {
        node for node in original_indegree
        if original_indegree[node] != 1 or original_outdegree[node] != 1
    }
    membership["source_was_anchor"] = membership.source_node_id.isin(anchor_nodes)
    return nodes, edges, membership


def tree_layout(nodes, edges):
    children = defaultdict(list)
    for edge in edges.itertuples(index=False):
        children[int(edge.parent_id)].append(int(edge.child_id))
    roots = nodes[nodes.node_type.eq("lineage_root")]
    stage = nodes.set_index("node_id").tree_stage.to_dict()
    x = {}
    for band, root_row in enumerate(roots.sort_values("lineage").itertuples()):
        root = int(root_row.node_id)
        descendants, stack = [], [root]
        while stack:
            node = stack.pop()
            descendants.append(node)
            stack.extend(children[node])
        leaves = sorted(
            [node for node in descendants if not children[node]],
            key=lambda node: (stage[node], node))
        if len(leaves) == 1:
            x[leaves[0]] = band + .5
        else:
            for position, leaf in enumerate(leaves):
                x[leaf] = band + .05 + .9 * position / (len(leaves) - 1)
        for node in sorted(descendants, key=lambda value: (stage[value], value), reverse=True):
            if node not in x:
                x[node] = float(np.mean([x[child] for child in children[node]]))
    global_roots = nodes[nodes.node_type.eq("global_root")]
    if len(global_roots) != 1:
        raise ValueError("Expected exactly one global root")
    global_root = int(global_roots.iloc[0].node_id)
    x[global_root] = float(np.mean([x[int(node)] for node in roots.node_id]))
    result = nodes.copy()
    result["tree_x"] = result.node_id.map(x).astype(float)
    result["tree_y"] = result.tree_stage.astype(float)
    return result


def discrete_colors(size, matplotlib):
    colors, seen = [], set()
    for name in ("tab20", "tab20b", "tab20c", "Set1", "Set2", "Set3",
                 "Dark2", "Paired", "Accent"):
        for value in matplotlib.colormaps[name].colors:
            color = matplotlib.colors.to_hex(value).upper()
            if color not in seen and color not in {"#FFFFFF", "#000000", "#B0B0B0"}:
                colors.append(color)
                seen.add(color)
    golden = .618033988749895
    index = 0
    while len(colors) < size:
        hsv = ((index * golden) % 1.0, (.62, .78, .9)[index % 3],
               (.68, .82, .94)[(index // 3) % 3])
        color = matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb(hsv)).upper()
        if color not in seen:
            colors.append(color)
            seen.add(color)
        index += 1
    return colors[:size]


def plot_outputs(nodes, edges, output, source_tree_dir, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    real = nodes.node_type.eq("temporal_node")
    lookup = nodes.set_index("node_id")
    lineages = sorted(nodes.loc[real, "lineage"].astype(str).unique())
    lineage_palette = dict(zip(
        lineages, plt.get_cmap("tab20")(np.linspace(0, 1, len(lineages)))))

    labels = nodes.loc[real, "dominant_celltype"].astype(str)
    weights = (nodes.loc[real].assign(_label=labels)
               .groupby("_label", sort=False).n_cells.sum().sort_values(ascending=False))
    celltypes = weights.index.tolist()
    colors = {}
    source_colors = source_tree_dir / "celltype_colors.csv"
    if source_colors.exists():
        table = pd.read_csv(source_colors)
        if {"celltype", "color"}.issubset(table):
            colors.update(dict(zip(table.celltype.astype(str), table.color.astype(str))))
    missing = [label for label in celltypes if label not in colors]
    for color in discrete_colors(len(celltypes) + len(colors), matplotlib):
        if not missing:
            break
        if color not in set(colors.values()):
            colors[missing.pop(0)] = color

    def draw_edges(ax, selected):
        for edge in selected.itertuples(index=False):
            ax.plot(
                [lookup.at[edge.parent_id, "tree_stage"],
                 lookup.at[edge.child_id, "tree_stage"]],
                [lookup.at[edge.parent_id, "tree_x"],
                 lookup.at[edge.child_id, "tree_x"]],
                color="#999999", linewidth=.45, alpha=.5, zorder=1)

    def save(fig, name):
        fig.savefig(output / f"{name}.png", dpi=dpi, bbox_inches="tight")
        fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(17, 13))
    draw_edges(ax, edges)
    for lineage in lineages:
        mask = real & nodes.lineage.eq(lineage)
        ax.scatter(nodes.loc[mask, "tree_stage"], nodes.loc[mask, "tree_x"],
                   s=10, color=lineage_palette[lineage], linewidths=0, zorder=2)
    virtual = ~real
    ax.scatter(nodes.loc[virtual, "tree_stage"], nodes.loc[virtual, "tree_x"],
               marker="*", s=55, color="black", zorder=3)
    ax.set(xlabel="Cell-weighted predicted stage", ylabel="Lineage-separated branch layout",
           title="Contracted Markov tree — lineage (early to late, left to right)")
    ax.set_yticks(np.arange(len(lineages)) + .5, lineages, fontsize=8)
    ax.grid(axis="x", color="#EEEEEE", linewidth=.5)
    save(fig, "tree_by_lineage")

    fig, ax = plt.subplots(figsize=(17, 13))
    draw_edges(ax, edges)
    points = ax.scatter(nodes.loc[real, "tree_stage"], nodes.loc[real, "tree_x"],
                        c=nodes.loc[real, "tree_stage"], cmap="viridis", s=10,
                        linewidths=0, zorder=2)
    fig.colorbar(points, ax=ax, label="Cell-weighted predicted stage")
    ax.set(xlabel="Cell-weighted predicted stage", ylabel="Lineage-separated branch layout",
           title="Contracted Markov tree — stage (early to late, left to right)")
    ax.set_yticks(np.arange(len(lineages)) + .5, lineages, fontsize=8)
    ax.grid(axis="x", color="#EEEEEE", linewidth=.5)
    save(fig, "tree_by_stage")

    fig, ax = plt.subplots(figsize=(18, 14))
    draw_edges(ax, edges)
    ax.scatter(nodes.loc[real, "tree_stage"], nodes.loc[real, "tree_x"],
               c=[colors[label] for label in labels], s=10, linewidths=0, zorder=2)
    handles = [Line2D([0], [0], marker="o", linestyle="", color=colors[label], label=label)
               for label in celltypes]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, -.08),
              fontsize=6, ncol=min(8, max(1, len(celltypes))), markerscale=1.5,
              columnspacing=.8, handletextpad=.3)
    ax.set(xlabel="Cell-weighted predicted stage", ylabel="Lineage-separated branch layout",
           title=f"Contracted Markov tree — all {len(celltypes)} dominant cell types")
    ax.set_yticks(np.arange(len(lineages)) + .5, lineages, fontsize=8)
    ax.grid(axis="x", color="#EEEEEE", linewidth=.5)
    save(fig, "tree_by_dominant_celltype")

    lineage_output = output / "trees_by_lineage_celltype"
    lineage_output.mkdir(exist_ok=True)
    for position, lineage in enumerate(lineages, 1):
        mask = real & nodes.lineage.eq(lineage)
        local_labels = nodes.loc[mask, "dominant_celltype"].astype(str)
        local_types = [label for label in celltypes if label in set(local_labels)]
        legend_columns = max(1, math.ceil(len(local_types) / 30))
        fig, ax = plt.subplots(figsize=(12 + 3 * legend_columns, 8))
        local_edges = edges[edges.lineage.eq(lineage) & edges.edge_kind.ne("global_root")]
        for edge in local_edges.itertuples(index=False):
            ax.plot(
                [lookup.at[edge.parent_id, "tree_x"],
                 lookup.at[edge.child_id, "tree_x"]],
                [lookup.at[edge.parent_id, "tree_stage"],
                 lookup.at[edge.child_id, "tree_stage"]],
                color="#999999", linewidth=.55, alpha=.6, zorder=1)
        ax.scatter(nodes.loc[mask, "tree_x"], nodes.loc[mask, "tree_stage"],
                   c=[colors[label] for label in local_labels], s=16,
                   linewidths=0, zorder=2)
        root = nodes[nodes.node_type.eq("lineage_root") & nodes.lineage.eq(lineage)]
        ax.scatter(root.tree_x, root.tree_stage, marker="*", s=80,
                   color="black", zorder=3)
        handles = [Line2D([0], [0], marker="o", linestyle="", color=colors[label], label=label)
                   for label in local_types]
        handles.append(Line2D([0], [0], marker="*", linestyle="", color="black",
                              markersize=9, label="Lineage root"))
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1, .5),
                  fontsize=8, ncol=legend_columns, markerscale=1.25,
                  columnspacing=1, handletextpad=.35)
        ax.set(xlabel="Branch layout", ylabel="Cell-weighted predicted stage",
               title=f"{lineage} — {mask.sum()} contracted nodes")
        ax.invert_yaxis()
        ax.grid(axis="y", color="#EEEEEE", linewidth=.5)
        safe = re.sub(r"[^A-Za-z0-9]+", "_", lineage).strip("_").lower()
        stem = lineage_output / f"{position:02d}_{safe}"
        fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)

    pd.DataFrame({
        "celltype": celltypes,
        "color": [colors[label] for label in celltypes],
        "n_cells": [int(weights[label]) for label in celltypes],
    }).to_csv(output / "celltype_colors.csv", index=False)


def topology_counts(nodes, edges):
    real_ids = set(nodes.loc[nodes.node_type.eq("temporal_node"), "node_id"].astype(int))
    indegree = {node: 0 for node in real_ids}
    outdegree = {node: 0 for node in real_ids}
    for edge in edges.itertuples(index=False):
        if edge.parent_id in real_ids and edge.child_id in real_ids:
            outdegree[int(edge.parent_id)] += 1
            indegree[int(edge.child_id)] += 1
    return {
        "n_lineage_roots": sum(value == 0 for value in indegree.values()),
        "n_branch_points": sum(value > 1 for value in outdegree.values()),
        "n_terminal_nodes": sum(value == 0 for value in outdegree.values()),
    }


def main(argv=None):
    args = parse_args(argv)
    if args.max_cosine_distance < 0 or not np.isfinite(args.max_cosine_distance):
        raise ValueError("--max-cosine-distance must be finite and nonnegative")
    if args.max_stage_span <= 0 or not np.isfinite(args.max_stage_span):
        raise ValueError("--max-stage-span must be positive and finite")
    source_tree = args.tree_dir.resolve()
    source_nodes = args.node_dir.resolve()
    output = args.output_dir.resolve()
    temporary = output.with_name(f".{output.name}.building")
    if output.exists():
        raise FileExistsError(f"Output path already exists: {output}")
    if temporary.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        args.tree_dir, args.node_dir = source_tree, source_nodes
        log("loading source tree, embeddings, and cell-type compositions")
        old_nodes, old_edges, real, embeddings, composition = load_inputs(args)
        parents, children, _ = real_tree_adjacency(real, old_edges)
        groups, chains, anchors = make_groups(
            real, embeddings, parents, children,
            args.max_cosine_distance, args.max_stage_span)
        contracted_composition, old_to_new = aggregate_composition(groups, composition)
        contracted_real, contracted_embeddings, membership = aggregate_real_nodes(
            groups, real, embeddings, contracted_composition)
        nodes, edges, membership = rebuild_tree(
            contracted_real, old_nodes, old_edges, old_to_new, membership,
            parents, children)
        nodes = tree_layout(nodes, edges)

        before = topology_counts(old_nodes, old_edges)
        after = topology_counts(nodes, edges)
        if before != after:
            raise RuntimeError(f"Branch topology changed: before={before}, after={after}")
        if int(nodes.loc[nodes.node_type.eq("temporal_node"), "n_cells"].sum()) != int(real.n_cells.sum()):
            raise RuntimeError("Cell count changed during contraction")
        if int(nodes.loc[nodes.node_type.eq("temporal_node"), "n_metacells"].sum()) != int(real.n_metacells.sum()):
            raise RuntimeError("Metacell count changed during contraction")

        nodes.to_parquet(temporary / "tree_nodes.parquet", index=False)
        edges.to_parquet(temporary / "tree_edges.parquet", index=False)
        membership.to_parquet(temporary / "source_to_contracted_nodes.parquet", index=False)
        contracted_composition.to_parquet(
            temporary / "node_celltype_composition.parquet", index=False)
        np.save(temporary / "node_embeddings.npy", contracted_embeddings)
        group_sizes = np.asarray([len(group) for group in groups])
        stage_spans = contracted_real.max_stage - contracted_real.min_stage
        summary = {
            "n_source_temporal_nodes": int(len(real)),
            "n_contracted_temporal_nodes": int(len(contracted_real)),
            "n_nodes_removed": int(len(real) - len(contracted_real)),
            "n_source_nonbranching_chains": int(len(chains)),
            "n_fixed_anchor_nodes": int(len(anchors)),
            "n_merged_contracted_nodes": int(np.count_nonzero(group_sizes > 1)),
            "max_source_nodes_per_contracted_node": int(group_sizes.max()),
            "median_source_nodes_per_contracted_node": float(np.median(group_sizes)),
            "median_contracted_stage_span": float(np.median(stage_spans)),
            "max_contracted_stage_span": float(stage_spans.max()),
            "max_cosine_distance": float(args.max_cosine_distance),
            "max_stage_span": float(args.max_stage_span),
            "topology_before": before,
            "topology_after": after,
            "topology_preserved": True,
            "protected_nodes": "lineage roots, branch points, and terminal nodes",
            "merge_inputs": "node embedding and stage span only",
            "celltype_usage": "post-contraction aggregation and plotting only",
            "markov_probabilities": "not recomputed; output represents the contracted tree topology",
        }
        (temporary / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        config = {key: str(value) if isinstance(value, Path) else value
                  for key, value in vars(args).items()}
        (temporary / "run_config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n")
        plot_outputs(nodes, edges, temporary, source_tree, args.dpi)
        (temporary / "complete.json").write_text(json.dumps({"complete": True}) + "\n")
        temporary.rename(output)
        log(f"done: {output}")
        log(f"temporal nodes: {len(real):,} -> {len(contracted_real):,}; "
            f"removed={len(real)-len(contracted_real):,}; "
            f"merged groups={np.count_nonzero(group_sizes > 1):,}")
    except Exception:
        log(f"failed; partial output kept at {temporary}")
        raise


if __name__ == "__main__":
    main()
