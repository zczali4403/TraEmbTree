#!/usr/bin/env python3
"""Build a lineage-aware Markov tree from temporal Leiden nodes.

Construction uses node embeddings only to choose and weight neighbors. Predicted
stage is used only to orient every real edge from earlier to later. Cell-type
annotations are carried into outputs and figures for post-hoc evaluation, but
never affect candidate edges, transition probabilities, roots, or tree edges.
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=None,
                        help="Directory containing nodes.parquet and node_embeddings.npy; required unless --plots-only.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--plots-only", action="store_true",
        help="Redraw plots from tables already in output-dir without rebuilding the graph or tree.")
    parser.add_argument(
        "--microcell-dir", type=Path, default=None,
        help="Directory with microcell_umap_coordinates.parquet. By default it is read from the Leiden run_config.json.")
    parser.add_argument("--knn", type=int, default=10,
                        help="Exact cosine neighbors computed independently per lineage.")
    parser.add_argument("--stage-column", default="cell_weighted_mean_stage")
    parser.add_argument("--mutual-knn-bonus", type=float, default=1.5)
    parser.add_argument(
        "--distance-temperature", type=float, default=0.0,
        help="Positive cosine-distance temperature; 0 selects the lineage median automatically.")
    parser.add_argument("--fate-probability-threshold", type=float, default=1e-4)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args(argv)


def log(message):
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), message, flush=True)


def normalize_rows(values):
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("node_embeddings.npy must be a finite two-dimensional array")
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("Node embeddings contain zero-norm or non-finite rows")
    return values / norms[:, None]


def load_inputs(root, stage_column):
    nodes = pd.read_parquet(root / "nodes.parquet").sort_values("node_id").reset_index(drop=True)
    embedding = np.load(root / "node_embeddings.npy")
    required = {"node_id", "lineage", "n_metacells", "n_cells", stage_column}
    if not required.issubset(nodes):
        raise ValueError(f"nodes.parquet is missing {sorted(required - set(nodes.columns))}")
    ids = nodes.node_id.to_numpy(dtype=np.int64)
    if not np.array_equal(ids, np.arange(len(nodes))):
        raise ValueError("node_id must be contiguous, ordered, and aligned to node_embeddings.npy")
    if len(nodes) != len(embedding):
        raise ValueError("nodes.parquet and node_embeddings.npy have different row counts")
    if nodes.lineage.isna().any() or nodes.lineage.astype(str).str.strip().eq("").any():
        raise ValueError("Every node must have a nonempty lineage")
    stage = nodes[stage_column].to_numpy(dtype=np.float64)
    if not np.isfinite(stage).all():
        raise ValueError(f"{stage_column} contains missing or non-finite values")
    if (nodes.n_cells.to_numpy(dtype=np.int64) <= 0).any():
        raise ValueError("n_cells must be positive")
    return nodes, normalize_rows(embedding), stage


def exact_knn_pairs(local_embedding, k):
    """Return undirected kNN-union pairs and directed-membership counts."""
    n = len(local_embedding)
    if n <= 1:
        return {}
    effective_k = min(k, n - 1)
    similarities = np.clip(local_embedding @ local_embedding.T, -1.0, 1.0)
    np.fill_diagonal(similarities, -np.inf)
    if effective_k == n - 1:
        neighbors = np.argsort(-similarities, axis=1)[:, :effective_k]
    else:
        neighbors = np.argpartition(-similarities, effective_k - 1, axis=1)[:, :effective_k]
        row = np.arange(n)[:, None]
        order = np.argsort(-similarities[row, neighbors], axis=1)
        neighbors = neighbors[row, order]
    pairs = {}
    for source in range(n):
        for target in neighbors[source]:
            target = int(target)
            pair = (source, target) if source < target else (target, source)
            if pair not in pairs:
                pairs[pair] = {"knn_memberships": 0,
                               "cosine_similarity": float(similarities[source, target])}
            pairs[pair]["knn_memberships"] += 1
    return pairs


def lineage_candidate_edges(node_ids, embedding, stage, k):
    """Orient the kNN union by stage and add a fallback parent where required."""
    local_embedding = embedding[node_ids]
    pairs = exact_knn_pairs(local_embedding, k)
    rows = []
    for (left_local, right_local), info in pairs.items():
        left, right = int(node_ids[left_local]), int(node_ids[right_local])
        if stage[left] == stage[right]:
            continue
        source, target = (left, right) if stage[left] < stage[right] else (right, left)
        rows.append({
            "source_id": source,
            "target_id": target,
            "cosine_similarity": info["cosine_similarity"],
            "knn_memberships": int(info["knn_memberships"]),
            "mutual_knn": bool(info["knn_memberships"] == 2),
            "fallback_edge": False,
        })

    incoming = {int(node): 0 for node in node_ids}
    for row in rows:
        incoming[row["target_id"]] += 1
    local_stage = stage[node_ids]
    min_stage = float(local_stage.min())
    root_ids = node_ids[local_stage == min_stage].astype(np.int64)

    # A non-root node without an earlier kNN neighbor gets its nearest earlier
    # node. There is deliberately no maximum stage-gap restriction.
    for target in node_ids[np.argsort(local_stage, kind="stable")]:
        target = int(target)
        if stage[target] == min_stage or incoming[target] > 0:
            continue
        earlier = node_ids[local_stage < stage[target]]
        similarity = embedding[earlier] @ embedding[target]
        source = int(earlier[int(np.argmax(similarity))])
        rows.append({
            "source_id": source,
            "target_id": target,
            "cosine_similarity": float(np.max(similarity)),
            "knn_memberships": 0,
            "mutual_knn": False,
            "fallback_edge": True,
        })
        incoming[target] += 1
    return rows, root_ids


def build_candidate_graph(nodes, embedding, stage, k, mutual_bonus, temperature):
    rows = []
    lineage_roots = {}
    lineage_temperatures = {}
    for position, (lineage, frame) in enumerate(nodes.groupby("lineage", sort=True), 1):
        ids = frame.node_id.to_numpy(dtype=np.int64)
        local_rows, roots = lineage_candidate_edges(ids, embedding, stage, k)
        distances = np.asarray(
            [np.clip(1.0 - row["cosine_similarity"], 0.0, 2.0) for row in local_rows])
        positive = distances[distances > 1e-8]
        local_temperature = (temperature if temperature > 0 else
                             float(np.median(positive)) if len(positive) else 0.05)
        local_temperature = max(local_temperature, 1e-6)
        for row, distance in zip(local_rows, distances):
            weight = math.exp(max(-80.0, -float(distance) / local_temperature))
            if row["mutual_knn"]:
                weight *= mutual_bonus
            row.update(
                lineage=str(lineage),
                source_stage=float(stage[row["source_id"]]),
                target_stage=float(stage[row["target_id"]]),
                stage_delta=float(stage[row["target_id"]] - stage[row["source_id"]]),
                cosine_distance=float(distance),
                distance_temperature=float(local_temperature),
                raw_weight=float(weight),
            )
            rows.append(row)
        lineage_roots[str(lineage)] = roots.tolist()
        lineage_temperatures[str(lineage)] = local_temperature
        log(f"[{position}] {lineage}: {len(ids):,} nodes, {len(local_rows):,} candidate edges, "
            f"{len(roots)} earliest roots, temperature={local_temperature:.5g}")
    edges = pd.DataFrame(rows)
    if edges.empty and len(nodes) > len(lineage_roots):
        raise RuntimeError("No directed candidate edges were constructed")
    if len(edges):
        totals = edges.groupby("source_id").raw_weight.transform("sum")
        edges["transition_probability"] = edges.raw_weight / totals
        edges.sort_values(["lineage", "source_stage", "source_id", "target_id"], inplace=True)
        edges.reset_index(drop=True, inplace=True)
    else:
        edges["transition_probability"] = pd.Series(dtype=float)
    return edges, lineage_roots, lineage_temperatures


def markov_quantities(nodes, edges, stage, lineage_roots, threshold):
    """Calculate forward edge flow and absorbing-terminal fate probabilities."""
    edges = edges.copy()
    edges["markov_flow"] = 0.0
    fate_rows = []
    most_likely = np.full(len(nodes), -1, dtype=np.int64)
    most_probability = np.full(len(nodes), np.nan)
    entropy = np.full(len(nodes), np.nan)
    terminal_flag = np.zeros(len(nodes), dtype=bool)

    for lineage, frame in nodes.groupby("lineage", sort=True):
        ids = frame.node_id.to_numpy(dtype=np.int64)
        id_set = set(ids.tolist())
        local_edge_indices = edges.index[edges.lineage.eq(str(lineage))].to_numpy(dtype=np.int64)
        outgoing = defaultdict(list)
        for edge_index in local_edge_indices:
            row = edges.loc[edge_index]
            outgoing[int(row.source_id)].append((int(edge_index), int(row.target_id),
                                                  float(row.transition_probability)))

        roots = np.asarray(lineage_roots[str(lineage)], dtype=np.int64)
        root_weights = nodes.set_index("node_id").loc[roots, "n_cells"].to_numpy(dtype=float)
        root_weights /= root_weights.sum()
        flow = {int(node): 0.0 for node in ids}
        for node, weight in zip(roots, root_weights):
            flow[int(node)] = float(weight)
        for source in sorted(ids.tolist(), key=lambda node: (stage[node], node)):
            for edge_index, target, probability in outgoing[source]:
                edge_flow = flow[source] * probability
                edges.at[edge_index, "markov_flow"] = edge_flow
                flow[target] += edge_flow

        fate = {}
        for source in sorted(ids.tolist(), key=lambda node: (stage[node], node), reverse=True):
            if not outgoing[source]:
                fate[source] = {source: 1.0}
                terminal_flag[source] = True
                continue
            distribution = defaultdict(float)
            for _, target, probability in outgoing[source]:
                for terminal, terminal_probability in fate[target].items():
                    distribution[terminal] += probability * terminal_probability
            total = sum(distribution.values())
            if total <= 0:
                raise RuntimeError(f"Invalid absorbing probabilities for node {source}")
            fate[source] = {terminal: value / total for terminal, value in distribution.items()}

        for source in ids:
            distribution = fate[int(source)]
            best_terminal, best_probability = max(distribution.items(), key=lambda item: item[1])
            most_likely[source] = best_terminal
            most_probability[source] = best_probability
            entropy[source] = -sum(value * math.log(value) for value in distribution.values() if value > 0)
            retained = [(terminal, value) for terminal, value in distribution.items()
                        if value >= threshold]
            if not retained:
                retained = [(best_terminal, best_probability)]
            retained_mass = sum(value for _, value in retained)
            for terminal, probability in retained:
                fate_rows.append({
                    "node_id": int(source), "terminal_node_id": int(terminal),
                    "fate_probability": float(probability),
                    "retained_probability_mass": float(retained_mass),
                    "lineage": str(lineage),
                })
        terminal_flow = sum(flow[node] for node in ids if terminal_flag[node])
        if not np.isclose(terminal_flow, 1.0, atol=1e-6):
            raise RuntimeError(f"Terminal Markov flow for {lineage} is {terminal_flow}, expected 1")

    annotated = nodes.copy()
    annotated["is_terminal"] = terminal_flag
    annotated["most_likely_terminal_id"] = most_likely
    annotated["most_likely_terminal_probability"] = most_probability
    annotated["fate_entropy"] = entropy
    return annotated, edges, pd.DataFrame(fate_rows)


def extract_tree(nodes, edges, stage, lineage_roots):
    """Extract the maximum-weight arborescence from the stage-directed DAG."""
    selected_indices = []
    root_set = {node for roots in lineage_roots.values() for node in roots}
    for target in nodes.node_id.to_numpy(dtype=np.int64):
        if int(target) in root_set:
            continue
        incoming = edges.index[edges.target_id.eq(target)].tolist()
        if not incoming:
            raise RuntimeError(f"Non-root node {target} has no earlier candidate parent")
        best = max(incoming, key=lambda index: (
            float(edges.at[index, "transition_probability"]),
            float(edges.at[index, "raw_weight"]),
            -float(edges.at[index, "cosine_distance"]),
            -int(edges.at[index, "source_id"])))
        selected_indices.append(best)
    edges = edges.copy()
    edges["selected_tree_edge"] = False
    edges.loc[selected_indices, "selected_tree_edge"] = True

    tree_rows = []
    for row in edges.loc[selected_indices].itertuples(index=False):
        tree_rows.append({
            "parent_id": int(row.source_id), "child_id": int(row.target_id),
            "lineage": str(row.lineage), "edge_kind": "node_knn",
            "virtual_edge": False, "cosine_distance": float(row.cosine_distance),
            "transition_probability": float(row.transition_probability),
            "markov_flow": float(row.markov_flow),
            "parent_stage": float(row.source_stage), "child_stage": float(row.target_stage),
            "stage_delta": float(row.stage_delta), "mutual_knn": bool(row.mutual_knn),
            "fallback_edge": bool(row.fallback_edge),
        })

    n_real = len(nodes)
    lineages = sorted(lineage_roots)
    lineage_virtual_ids = {lineage: n_real + position
                           for position, lineage in enumerate(lineages)}
    global_root_id = n_real + len(lineages)
    all_stage_range = float(stage.max() - stage.min())
    virtual_step = max(0.1, all_stage_range * 0.01)
    virtual_rows = []
    indexed_nodes = nodes.set_index("node_id")
    for lineage in lineages:
        virtual_id = lineage_virtual_ids[lineage]
        roots = np.asarray(lineage_roots[lineage], dtype=np.int64)
        lineage_stage = float(stage[roots].min()) - virtual_step
        virtual_rows.append({"node_id": virtual_id, "lineage": lineage,
                             "node_type": "lineage_root", "tree_stage": lineage_stage,
                             "n_metacells": 0, "n_cells": 0})
        root_weights = indexed_nodes.loc[roots, "n_cells"].to_numpy(dtype=float)
        root_weights /= root_weights.sum()
        for child, probability in zip(roots, root_weights):
            tree_rows.append({
                "parent_id": virtual_id, "child_id": int(child), "lineage": lineage,
                "edge_kind": "lineage_root", "virtual_edge": True,
                "cosine_distance": np.nan, "transition_probability": float(probability),
                "markov_flow": float(probability), "parent_stage": lineage_stage,
                "child_stage": float(stage[child]), "stage_delta": float(stage[child] - lineage_stage),
                "mutual_knn": False, "fallback_edge": False,
            })
    global_stage = min(row["tree_stage"] for row in virtual_rows) - virtual_step
    virtual_rows.append({"node_id": global_root_id, "lineage": "Global",
                         "node_type": "global_root", "tree_stage": global_stage,
                         "n_metacells": 0, "n_cells": 0})
    for lineage in lineages:
        child = lineage_virtual_ids[lineage]
        child_stage = next(row["tree_stage"] for row in virtual_rows if row["node_id"] == child)
        tree_rows.append({
            "parent_id": global_root_id, "child_id": child, "lineage": lineage,
            "edge_kind": "global_root", "virtual_edge": True,
            "cosine_distance": np.nan, "transition_probability": 1.0 / len(lineages),
            "markov_flow": 1.0 / len(lineages), "parent_stage": global_stage,
            "child_stage": child_stage, "stage_delta": child_stage - global_stage,
            "mutual_knn": False, "fallback_edge": False,
        })

    real_nodes = nodes.copy()
    real_nodes["node_type"] = "temporal_node"
    real_nodes["tree_stage"] = stage
    virtual = pd.DataFrame(virtual_rows)
    for column in real_nodes.columns:
        if column not in virtual:
            virtual[column] = np.nan
    virtual = virtual[real_nodes.columns]
    tree_nodes = pd.concat([real_nodes, virtual], ignore_index=True)
    tree_nodes["node_id"] = tree_nodes.node_id.astype(np.int64)
    tree_edges = pd.DataFrame(tree_rows).sort_values(
        ["parent_stage", "parent_id", "child_id"]).reset_index(drop=True)
    validate_tree(tree_nodes, tree_edges, global_root_id)
    return tree_nodes, tree_edges, edges, global_root_id, lineage_virtual_ids


def validate_tree(nodes, edges, global_root):
    if len(edges) != len(nodes) - 1:
        raise RuntimeError(f"Tree has {len(edges)} edges for {len(nodes)} nodes")
    indegree = edges.groupby("child_id").size().to_dict()
    for node in nodes.node_id:
        expected = 0 if int(node) == global_root else 1
        if indegree.get(int(node), 0) != expected:
            raise RuntimeError(f"Node {node} has invalid tree indegree")
    real = edges[~edges.virtual_edge]
    if not (real.stage_delta > 0).all():
        raise RuntimeError("A real tree edge does not point strictly forward in stage")
    children = defaultdict(list)
    for row in edges.itertuples(index=False):
        children[int(row.parent_id)].append(int(row.child_id))
    visited, stack = set(), [global_root]
    while stack:
        node = stack.pop()
        if node in visited:
            raise RuntimeError("Tree contains a cycle")
        visited.add(node)
        stack.extend(children[node])
    if len(visited) != len(nodes):
        raise RuntimeError(f"Tree reaches {len(visited)}/{len(nodes)} nodes")


def compress_branches(tree_nodes, tree_edges):
    node_lookup = tree_nodes.set_index("node_id")
    children = defaultdict(list)
    indegree = defaultdict(int)
    for row in tree_edges.itertuples(index=False):
        children[int(row.parent_id)].append(int(row.child_id))
        indegree[int(row.child_id)] += 1
    for parent in children:
        children[parent].sort()
    virtual = set(tree_nodes.loc[tree_nodes.node_type.ne("temporal_node"), "node_id"].astype(int))
    key_nodes = {int(node) for node in tree_nodes.node_id
                 if int(node) in virtual or indegree[int(node)] != 1 or len(children[int(node)]) != 1}
    rows = []
    for start in sorted(key_nodes):
        for first_child in children[start]:
            path = [start, first_child]
            current = first_child
            while current not in key_nodes:
                current = children[current][0]
                path.append(current)
            real_path = [node for node in path if node not in virtual]
            real_frame = node_lookup.loc[real_path] if real_path else pd.DataFrame()
            lineage_values = sorted(set(node_lookup.loc[path, "lineage"].astype(str)) - {"Global"})
            purity = np.nan
            if len(real_path) and "celltype_purity" in real_frame:
                valid = real_frame.celltype_purity.notna()
                if valid.any():
                    purity = float(np.average(real_frame.loc[valid, "celltype_purity"],
                                              weights=real_frame.loc[valid, "n_cells"]))
            rows.append({
                "segment_id": len(rows), "start_node_id": int(start),
                "end_node_id": int(path[-1]), "path_node_ids": json.dumps(path),
                "n_edges": len(path) - 1, "n_real_nodes": len(real_path),
                "lineage": lineage_values[0] if len(lineage_values) == 1 else "Global",
                "start_stage": float(node_lookup.at[start, "tree_stage"]),
                "end_stage": float(node_lookup.at[path[-1], "tree_stage"]),
                "stage_span": float(node_lookup.at[path[-1], "tree_stage"] -
                                    node_lookup.at[start, "tree_stage"]),
                "n_metacells": int(real_frame.n_metacells.sum()) if len(real_path) else 0,
                "n_cells": int(real_frame.n_cells.sum()) if len(real_path) else 0,
                "cell_weighted_celltype_purity": purity,
            })
    return pd.DataFrame(rows)


def summarize_lineages(nodes, candidate_edges, tree_edges, segments):
    rows = []
    for lineage, frame in nodes.groupby("lineage", sort=True):
        candidate = candidate_edges[candidate_edges.lineage.eq(str(lineage))]
        selected = tree_edges[
            tree_edges.lineage.eq(str(lineage)) & ~tree_edges.virtual_edge]
        local_segments = segments[segments.lineage.eq(str(lineage))]
        rows.append({
            "lineage": str(lineage),
            "n_nodes": int(len(frame)),
            "n_candidate_edges": int(len(candidate)),
            "n_mutual_knn_edges": int(candidate.mutual_knn.sum()),
            "n_fallback_candidate_edges": int(candidate.fallback_edge.sum()),
            "n_tree_edges": int(len(selected)),
            "n_fallback_tree_edges": int(selected.fallback_edge.sum()),
            "n_terminal_nodes": int(frame.is_terminal.sum()),
            "n_branch_segments": int(len(local_segments)),
            "median_tree_cosine_distance": (
                float(selected.cosine_distance.median()) if len(selected) else None),
            "max_tree_cosine_distance": (
                float(selected.cosine_distance.max()) if len(selected) else None),
            "median_tree_stage_delta": (
                float(selected.stage_delta.median()) if len(selected) else None),
            "max_tree_stage_delta": (
                float(selected.stage_delta.max()) if len(selected) else None),
        })
    return pd.DataFrame(rows)


def attach_display_umap(tree_nodes, leiden_root, microcell_dir):
    """Aggregate metacell UMAP positions to nodes for plotting only."""
    assignment_path = leiden_root / "microcell_node_assignments.parquet"
    if microcell_dir is None:
        config_path = leiden_root / "run_config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
            configured = config.get("input_dir")
            if configured:
                microcell_dir = Path(configured)
    if microcell_dir is None:
        log("no microcell directory found; skipping UMAP tree overlay")
        return tree_nodes, False
    coordinate_path = microcell_dir.resolve() / "microcell_umap_coordinates.parquet"
    if not assignment_path.exists() or not coordinate_path.exists():
        log("node assignments or microcell UMAP coordinates are absent; skipping UMAP tree overlay")
        return tree_nodes, False
    log(f"aggregating display-only node UMAP coordinates from {coordinate_path}")
    assignment = pd.read_parquet(
        assignment_path, columns=["microcell_id", "node_id", "n_cells"])
    coordinates = pd.read_parquet(
        coordinate_path, columns=["microcell_id", "UMAP1", "UMAP2"])
    merged = assignment.merge(
        coordinates, on="microcell_id", how="left", validate="one_to_one")
    if merged[["UMAP1", "UMAP2"]].isna().any().any():
        raise ValueError("Some assigned metacells have no saved UMAP coordinates")
    real_count = int(tree_nodes.node_type.eq("temporal_node").sum())
    labels = merged.node_id.to_numpy(dtype=np.int64)
    if labels.max() >= real_count or labels.min() < -1:
        raise ValueError("microcell_node_assignments.parquet contains invalid node_id")
    retained = labels >= 0
    if not retained.any():
        raise ValueError("No metacells pass the temporal-node support filter")
    labels = labels[retained]
    merged = merged.loc[retained].reset_index(drop=True)
    weights = merged.n_cells.to_numpy(dtype=np.float64)
    totals = np.bincount(labels, weights=weights, minlength=real_count)
    if np.any(totals <= 0):
        raise ValueError("At least one temporal node has no metacell UMAP members")
    x = np.bincount(
        labels, weights=weights * merged.UMAP1.to_numpy(), minlength=real_count) / totals
    y = np.bincount(
        labels, weights=weights * merged.UMAP2.to_numpy(), minlength=real_count) / totals
    result = tree_nodes.copy()
    result["node_umap1"] = np.nan
    result["node_umap2"] = np.nan
    result.loc[result.node_id.lt(real_count), "node_umap1"] = x
    result.loc[result.node_id.lt(real_count), "node_umap2"] = y
    return result, True


def tree_layout(tree_nodes, tree_edges, lineage_virtual_ids, global_root):
    children = defaultdict(list)
    for row in tree_edges.itertuples(index=False):
        children[int(row.parent_id)].append(int(row.child_id))
    stage = tree_nodes.set_index("node_id").tree_stage.to_dict()
    x = {}
    for band, lineage in enumerate(sorted(lineage_virtual_ids)):
        root = lineage_virtual_ids[lineage]
        descendants, stack = [], [root]
        while stack:
            node = stack.pop()
            descendants.append(node)
            stack.extend(children[node])
        leaves = sorted([node for node in descendants if not children[node]],
                        key=lambda node: (stage[node], node))
        if len(leaves) == 1:
            x[leaves[0]] = band + 0.5
        else:
            for position, leaf in enumerate(leaves):
                x[leaf] = band + 0.05 + 0.9 * position / (len(leaves) - 1)
        for node in sorted(descendants, key=lambda value: (stage[value], value), reverse=True):
            if node not in x:
                x[node] = float(np.mean([x[child] for child in children[node]]))
    x[global_root] = float(np.mean([x[root] for root in lineage_virtual_ids.values()]))
    result = tree_nodes.copy()
    result["tree_x"] = result.node_id.map(x).astype(float)
    result["tree_y"] = result.tree_stage.astype(float)
    return result


def plot_outputs(nodes, edges, candidate_edges, output, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    real = nodes.node_type.eq("temporal_node")
    lineages = sorted(nodes.loc[real, "lineage"].astype(str).unique())
    palette = dict(zip(lineages, plt.get_cmap("tab20")(np.linspace(0, 1, len(lineages)))))

    def discrete_colors(size):
        colors, seen = [], set()
        for name in ("tab20", "tab20b", "tab20c", "Set1", "Set2", "Set3",
                     "Dark2", "Paired", "Accent"):
            for value in plt.get_cmap(name).colors:
                color = matplotlib.colors.to_hex(value).upper()
                if color not in seen and color not in {"#FFFFFF", "#000000", "#B0B0B0"}:
                    seen.add(color)
                    colors.append(color)
        if len(colors) < size:
            # Deterministic golden-ratio hues extend the qualitative palettes.
            golden = 0.618033988749895
            index = 0
            while len(colors) < size:
                hue = (index * golden) % 1.0
                saturation = (0.62, 0.78, 0.9)[index % 3]
                value = (0.68, 0.82, 0.94)[(index // 3) % 3]
                color = matplotlib.colors.to_hex(
                    matplotlib.colors.hsv_to_rgb((hue, saturation, value))).upper()
                if color not in seen:
                    seen.add(color)
                    colors.append(color)
                index += 1
        return colors[:size]

    def save(fig, name):
        fig.savefig(output / f"{name}.png", dpi=dpi, bbox_inches="tight")
        fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    lookup = nodes.set_index("node_id")
    lineage_centers = np.arange(len(lineages), dtype=float) + 0.5
    fig, ax = plt.subplots(figsize=(17, 13))
    for edge in edges.itertuples(index=False):
        ax.plot([lookup.at[edge.parent_id, "tree_y"], lookup.at[edge.child_id, "tree_y"]],
                [lookup.at[edge.parent_id, "tree_x"], lookup.at[edge.child_id, "tree_x"]],
                color="#999999", linewidth=.35, alpha=.55, zorder=1)
    for lineage in lineages:
        mask = real & nodes.lineage.eq(lineage)
        ax.scatter(nodes.loc[mask, "tree_y"], nodes.loc[mask, "tree_x"], s=7,
                   color=palette[lineage], linewidths=0, zorder=2)
    virtual = ~real
    ax.scatter(nodes.loc[virtual, "tree_y"], nodes.loc[virtual, "tree_x"], marker="*",
               s=55, color="black", zorder=3)
    ax.set(xlabel="Predicted stage", ylabel="Lineage-separated branch layout",
           title="Markov node tree — lineage (early to late, left to right)")
    ax.set_yticks(lineage_centers, lineages, fontsize=8)
    ax.grid(axis="x", color="#EEEEEE", linewidth=.5)
    save(fig, "tree_by_lineage")

    fig, ax = plt.subplots(figsize=(17, 13))
    for edge in edges.itertuples(index=False):
        ax.plot([lookup.at[edge.parent_id, "tree_y"], lookup.at[edge.child_id, "tree_y"]],
                [lookup.at[edge.parent_id, "tree_x"], lookup.at[edge.child_id, "tree_x"]],
                color="#AAAAAA", linewidth=.3, alpha=.45, zorder=1)
    points = ax.scatter(nodes.loc[real, "tree_y"], nodes.loc[real, "tree_x"],
                        c=nodes.loc[real, "tree_stage"], cmap="viridis", s=7,
                        linewidths=0, zorder=2)
    fig.colorbar(points, ax=ax, label="Predicted stage")
    ax.set(xlabel="Predicted stage", ylabel="Lineage-separated branch layout",
           title="Markov node tree — predicted stage (early to late, left to right)")
    ax.set_yticks(lineage_centers, lineages, fontsize=8)
    ax.grid(axis="x", color="#EEEEEE", linewidth=.5)
    save(fig, "tree_by_stage")

    if "dominant_celltype" in nodes:
        labels = nodes.loc[real, "dominant_celltype"].fillna("Unknown").astype(str)
        weights = (nodes.loc[real].assign(_label=labels)
                   .groupby("_label", sort=False).n_cells.sum().sort_values(ascending=False))
        celltypes = weights.index.tolist()
        colors = dict(zip(celltypes, discrete_colors(len(celltypes))))
        legend_columns = min(8, max(1, len(celltypes)))
        fig, ax = plt.subplots(figsize=(18, 14))
        for edge in edges.itertuples(index=False):
            ax.plot([lookup.at[edge.parent_id, "tree_y"], lookup.at[edge.child_id, "tree_y"]],
                    [lookup.at[edge.parent_id, "tree_x"], lookup.at[edge.child_id, "tree_x"]],
                    color="#BBBBBB", linewidth=.3, alpha=.4, zorder=1)
        color_values = [colors[label] for label in labels]
        ax.scatter(nodes.loc[real, "tree_y"], nodes.loc[real, "tree_x"],
                   c=color_values, s=7, linewidths=0, zorder=2)
        handles = [Line2D([0], [0], marker="o", linestyle="", color=colors[label], label=label)
                   for label in celltypes]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08),
                  fontsize=6, ncol=legend_columns, markerscale=1.35,
                  columnspacing=.8, handletextpad=.3)
        ax.set(xlabel="Predicted stage", ylabel="Lineage-separated branch layout",
               title=f"Markov node tree — all {len(celltypes)} dominant cell types "
                     "(early to late, left to right; evaluation only)")
        ax.set_yticks(lineage_centers, lineages, fontsize=8)
        ax.grid(axis="x", color="#EEEEEE", linewidth=.5)
        save(fig, "tree_by_dominant_celltype")

        pd.DataFrame({
            "celltype": celltypes,
            "color": [colors[label] for label in celltypes],
            "n_cells": [int(weights[label]) for label in celltypes],
        }).to_csv(output / "celltype_colors.csv", index=False)

        lineage_output = output / "trees_by_lineage_celltype"
        lineage_output.mkdir(exist_ok=True)
        for position, lineage in enumerate(lineages, 1):
            local_mask = real & nodes.lineage.eq(lineage)
            local_labels = nodes.loc[local_mask, "dominant_celltype"].fillna("Unknown").astype(str)
            local_celltypes = [label for label in celltypes if label in set(local_labels)]
            local_legend_columns = max(1, math.ceil(len(local_celltypes) / 30))
            fig, ax = plt.subplots(figsize=(12 + 3 * local_legend_columns, 10))
            local_edges = edges[
                edges.lineage.eq(lineage) & edges.edge_kind.ne("global_root")]
            for edge in local_edges.itertuples(index=False):
                ax.plot([lookup.at[edge.parent_id, "tree_x"],
                         lookup.at[edge.child_id, "tree_x"]],
                        [lookup.at[edge.parent_id, "tree_y"],
                         lookup.at[edge.child_id, "tree_y"]],
                        color="#999999", linewidth=.55, alpha=.6, zorder=1)
            ax.scatter(
                nodes.loc[local_mask, "tree_x"], nodes.loc[local_mask, "tree_y"],
                c=[colors[label] for label in local_labels], s=14,
                linewidths=0, zorder=2)
            local_root = nodes.node_type.eq("lineage_root") & nodes.lineage.eq(lineage)
            ax.scatter(nodes.loc[local_root, "tree_x"], nodes.loc[local_root, "tree_y"],
                       marker="*", s=80, color="black", zorder=3, label="Lineage root")
            handles = [
                Line2D([0], [0], marker="o", linestyle="", color=colors[label], label=label)
                for label in local_celltypes]
            handles.append(Line2D([0], [0], marker="*", linestyle="", color="black",
                                  markersize=9, label="Lineage root"))
            ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1, .5),
                      fontsize=8, ncol=local_legend_columns, markerscale=1.25,
                      columnspacing=1.0, handletextpad=.35)
            ax.set(
                xlabel="Branch layout", ylabel="Predicted stage",
                title=f"{lineage} — all {len(local_celltypes)} dominant cell types")
            ax.invert_yaxis()
            safe_lineage = re.sub(r"[^A-Za-z0-9]+", "_", lineage).strip("_").lower()
            stem = lineage_output / f"{position:02d}_{safe_lineage}"
            fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
            fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
            plt.close(fig)

    if {"node_umap1", "node_umap2"}.issubset(nodes):
        fig, ax = plt.subplots(figsize=(14, 11))
        real_edges = edges[~edges.virtual_edge]
        for edge in real_edges.itertuples(index=False):
            ax.plot([lookup.at[edge.parent_id, "node_umap1"],
                     lookup.at[edge.child_id, "node_umap1"]],
                    [lookup.at[edge.parent_id, "node_umap2"],
                     lookup.at[edge.child_id, "node_umap2"]],
                    color="#555555", linewidth=.35, alpha=.35, zorder=1)
        for lineage in lineages:
            mask = real & nodes.lineage.eq(lineage)
            ax.scatter(nodes.loc[mask, "node_umap1"], nodes.loc[mask, "node_umap2"],
                       s=9, color=palette[lineage], linewidths=0,
                       label=lineage, zorder=2)
        ax.legend(loc="center left", bbox_to_anchor=(1, .5), fontsize=8, markerscale=2)
        ax.set(xlabel="UMAP 1", ylabel="UMAP 2",
               title="Final tree on metacell UMAP (display projection only)")
        save(fig, "tree_on_microcell_umap")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].hist(candidate_edges.cosine_distance, bins=50)
    axes[0, 0].set(xlabel="Candidate-edge cosine distance", ylabel="Edges")
    axes[0, 1].hist(candidate_edges.transition_probability, bins=50)
    axes[0, 1].set(xlabel="Markov transition probability", ylabel="Edges")
    axes[1, 0].hist(candidate_edges.markov_flow, bins=50)
    axes[1, 0].set(xlabel="Markov edge flow", ylabel="Edges")
    outdegree = candidate_edges.groupby("source_id").size().reindex(
        nodes.loc[real, "node_id"], fill_value=0)
    axes[1, 1].hist(outdegree, bins=np.arange(outdegree.max() + 2) - .5)
    axes[1, 1].set(xlabel="Directed candidate out-degree", ylabel="Nodes")
    fig.tight_layout()
    save(fig, "markov_graph_diagnostics")


def main(argv=None):
    args = parse_args(argv)
    if args.knn < 1:
        raise ValueError("--knn must be positive")
    if args.mutual_knn_bonus <= 0 or not np.isfinite(args.mutual_knn_bonus):
        raise ValueError("--mutual-knn-bonus must be positive and finite")
    if args.distance_temperature < 0 or not np.isfinite(args.distance_temperature):
        raise ValueError("--distance-temperature must be zero or positive and finite")
    if not 0 <= args.fate_probability_threshold <= 1:
        raise ValueError("--fate-probability-threshold must lie in [0, 1]")
    output = args.output_dir.resolve()
    if args.plots_only:
        if not output.is_dir():
            raise FileNotFoundError(f"Existing tree output directory not found: {output}")
        log(f"redrawing plots from existing tree tables in {output}")
        plot_outputs(
            pd.read_parquet(output / "tree_nodes.parquet"),
            pd.read_parquet(output / "tree_edges.parquet"),
            pd.read_parquet(output / "candidate_markov_edges.parquet"),
            output, args.dpi)
        log(f"plots updated: {output}")
        return
    if args.input_dir is None:
        raise ValueError("--input-dir is required unless --plots-only is used")
    root = args.input_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Output path already exists: {output}")
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        log(f"loading temporal nodes from {root}")
        nodes, embedding, stage = load_inputs(root, args.stage_column)
        log(f"building exact lineage-local cosine kNN graph for {len(nodes):,} nodes")
        candidate, lineage_roots, temperatures = build_candidate_graph(
            nodes, embedding, stage, args.knn, args.mutual_knn_bonus,
            args.distance_temperature)
        nodes, candidate, fate = markov_quantities(
            nodes, candidate, stage, lineage_roots, args.fate_probability_threshold)
        tree_nodes, tree_edges, candidate, global_root, lineage_virtual_ids = extract_tree(
            nodes, candidate, stage, lineage_roots)
        tree_nodes = tree_layout(tree_nodes, tree_edges, lineage_virtual_ids, global_root)
        tree_nodes, has_umap = attach_display_umap(tree_nodes, root, args.microcell_dir)
        segments = compress_branches(tree_nodes, tree_edges)
        lineage_summary = summarize_lineages(nodes, candidate, tree_edges, segments)

        candidate.to_parquet(temporary / "candidate_markov_edges.parquet", index=False)
        tree_nodes.to_parquet(temporary / "tree_nodes.parquet", index=False)
        tree_edges.to_parquet(temporary / "tree_edges.parquet", index=False)
        segments.to_parquet(temporary / "branch_segments.parquet", index=False)
        lineage_summary.to_csv(temporary / "lineage_tree_summary.csv", index=False)
        fate.to_parquet(temporary / "node_fate_probabilities.parquet", index=False)
        if has_umap:
            tree_nodes.loc[tree_nodes.node_type.eq("temporal_node"),
                           ["node_id", "node_umap1", "node_umap2"]].to_parquet(
                               temporary / "node_umap_coordinates.parquet", index=False)
        summary = {
            "n_temporal_nodes": int(len(nodes)),
            "n_lineages": int(nodes.lineage.nunique()),
            "n_candidate_edges": int(len(candidate)),
            "n_mutual_knn_edges": int(candidate.mutual_knn.sum()),
            "n_fallback_edges": int(candidate.fallback_edge.sum()),
            "n_selected_fallback_edges": int(
                tree_edges.loc[~tree_edges.virtual_edge, "fallback_edge"].sum()),
            "n_terminal_nodes": int(nodes.is_terminal.sum()),
            "n_tree_nodes_including_virtual": int(len(tree_nodes)),
            "n_tree_edges": int(len(tree_edges)),
            "n_branch_segments": int(len(segments)),
            "global_root_id": int(global_root),
            "lineage_virtual_root_ids": lineage_virtual_ids,
            "stage_column": args.stage_column,
            "stage_usage": "strict edge direction only; no maximum stage gap and no stage weight",
            "edge_weight": "exp(-cosine_distance / lineage_temperature), with optional mutual-kNN bonus",
            "tree_extraction": "maximum-weight arborescence on the stage-directed DAG",
            "same_state_continuity_edges": False,
            "celltype_usage": "output annotation and plotting only",
            "umap_usage": "display projection only" if has_umap else "not available",
        }
        (temporary / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        config = {key: str(value) if isinstance(value, Path) else value
                  for key, value in vars(args).items()}
        config["lineage_distance_temperatures"] = temperatures
        (temporary / "run_config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n")
        plot_outputs(tree_nodes, tree_edges, candidate, temporary, args.dpi)
        (temporary / "complete.json").write_text(json.dumps({"complete": True}) + "\n")
        temporary.rename(output)
        log(f"done: {output}")
        log(f"candidate edges={len(candidate):,}; terminals={nodes.is_terminal.sum():,}; "
            f"branch segments={len(segments):,}")
    except Exception:
        log(f"failed; partial output kept at {temporary}")
        raise


if __name__ == "__main__":
    main()
