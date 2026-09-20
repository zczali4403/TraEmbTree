#!/usr/bin/env python3
"""Summarize temporal edge gaps, terminal stages, and contracted-node sizes."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tree-dir", type=Path, required=True,
        help="Contracted tree directory containing tree_nodes.parquet and tree_edges.parquet")
    parser.add_argument(
        "--output-dir", type=Path,
        help="Output directory (default: TREE_DIR/tree_statistics)")
    parser.add_argument(
        "--short-edge-threshold", type=float, default=1.0,
        help="Count real parent-child edges with a stage gap below this value (default: 1.0)")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def quantiles(values):
    values = np.asarray(values, dtype=float)
    return {
        "q10": float(np.quantile(values, .10)),
        "q25": float(np.quantile(values, .25)),
        "median": float(np.quantile(values, .50)),
        "q75": float(np.quantile(values, .75)),
        "q90": float(np.quantile(values, .90)),
    }


def load_tree(tree_dir):
    node_path = tree_dir / "tree_nodes.parquet"
    edge_path = tree_dir / "tree_edges.parquet"
    if not node_path.is_file() or not edge_path.is_file():
        raise FileNotFoundError(
            f"Expected tree_nodes.parquet and tree_edges.parquet in {tree_dir}")

    nodes = pd.read_parquet(node_path)
    edges = pd.read_parquet(edge_path)
    required_nodes = {
        "node_id", "node_type", "lineage", "n_cells",
        "cell_weighted_mean_stage",
    }
    required_edges = {"parent_id", "child_id", "lineage"}
    if not required_nodes.issubset(nodes):
        raise ValueError(
            f"tree_nodes.parquet is missing {sorted(required_nodes - set(nodes))}")
    if not required_edges.issubset(edges):
        raise ValueError(
            f"tree_edges.parquet is missing {sorted(required_edges - set(edges))}")
    return nodes, edges


def prepare_tables(nodes, edges, threshold):
    real = nodes.loc[nodes.node_type.eq("temporal_node")].copy()
    real["node_id"] = real.node_id.astype(np.int64)
    real["n_cells"] = real.n_cells.astype(np.int64)
    real_ids = set(real.node_id)

    real_edges = edges.loc[
        edges.parent_id.isin(real_ids) & edges.child_id.isin(real_ids)
    ].copy()
    lookup = real.set_index("node_id")
    real_edges["parent_stage"] = real_edges.parent_id.map(
        lookup.cell_weighted_mean_stage)
    real_edges["child_stage"] = real_edges.child_id.map(
        lookup.cell_weighted_mean_stage)
    real_edges["signed_stage_delta"] = (
        real_edges.child_stage - real_edges.parent_stage)
    real_edges["stage_gap_weeks"] = real_edges.signed_stage_delta.abs()
    real_edges["below_threshold"] = (
        real_edges.stage_gap_weeks < threshold)
    real_edges = real_edges[[
        "parent_id", "child_id", "lineage", "parent_stage", "child_stage",
        "signed_stage_delta", "stage_gap_weeks", "below_threshold",
    ]].sort_values(["lineage", "parent_stage", "child_stage"])

    children = set(real_edges.parent_id.astype(np.int64))
    terminals = real.loc[~real.node_id.isin(children)].copy()
    terminal_columns = [
        "node_id", "lineage", "cell_weighted_mean_stage", "n_cells"]
    for optional in ("dominant_celltype", "celltype_purity"):
        if optional in terminals:
            terminal_columns.append(optional)
    terminals = terminals[terminal_columns].rename(columns={
        "cell_weighted_mean_stage": "terminal_stage",
    }).sort_values(["lineage", "terminal_stage", "node_id"])

    node_columns = [
        "node_id", "lineage", "cell_weighted_mean_stage", "n_cells"]
    for optional in (
            "n_metacells", "n_source_nodes", "dominant_celltype",
            "celltype_purity"):
        if optional in real:
            node_columns.append(optional)
    node_sizes = real[node_columns].rename(columns={
        "cell_weighted_mean_stage": "node_stage",
    }).sort_values(["lineage", "node_stage", "node_id"])
    return real, real_edges, terminals, node_sizes


def build_summary(nodes, edges, real, real_edges, terminals, threshold):
    n_short = int(real_edges.below_threshold.sum())
    n_real_edges = int(len(real_edges))
    summary = {
        "tree_directory": str(nodes.attrs.get("tree_directory", "")),
        "short_edge_threshold_weeks": float(threshold),
        "n_nodes_including_virtual": int(len(nodes)),
        "n_temporal_nodes": int(len(real)),
        "n_edges_including_virtual": int(len(edges)),
        "n_real_edges": n_real_edges,
        "n_edges_below_threshold": n_short,
        "fraction_edges_below_threshold": (
            float(n_short / n_real_edges) if n_real_edges else None),
        "n_nonforward_real_edges": int(
            (real_edges.signed_stage_delta < 0).sum()),
        "n_terminal_nodes": int(len(terminals)),
        "total_cells_across_nodes": int(real.n_cells.sum()),
        "edge_stage_gap_weeks": quantiles(real_edges.stage_gap_weeks),
        "terminal_stage": quantiles(terminals.terminal_stage),
        "node_cell_count": quantiles(real.n_cells),
        "min_node_cells": int(real.n_cells.min()),
        "max_node_cells": int(real.n_cells.max()),
    }
    return summary


def lineage_summary(real, real_edges, terminals, threshold):
    rows = []
    for lineage, local_nodes in real.groupby("lineage", sort=True):
        local_edges = real_edges.loc[real_edges.lineage.eq(lineage)]
        local_terminals = terminals.loc[terminals.lineage.eq(lineage)]
        n_edges = len(local_edges)
        n_short = int((local_edges.stage_gap_weeks < threshold).sum())
        rows.append({
            "lineage": lineage,
            "n_nodes": int(len(local_nodes)),
            "n_edges": int(n_edges),
            "n_edges_below_threshold": n_short,
            "fraction_edges_below_threshold": (
                n_short / n_edges if n_edges else np.nan),
            "n_terminal_nodes": int(len(local_terminals)),
            "median_terminal_stage": float(local_terminals.terminal_stage.median()),
            "median_node_cells": float(local_nodes.n_cells.median()),
            "total_node_cells": int(local_nodes.n_cells.sum()),
        })
    return pd.DataFrame(rows)


def plot_summary(real_edges, terminals, node_sizes, threshold, output, dpi):
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10,
        "axes.edgecolor": "#BCC4CB",
        "axes.linewidth": .8,
        "xtick.color": "#4D5660",
        "ytick.color": "#4D5660",
        "text.color": "#27313A",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7))
    fig.patch.set_facecolor("#F5F7F8")
    palette = ["#3F7CAC", "#E07A5F", "#4F8A6D"]

    ax = axes[0]
    gaps = real_edges.stage_gap_weeks.to_numpy(dtype=float)
    bins = np.linspace(0, max(float(gaps.max()), threshold) * 1.02, 36)
    ax.hist(gaps, bins=bins, color=palette[0], alpha=.88,
            edgecolor="white", linewidth=.35)
    ax.axvspan(0, threshold, color="#E07A5F", alpha=.11, linewidth=0)
    ax.axvline(threshold, color="#C4513C", linestyle="--", linewidth=1.25)
    n_short = int((gaps < threshold).sum())
    ax.text(
        .97, .95, f"gap < {threshold:g}: {n_short:,} / {len(gaps):,} "
        f"({n_short / len(gaps):.1%})",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox={"boxstyle": "round,pad=.35", "facecolor": "white",
              "edgecolor": "#D5DBE0", "alpha": .95})
    ax.set_title("A  Parent-child stage gaps", loc="left")
    ax.set_xlabel("Absolute mean-stage gap (weeks)")
    ax.set_ylabel("Number of edges")

    ax = axes[1]
    terminal_stage = terminals.terminal_stage.to_numpy(dtype=float)
    ax.hist(terminal_stage, bins=24, color=palette[1], alpha=.88,
            edgecolor="white", linewidth=.35)
    for value in terminal_stage:
        ax.plot([value, value], [0, -.018], color="#7B4336", alpha=.32,
                linewidth=.55, transform=ax.get_xaxis_transform(), clip_on=False)
    median = float(np.median(terminal_stage))
    ax.axvline(median, color="#873D2E", linestyle="--", linewidth=1.15)
    ax.text(
        .97, .95, f"n = {len(terminal_stage):,}\nmedian = {median:.2f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox={"boxstyle": "round,pad=.35", "facecolor": "white",
              "edgecolor": "#D5DBE0", "alpha": .95})
    ax.set_title("B  Terminal-node stages", loc="left")
    ax.set_xlabel("Terminal mean stage")
    ax.set_ylabel("Number of terminal nodes")

    ax = axes[2]
    counts = node_sizes.n_cells.to_numpy(dtype=float)
    lo, hi = counts.min(), counts.max()
    bins = np.geomspace(max(1, lo), hi * 1.001, 32)
    ax.hist(counts, bins=bins, color=palette[2], alpha=.9,
            edgecolor="white", linewidth=.35)
    ax.set_xscale("log")
    median = float(np.median(counts))
    ax.axvline(median, color="#285A43", linestyle="--", linewidth=1.15)
    ax.text(
        .97, .95,
        f"n = {len(counts):,}\nmedian = {median:,.0f}\n"
        f"range = {lo:,.0f}-{hi:,.0f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox={"boxstyle": "round,pad=.35", "facecolor": "white",
              "edgecolor": "#D5DBE0", "alpha": .95})
    ax.set_title("C  Cells per contracted node", loc="left")
    ax.set_xlabel("Number of cells (log scale)")
    ax.set_ylabel("Number of nodes")

    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(axis="y", color="#E7EBEE", linewidth=.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Contracted tree: temporal gaps, terminal timing, and node support",
        x=.055, ha="left", fontsize=15, fontweight="semibold")
    fig.subplots_adjust(left=.06, right=.985, bottom=.16, top=.83, wspace=.28)
    fig.savefig(output / "tree_statistics.png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    fig.savefig(output / "tree_statistics.pdf", bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    args = parse_args()
    if not np.isfinite(args.short_edge_threshold) or args.short_edge_threshold <= 0:
        raise ValueError("--short-edge-threshold must be positive and finite")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")

    tree_dir = args.tree_dir.resolve()
    output = (args.output_dir or tree_dir / "tree_statistics").resolve()
    output.mkdir(parents=True, exist_ok=True)
    nodes, edges = load_tree(tree_dir)
    nodes.attrs["tree_directory"] = str(tree_dir)
    real, real_edges, terminals, node_sizes = prepare_tables(
        nodes, edges, args.short_edge_threshold)
    summary = build_summary(
        nodes, edges, real, real_edges, terminals,
        args.short_edge_threshold)
    by_lineage = lineage_summary(
        real, real_edges, terminals, args.short_edge_threshold)

    real_edges.to_csv(output / "edge_stage_gaps.csv", index=False)
    terminals.to_csv(output / "terminal_nodes.csv", index=False)
    node_sizes.to_csv(output / "node_cell_counts.csv", index=False)
    by_lineage.to_csv(output / "lineage_summary.csv", index=False)
    with (output / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    plot_summary(
        real_edges, terminals, node_sizes, args.short_edge_threshold,
        output, args.dpi)

    print(json.dumps(summary, indent=2))
    print(f"Done: {output}")


if __name__ == "__main__":
    main()
