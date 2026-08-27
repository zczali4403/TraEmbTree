#!/usr/bin/env python3
"""Write one stage-oriented tree figure per lineage with categorical colors.

Example::

    /home/aiscuser/.conda/envs/train/bin/python plot_each_lineage_tree.py \
      --tree-dir /mnt/input/sc_cz/Concord/eval/2026_08_19/landmark_tree
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


def parse_args():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree-dir", type=Path, default=here / "landmark_tree")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--width", type=float, default=14.0)
    parser.add_argument("--height", type=float, default=14.0)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--label-nodes", action="store_true")
    parser.add_argument(
        "--local-stage-range", action="store_true",
        help="Use each lineage's own stage limits instead of shared global limits.",
    )
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "lineage"


def save_figure_with_retry(fig, path: Path, attempts: int = 3, **kwargs) -> None:
    """Handle transient disappearance of a mounted output directory."""
    for attempt in range(1, attempts + 1):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, **kwargs)
            return
        except FileNotFoundError:
            if attempt == attempts:
                raise
            print(f"Output path temporarily unavailable; retrying {path} ({attempt}/{attempts})", flush=True)
            time.sleep(1)


def qualitative_palette(count: int) -> list[str]:
    """Return fixed categorical colors; never sample a sequential gradient."""
    palette = []
    qualitative_maps = (
        "tab20", "tab20b", "tab20c", "Set3", "Paired", "Set1",
        "Set2", "Accent", "Dark2", "Pastel1", "Pastel2",
    )
    for name in qualitative_maps:
        cmap = plt.get_cmap(name)
        raw = getattr(cmap, "colors", [cmap(i) for i in range(cmap.N)])
        for color in raw:
            value = matplotlib.colors.to_hex(color)
            if value not in palette:
                palette.append(value)
    # Add named categorical colors only if the qualitative maps are exhausted.
    excluded = {"white", "snow", "whitesmoke", "floralwhite", "ivory", "honeydew"}
    for name, value in matplotlib.colors.CSS4_COLORS.items():
        if name.lower() in excluded:
            continue
        rgb = np.asarray(matplotlib.colors.to_rgb(value))
        if rgb.max() - rgb.min() < 0.12 or rgb.mean() > 0.88 or rgb.mean() < 0.15:
            continue
        value = matplotlib.colors.to_hex(value)
        if value not in palette:
            palette.append(value)
    if count > len(palette):
        raise ValueError(f"Need {count} colors, but categorical palette has {len(palette)}")
    return palette[:count]


def main():
    args = parse_args()
    tree_dir = args.tree_dir.resolve()
    output_dir = (args.output_dir or tree_dir / "trees_by_lineage").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    nodes = pd.read_parquet(tree_dir / "tree_nodes.parquet").sort_values("node_id")
    edges = pd.read_parquet(tree_dir / "tree_edges.parquet")
    if "celltype_purity" not in nodes.columns:
        raise ValueError("tree_nodes.parquet lacks required column: celltype_purity")
    purity = pd.to_numeric(nodes["celltype_purity"], errors="coerce")
    finite_purity = purity[np.isfinite(purity)]
    if finite_purity.empty:
        raise ValueError("celltype_purity contains no finite values")
    if ((finite_purity < 0) | (finite_purity > 1)).any():
        raise ValueError("celltype_purity must be between 0 and 1")
    celltype_column = "dominant_celltype"
    nodes[celltype_column] = nodes[celltype_column].fillna("Unknown").astype(str)
    global_min = float(nodes["tree_y_stage"].min())
    global_max = float(nodes["tree_y_stage"].max())
    manifest = []
    color_rows = []

    for lineage in sorted(nodes["lineage"].astype(str).unique()):
        subset = nodes[nodes["lineage"].astype(str) == lineage].copy()
        local_celltypes = sorted(subset[celltype_column].unique())
        local_colors = dict(zip(local_celltypes, qualitative_palette(len(local_celltypes))))
        if "Unknown" in local_colors:
            local_colors["Unknown"] = "#B0B0B0"
        color_rows.extend(
            {"lineage": lineage, "celltype": label, "color": local_colors[label]}
            for label in local_celltypes
        )
        ids = subset["node_id"].to_numpy(dtype=np.int64)
        id_set = set(ids.tolist())
        local_edges = edges[
            edges["parent_id"].isin(id_set)
            & edges["child_id"].isin(id_set)
            & ~edges["cross_lineage"]
        ]
        raw_x = subset["tree_x"].to_numpy(dtype=float)
        x = ((raw_x - raw_x.min()) / (raw_x.max() - raw_x.min())) if raw_x.max() > raw_x.min() else np.full(len(raw_x), 0.5)
        x_by_id = dict(zip(ids, x))
        y_by_id = dict(zip(ids, subset["tree_y_stage"].to_numpy(dtype=float)))
        fig, ax = plt.subplots(figsize=(args.width, args.height))
        for edge in local_edges.itertuples(index=False):
            parent, child = int(edge.parent_id), int(edge.child_id)
            ax.plot([x_by_id[parent], x_by_id[child]], [y_by_id[parent], y_by_id[child]],
                    color="#9E9E9E", linewidth=1.25, alpha=0.85, zorder=1)
        sizes = 55.0 + 190.0 * np.sqrt(subset["n_cells"].to_numpy() / nodes["n_cells"].max())
        ax.scatter(x, subset["tree_y_stage"], s=sizes,
                   c=subset[celltype_column].map(local_colors), edgecolors="white",
                   linewidths=0.7, zorder=2)
        if args.label_nodes:
            for node_id, xv, yv in zip(ids, x, subset["tree_y_stage"]):
                ax.text(xv, yv, str(node_id), fontsize=7, ha="center", va="bottom")
        stage_min = float(subset["tree_y_stage"].min()) if args.local_stage_range else global_min
        stage_max = float(subset["tree_y_stage"].max()) if args.local_stage_range else global_max
        padding = max(0.1, (stage_max - stage_min) * 0.03)
        ax.set_ylim(stage_max + padding, stage_min - padding)
        ax.set_xlim(-0.08, 1.08)
        ax.set_xticks([])
        ax.set_ylabel("Mean developmental stage (early to late)")
        ax.set_xlabel("Within-lineage tree topology")
        ax.set_title(f"{lineage} landmark tree (n={len(subset)} nodes)", fontsize=15)
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
        handles = [Line2D([0], [0], marker="o", color="none",
                          markerfacecolor=local_colors[label], markeredgecolor="none",
                          markersize=7, label=label) for label in local_celltypes]
        ax.legend(handles=handles, title="Cell type", loc="upper left",
                  bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8,
                  title_fontsize=9, ncol=max(1, int(np.ceil(len(local_celltypes) / 30))))
        fig.tight_layout()
        stem = output_dir / safe_name(lineage)
        save_figure_with_retry(fig, stem.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
        save_figure_with_retry(fig, stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)

        purity_values = pd.to_numeric(subset["celltype_purity"], errors="coerce").to_numpy(dtype=float)
        purity_cmap = matplotlib.colormaps["viridis"].copy()
        purity_cmap.set_bad("#B0B0B0")
        fig, ax = plt.subplots(figsize=(args.width, args.height))
        for edge in local_edges.itertuples(index=False):
            parent, child = int(edge.parent_id), int(edge.child_id)
            ax.plot([x_by_id[parent], x_by_id[child]], [y_by_id[parent], y_by_id[child]],
                    color="#9E9E9E", linewidth=1.25, alpha=0.85, zorder=1)
        points = ax.scatter(
            x, subset["tree_y_stage"], s=sizes,
            c=np.ma.masked_invalid(purity_values), cmap=purity_cmap, vmin=0.0, vmax=1.0,
            edgecolors="white", linewidths=0.7, zorder=2,
        )
        if args.label_nodes:
            for node_id, xv, yv in zip(ids, x, subset["tree_y_stage"]):
                ax.text(xv, yv, str(node_id), fontsize=7, ha="center", va="bottom")
        ax.set_ylim(stage_max + padding, stage_min - padding)
        ax.set_xlim(-0.08, 1.08)
        ax.set_xticks([])
        ax.set_ylabel("Mean developmental stage (early to late)")
        ax.set_xlabel("Within-lineage tree topology")
        ax.set_title(f"{lineage} landmark tree by cell-type purity (n={len(subset)} nodes)", fontsize=15)
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
        colorbar = fig.colorbar(points, ax=ax, pad=0.015, fraction=0.035)
        colorbar.set_label("Cell-type purity")
        fig.tight_layout()
        purity_stem = output_dir / f"{safe_name(lineage)}_celltype_purity"
        save_figure_with_retry(fig, purity_stem.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
        save_figure_with_retry(fig, purity_stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        manifest.append({"lineage": lineage, "n_nodes": len(subset),
                         "n_celltypes": len(local_celltypes), "file_stem": stem.name})
        print(f"Saved {stem.with_suffix('.png')}", flush=True)

    pd.DataFrame(manifest).to_csv(output_dir / "manifest.csv", index=False)
    pd.DataFrame(color_rows).to_csv(output_dir / "celltype_colors.csv", index=False)


if __name__ == "__main__":
    main()
