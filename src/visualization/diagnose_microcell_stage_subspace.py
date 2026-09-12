#!/usr/bin/env python3
"""Select stage-correlated embedding dimensions and visualize top-k subspaces.

This is a supervised diagnostic, not an independent embedding evaluation.
Dimensions are ranked by absolute pooled within-lineage Spearman correlation
between L2-normalized microcell embeddings and aggregated true stage.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from plot_microcell_phate import aggregate_true_stage, discrete_palette, validate_inputs


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--predicted-stage",
        type=Path,
        default=Path("/mnt/input/sc_cz/Concord/eval/2026_09_03/predicted_stage.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, nargs="+", default=[6, 10])
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.25)
    parser.add_argument("--spread", type=float, default=1.0)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--stage-vmin", type=float, default=4.0)
    parser.add_argument("--stage-vmax", type=float, default=26.0)
    parser.add_argument("--point-size", type=float, default=0.35)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def column_pearson(values, target):
    values = values - values.mean(axis=0)
    target = target - target.mean()
    denominator = np.sqrt((values * values).sum(axis=0) * np.dot(target, target))
    return np.divide(
        values.T @ target,
        denominator,
        out=np.full(values.shape[1], np.nan),
        where=denominator > 0,
    )


def calculate_correlations(embedding, stage, lineage):
    global_pearson = column_pearson(embedding, stage)
    global_spearman = column_pearson(rankdata(embedding, axis=0), rankdata(stage))
    numerator = np.zeros(embedding.shape[1])
    denominator_x = np.zeros(embedding.shape[1])
    denominator_y = 0.0
    per_lineage = []
    groups = pd.Series(np.arange(len(lineage))).groupby(lineage, sort=True).indices
    for name, indices in groups.items():
        x_rank = rankdata(embedding[indices], axis=0)
        y_rank = rankdata(stage[indices])
        x_rank -= x_rank.mean(axis=0)
        y_rank -= y_rank.mean()
        x_ss = (x_rank * x_rank).sum(axis=0)
        y_ss = np.dot(y_rank, y_rank)
        values = np.divide(
            x_rank.T @ y_rank,
            np.sqrt(x_ss * y_ss),
            out=np.full(embedding.shape[1], np.nan),
            where=x_ss > 0,
        )
        per_lineage.append(
            pd.DataFrame(
                {"lineage": name, "dimension": np.arange(embedding.shape[1]), "spearman": values}
            )
        )
        numerator += x_rank.T @ y_rank
        denominator_x += x_ss
        denominator_y += y_ss
        print(f"correlation: {name} ({len(indices):,} microcells)", flush=True)
    per_lineage = pd.concat(per_lineage, ignore_index=True)
    within = numerator / np.sqrt(denominator_x * denominator_y)
    pivot = per_lineage.pivot(index="dimension", columns="lineage", values="spearman")
    summary = pd.DataFrame(
        {
            "dimension": np.arange(embedding.shape[1]),
            "global_pearson": global_pearson,
            "global_spearman": global_spearman,
            "within_lineage_spearman": within,
            "lineage_median_abs_spearman": pivot.abs().median(axis=1).to_numpy(),
            "lineage_positive_fraction": (pivot > 0).mean(axis=1).to_numpy(),
            "lineage_negative_fraction": (pivot < 0).mean(axis=1).to_numpy(),
        }
    )
    summary["lineage_same_sign_fraction"] = summary[
        ["lineage_positive_fraction", "lineage_negative_fraction"]
    ].max(axis=1)
    summary["abs_within_lineage_spearman"] = summary.within_lineage_spearman.abs()
    summary.sort_values("abs_within_lineage_spearman", ascending=False, inplace=True)
    return summary, per_lineage


def configure_gpu(device):
    import cupy as cp
    import rmm
    from rmm.allocators.cupy import rmm_cupy_allocator

    cp.cuda.Device(device).use()
    rmm.reinitialize(managed_memory=False, pool_allocator=False, devices=device)
    cp.cuda.set_allocator(rmm_cupy_allocator)
    return cp


def run_umap(values, n_neighbors, min_dist, spread, seed, cp):
    import rapids_singlecell as rsc

    adata = ad.AnnData(X=values)
    rsc.get.anndata_to_GPU(adata, convert_all=True)
    rsc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        n_pcs=values.shape[1],
        use_rep="X",
        metric="inner_product",
        algorithm="cagra",
        rng=seed,
    )
    rsc.tl.umap(
        adata,
        min_dist=min_dist,
        spread=spread,
        n_components=2,
        rng=seed,
        key_added="X_umap",
    )
    raw = adata.obsm["X_umap"]
    coordinates = cp.asnumpy(raw) if isinstance(raw, cp.ndarray) else np.asarray(raw)
    return np.asarray(coordinates, dtype=np.float32)


def save_stage(coordinates, stage, path, title, label, args, order):
    figure, axis = plt.subplots(figsize=(12, 10))
    points = axis.scatter(
        coordinates[order, 0], coordinates[order, 1], c=stage[order],
        cmap="plasma", vmin=args.stage_vmin, vmax=args.stage_vmax,
        s=args.point_size, alpha=0.75, linewidths=0, rasterized=True,
    )
    figure.colorbar(points, ax=axis, label=label)
    axis.set(title=title, xlabel="UMAP 1", ylabel="UMAP 2")
    axis.set_xticks([]); axis.set_yticks([])
    figure.savefig(path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def save_lineage(coordinates, lineage, path, title, args, order):
    categories = sorted(np.unique(lineage))
    palette = discrete_palette(len(categories))
    mapping = dict(zip(categories, palette))
    colors = np.asarray([mapping[value] for value in lineage], dtype=object)
    figure, axis = plt.subplots(figsize=(15, 10))
    axis.scatter(
        coordinates[order, 0], coordinates[order, 1], c=colors[order],
        s=args.point_size, alpha=0.75, linewidths=0, rasterized=True,
    )
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=5,
                      markerfacecolor=mapping[value], markeredgecolor="none", label=value)
               for value in categories]
    axis.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, .5), frameon=False, fontsize=8)
    axis.set(title=title, xlabel="UMAP 1", ylabel="UMAP 2")
    axis.set_xticks([]); axis.set_yticks([])
    figure.savefig(path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main():
    args = parse_args()
    root = args.input_dir.resolve()
    output = (args.output_dir or root / "stage_subspace_diagnostics").resolve()
    output.mkdir(parents=True, exist_ok=True)
    embedding_disk, metadata, assignment = validate_inputs(root)
    true_stage, _ = aggregate_true_stage(
        args.predicted_stage, assignment, metadata.n_cells.to_numpy(np.int64), args.chunk_size
    )
    embedding = np.array(embedding_disk, dtype=np.float64, copy=True)
    norms = np.linalg.norm(embedding, axis=1)
    if np.any(norms == 0):
        raise ValueError("zero-norm microcell embeddings found")
    embedding /= norms[:, None]
    lineage = metadata.lineage.fillna("Unknown").astype(str).to_numpy()
    summary, per_lineage = calculate_correlations(embedding, true_stage, lineage)
    summary.to_csv(output / "dimension_correlations.csv", index=False)
    per_lineage.to_csv(output / "dimension_correlations_by_lineage.csv", index=False)
    max_k = max(args.top_k)
    if min(args.top_k) < 2 or max_k > embedding.shape[1]:
        raise ValueError("top-k values must be between 2 and embedding dimension")
    ranked = summary.dimension.to_numpy(dtype=np.int64)
    json_config = {
        "selection": "absolute pooled within-lineage Spearman with aggregated true stage",
        "representation_for_selection": "full L2-normalized microcell embedding",
        "ranked_dimensions": ranked.tolist(),
        "top_k": args.top_k,
        "supervised_diagnostic": True,
    }
    (output / "selected_dimensions.json").write_text(json.dumps(json_config, indent=2) + "\n")
    print("top dimensions:", ranked[:max_k].tolist(), flush=True)

    cp = configure_gpu(args.gpu)
    predicted_stage = pd.to_numeric(metadata.mean_stage, errors="raise").to_numpy(float)
    order = np.random.default_rng(args.seed).permutation(len(metadata))
    for k in sorted(set(args.top_k)):
        dimensions = ranked[:k]
        stem = f"top{k:02d}_dims_" + "-".join(map(str, dimensions))
        coordinate_path = output / f"{stem}_umap.npy"
        if coordinate_path.exists() and not args.overwrite:
            coordinates = np.load(coordinate_path)
            print(f"reusing {coordinate_path}", flush=True)
        else:
            selected = np.asarray(embedding[:, dimensions], dtype=np.float32)
            selected_norm = np.linalg.norm(selected, axis=1)
            if np.any(selected_norm == 0):
                raise ValueError(f"top-{k} subspace contains zero-norm rows")
            selected /= selected_norm[:, None]
            print(f"GPU UMAP top-{k}: dimensions={dimensions.tolist()}", flush=True)
            coordinates = run_umap(
                selected, args.n_neighbors, args.min_dist, args.spread, args.seed, cp
            )
            np.save(coordinate_path, coordinates)
            del selected
            cp.get_default_memory_pool().free_all_blocks()
        table = pd.DataFrame({
            "microcell_id": metadata.microcell_id,
            "mean_true_stage": true_stage,
            "mean_predicted_stage": predicted_stage,
            "lineage": lineage,
            "UMAP1": coordinates[:, 0],
            "UMAP2": coordinates[:, 1],
        })
        table.to_parquet(output / f"{stem}_umap.parquet", index=False)
        save_stage(coordinates, true_stage, output / f"{stem}_true_stage.png",
                   f"Top-{k} stage subspace — mean true stage", "Mean true stage", args, order)
        save_stage(coordinates, predicted_stage, output / f"{stem}_predicted_stage.png",
                   f"Top-{k} stage subspace — mean predicted stage", "Mean predicted stage", args, order)
        save_lineage(coordinates, lineage, output / f"{stem}_lineage.png",
                     f"Top-{k} stage subspace — lineage", args, order)
    (output / "complete.json").write_text('{"complete": true}\n')
    print(f"done: {output}", flush=True)


if __name__ == "__main__":
    main()
