"""Compute and plot 2D/3D diffusion maps for TraEmbTree microcells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import pearsonr, spearmanr


STANDARD_PAIRS = ((0, 1), (0, 2), (1, 2))


def stage_correlations(components: np.ndarray, stage: np.ndarray) -> pd.DataFrame:
    rows = []
    for index in range(components.shape[1]):
        values = components[:, index]
        rows.append(
            {
                "component": f"DC{index + 1}",
                "component_index": index,
                "eigenvalue_order": index + 1,
                "stage_pearson": float(pearsonr(values, stage)[0]),
                "stage_spearman": float(spearmanr(values, stage)[0]),
            }
        )
    table = pd.DataFrame(rows)
    table["abs_stage_pearson"] = table["stage_pearson"].abs()
    table["abs_stage_spearman"] = table["stage_spearman"].abs()
    return table.sort_values(
        "abs_stage_spearman", ascending=False, ignore_index=True
    )


def plot_order(n_points: int, max_points: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if max_points and n_points > max_points:
        return rng.choice(n_points, max_points, replace=False)
    return rng.permutation(n_points)


def plot_stage_panels(
    components: np.ndarray,
    stage: np.ndarray,
    pairs: tuple[tuple[int, int], ...],
    outpath: Path,
    point_size: float,
    vmin: float,
    vmax: float,
    seed: int,
    max_points: int,
    title: str,
) -> None:
    order = plot_order(len(components), max_points, seed)
    colors = np.clip(stage, vmin, vmax)
    fig, axes = plt.subplots(
        1, len(pairs), figsize=(7 * len(pairs), 6), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    points = None
    for ax, (left, right) in zip(axes, pairs):
        points = ax.scatter(
            components[order, left],
            components[order, right],
            c=colors[order],
            cmap="plasma",
            vmin=vmin,
            vmax=vmax,
            s=point_size,
            alpha=0.75,
            linewidths=0,
            rasterized=True,
        )
        ax.set_xlabel(f"DC{left + 1}")
        ax.set_ylabel(f"DC{right + 1}")
        ax.set_title(f"DC{left + 1} vs DC{right + 1}")
    fig.colorbar(
        points,
        ax=axes.tolist(),
        label=f"Predicted stage (clipped {vmin:g}-{vmax:g})",
        shrink=0.85,
    )
    fig.suptitle(title, fontsize=16)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def category_codes(values: np.ndarray) -> tuple[np.ndarray, list[str]]:
    labels = pd.Series(values).fillna("Unknown").astype(str).to_numpy()
    names = sorted(np.unique(labels))
    mapping = {name: index for index, name in enumerate(names)}
    codes = np.fromiter((mapping[label] for label in labels), dtype=np.int32)
    return codes, names


def category_cmap(n_categories: int):
    if n_categories <= 20:
        return plt.get_cmap("tab20", max(n_categories, 1))
    return plt.get_cmap("turbo", max(n_categories, 1))


def add_category_legend(ax, names: list[str], cmap) -> None:
    if len(names) > 40:
        text_method = getattr(ax, "text2D", ax.text)
        text_method(
            1.02,
            1.0,
            f"{len(names)} categories; see color CSV",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )
        return
    handles = [
        plt.Line2D(
            [], [], marker="o", linestyle="", color=cmap(i), markersize=5, label=name
        )
        for i, name in enumerate(names)
    ]
    ax.legend(
        handles=handles,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
        fontsize=7,
    )


def save_category_colors(names: list[str], cmap, outpath: Path) -> None:
    pd.DataFrame(
        {
            "category": names,
            "color": [matplotlib.colors.to_hex(cmap(i)) for i in range(len(names))],
        }
    ).to_csv(outpath, index=False)


def plot_category_panels(
    components: np.ndarray,
    values: np.ndarray,
    category: str,
    outpath: Path,
    colors_path: Path,
    point_size: float,
    seed: int,
    max_points: int,
) -> None:
    order = plot_order(len(components), max_points, seed)
    codes, names = category_codes(values)
    cmap = category_cmap(len(names))
    save_category_colors(names, cmap, colors_path)
    fig, axes = plt.subplots(1, 3, figsize=(23, 6), constrained_layout=True)
    for ax, (left, right) in zip(axes, STANDARD_PAIRS):
        ax.scatter(
            components[order, left],
            components[order, right],
            c=codes[order],
            cmap=cmap,
            vmin=-0.5,
            vmax=max(len(names) - 0.5, 0.5),
            s=point_size,
            alpha=0.7,
            linewidths=0,
            rasterized=True,
        )
        ax.set_xlabel(f"DC{left + 1}")
        ax.set_ylabel(f"DC{right + 1}")
        ax.set_title(f"DC{left + 1} vs DC{right + 1}")
    add_category_legend(axes[-1], names, cmap)
    fig.suptitle(f"Microcell diffusion map by {category}", fontsize=16)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def style_3d_axis(ax, indices: tuple[int, int, int], title: str) -> None:
    ax.set_xlabel(f"DC{indices[0] + 1}")
    ax.set_ylabel(f"DC{indices[1] + 1}")
    ax.set_zlabel(f"DC{indices[2] + 1}")
    ax.set_title(title)
    ax.grid(False)


def plot_3d_stage(
    components: np.ndarray,
    stage: np.ndarray,
    indices: tuple[int, int, int],
    outpath: Path,
    point_size: float,
    vmin: float,
    vmax: float,
    seed: int,
    max_points: int,
    title: str,
) -> None:
    order = plot_order(len(components), max_points, seed)
    colors = np.clip(stage, vmin, vmax)
    views = ((20, -60), (20, 30), (20, 120), (60, -60))
    fig = plt.figure(figsize=(18, 15))
    axes = []
    for plot_index, (elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 2, plot_index, projection="3d")
        axes.append(ax)
        ax.scatter(
            components[order, indices[0]],
            components[order, indices[1]],
            components[order, indices[2]],
            c=colors[order],
            cmap="plasma",
            vmin=vmin,
            vmax=vmax,
            s=point_size,
            alpha=0.75,
            linewidths=0,
            depthshade=False,
            rasterized=True,
        )
        ax.view_init(elev=elev, azim=azim)
        style_3d_axis(ax, indices, f"elev={elev}, azim={azim}")
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(
            norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap="plasma"
        ),
        ax=axes,
        shrink=0.65,
        pad=0.04,
    )
    colorbar.set_label(f"Predicted stage (clipped {vmin:g}-{vmax:g})")
    fig.suptitle(title, fontsize=17)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_3d_category(
    components: np.ndarray,
    values: np.ndarray,
    indices: tuple[int, int, int],
    category: str,
    outpath: Path,
    colors_path: Path,
    point_size: float,
    seed: int,
    max_points: int,
) -> None:
    order = plot_order(len(components), max_points, seed)
    codes, names = category_codes(values)
    cmap = category_cmap(len(names))
    save_category_colors(names, cmap, colors_path)
    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        components[order, indices[0]],
        components[order, indices[1]],
        components[order, indices[2]],
        c=codes[order],
        cmap=cmap,
        vmin=-0.5,
        vmax=max(len(names) - 0.5, 0.5),
        s=point_size,
        alpha=0.7,
        linewidths=0,
        depthshade=False,
        rasterized=True,
    )
    ax.view_init(elev=20, azim=-60)
    style_3d_axis(ax, indices, f"Microcell diffusion map by {category}")
    add_category_legend(ax, names, cmap)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--n-components", type=int, default=15)
    parser.add_argument("--point-size", type=float, default=1.0)
    parser.add_argument("--stage-vmin", type=float, default=4.0)
    parser.add_argument("--stage-vmax", type=float, default=26.0)
    parser.add_argument("--max-microcells", type=int, default=0)
    parser.add_argument(
        "--plot-max-points",
        type=int,
        default=0,
        help="Plot a deterministic subset while still fitting all cells; 0 plots all",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reuse-components",
        action="store_true",
        help="Reuse saved components instead of rebuilding the graph",
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.input_dir / "diffusion_map"
    if args.n_neighbors < 2 or args.n_components < 3:
        parser.error("n-neighbors must be >= 2 and n-components must be >= 3")
    if args.point_size <= 0:
        parser.error("point-size must be positive")
    if args.stage_vmin >= args.stage_vmax:
        parser.error("stage-vmin must be smaller than stage-vmax")
    if args.max_microcells < 0 or args.plot_max_points < 0:
        parser.error("sampling limits must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    embedding_path = args.input_dir / "microcell_embeddings.npy"
    metadata_path = args.input_dir / "microcells.parquet"
    embeddings = np.load(embedding_path, mmap_mode="r")
    metadata = pd.read_parquet(metadata_path).sort_values(
        "microcell_id", kind="stable"
    ).reset_index(drop=True)
    required = {"microcell_id", "mean_stage", "lineage", "dominant_celltype"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"Missing metadata columns: {missing}")
    if embeddings.ndim != 2 or embeddings.shape[0] != len(metadata):
        raise ValueError(
            f"Embedding shape {embeddings.shape} does not align with "
            f"{len(metadata):,} metadata rows"
        )
    stage_all = pd.to_numeric(metadata["mean_stage"], errors="coerce")
    if stage_all.isna().any():
        raise ValueError("mean_stage contains missing or nonnumeric values")

    if args.max_microcells and len(metadata) > args.max_microcells:
        selected = np.sort(
            np.random.default_rng(args.seed).choice(
                len(metadata), args.max_microcells, replace=False
            )
        )
        print(
            f"[sample] computing diffusion map for "
            f"{len(selected):,}/{len(metadata):,} microcells",
            flush=True,
        )
    else:
        selected = slice(None)
        print(
            f"[sample] computing diffusion map for all "
            f"{len(metadata):,} microcells",
            flush=True,
        )
    obs = metadata.iloc[selected].copy()
    stage = pd.to_numeric(obs["mean_stage"]).to_numpy(dtype=np.float64)

    components_path = args.output_dir / "microcell_diffusion_components.npy"
    eigenvalues_path = args.output_dir / "microcell_diffusion_eigenvalues.csv"
    if args.reuse_components:
        if not components_path.exists():
            raise ValueError(f"Cannot reuse missing file: {components_path}")
        components = np.load(components_path)
        expected = (len(obs), args.n_components)
        if components.shape != expected:
            raise ValueError(
                f"Stored components have shape {components.shape}; expected {expected}"
            )
        print(f"Reusing {components_path}", flush=True)
    else:
        matrix = np.array(
            embeddings[selected], dtype=np.float32, order="C", copy=True
        )
        if not np.isfinite(matrix).all():
            raise ValueError("Embedding contains nonfinite values")
        adata = ad.AnnData(X=matrix)
        adata.obsm["X_embedding"] = matrix
        print("Building adaptive Gaussian cosine neighbor graph...", flush=True)
        sc.pp.neighbors(
            adata,
            n_neighbors=min(args.n_neighbors, adata.n_obs - 1),
            use_rep="X_embedding",
            metric="cosine",
            method="gauss",
            random_state=args.seed,
        )
        print(
            f"Computing {args.n_components} diffusion components...",
            flush=True,
        )
        sc.tl.diffmap(
            adata, n_comps=args.n_components, random_state=args.seed
        )
        components = np.asarray(adata.obsm["X_diffmap"], dtype=np.float32)
        eigenvalues = np.asarray(adata.uns["diffmap_evals"], dtype=np.float64)
        expected = (len(obs), args.n_components)
        if components.shape != expected:
            raise RuntimeError(
                f"Diffmap returned shape {components.shape}; expected {expected}"
            )
        if not np.isfinite(components).all():
            raise RuntimeError("Diffmap returned nonfinite components")
        np.save(components_path, components)
        pd.DataFrame(
            {
                "component": [f"DC{i + 1}" for i in range(len(eigenvalues))],
                "eigenvalue": eigenvalues,
            }
        ).to_csv(eigenvalues_path, index=False)
        del adata, matrix
        print(f"Saved components to {components_path}", flush=True)

    correlations = stage_correlations(components, stage)
    correlations.to_csv(
        args.output_dir / "microcell_diffusion_stage_correlations.csv",
        index=False,
    )
    coordinate_columns = [
        column
        for column in (
            "microcell_id",
            "n_cells",
            "mean_stage",
            "stage_std",
            "lineage",
            "dominant_celltype",
            "celltype_purity",
        )
        if column in obs.columns
    ]
    coordinates = obs[coordinate_columns].copy()
    for index in range(components.shape[1]):
        coordinates[f"DC{index + 1}"] = components[:, index]
    coordinates.to_parquet(
        args.output_dir / "microcell_diffusion_coordinates.parquet",
        index=False,
    )

    print("Rendering 2D and 3D diffusion-map plots...", flush=True)
    plot_stage_panels(
        components,
        stage,
        STANDARD_PAIRS,
        args.output_dir / "microcell_diffmap_stage_dc1_dc2_dc3.png",
        args.point_size,
        args.stage_vmin,
        args.stage_vmax,
        args.seed,
        args.plot_max_points,
        "Microcell diffusion map by predicted stage",
    )
    for category, column in (
        ("lineage", "lineage"),
        ("dominant cell type", "dominant_celltype"),
    ):
        stem = column
        plot_category_panels(
            components,
            obs[column].to_numpy(),
            category,
            args.output_dir / f"microcell_diffmap_{stem}_dc1_dc2_dc3.png",
            args.output_dir / f"microcell_diffmap_{stem}_colors.csv",
            args.point_size,
            args.seed,
            args.plot_max_points,
        )

    top_two = tuple(
        int(value)
        for value in correlations["component_index"].head(2).to_numpy()
    )
    plot_stage_panels(
        components,
        stage,
        (top_two,),
        args.output_dir / "microcell_diffmap_stage_top2_components.png",
        args.point_size,
        args.stage_vmin,
        args.stage_vmax,
        args.seed,
        args.plot_max_points,
        "Microcell diffusion map - top two stage-correlated components",
    )
    standard_3d = (0, 1, 2)
    plot_3d_stage(
        components,
        stage,
        standard_3d,
        args.output_dir / "microcell_diffmap3d_stage_dc1_dc2_dc3.png",
        args.point_size,
        args.stage_vmin,
        args.stage_vmax,
        args.seed,
        args.plot_max_points,
        "Microcell diffusion map - DC1/DC2/DC3 by predicted stage",
    )
    for category, column in (
        ("lineage", "lineage"),
        ("dominant cell type", "dominant_celltype"),
    ):
        stem = column
        plot_3d_category(
            components,
            obs[column].to_numpy(),
            standard_3d,
            category,
            args.output_dir / f"microcell_diffmap3d_{stem}_dc1_dc2_dc3.png",
            args.output_dir / f"microcell_diffmap_{stem}_colors.csv",
            args.point_size,
            args.seed,
            args.plot_max_points,
        )
    top_three = tuple(
        int(value)
        for value in correlations["component_index"].head(3).to_numpy()
    )
    plot_3d_stage(
        components,
        stage,
        top_three,
        args.output_dir / "microcell_diffmap3d_stage_top3_components.png",
        args.point_size,
        args.stage_vmin,
        args.stage_vmax,
        args.seed,
        args.plot_max_points,
        "Microcell diffusion map - top three stage-correlated components",
    )

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update(
        {
            "n_microcells": len(obs),
            "embedding_dim": embeddings.shape[1],
            "neighbor_metric": "cosine",
            "neighbor_method": "gauss",
            "stage_column": "mean_stage",
        }
    )
    with (args.output_dir / "microcell_diffusion_config.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(config, handle, indent=2)

    print(correlations.head(10).to_string(index=False))
    print(f"Saved diffusion-map outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
