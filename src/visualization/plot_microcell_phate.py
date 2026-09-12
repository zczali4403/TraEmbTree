#!/usr/bin/env python3
"""Compute and visualize PHATE for TraEmbTree microcells.

All microcells participate directly in PHATE.  The existing ``mean_stage`` is
the mean predicted stage used during microcell construction; for an independent
view, this script also aggregates ``true_stage`` from the underlying cells.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


def log(message: str) -> None:
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run cosine PHATE on all TraEmbTree microcells.",
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--predicted-stage",
        type=Path,
        default=Path("/mnt/input/sc_cz/Concord/eval/2026_09_03/predicted_stage.csv"),
        help="Cell-level table containing contiguous idx and true_stage.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to INPUT_DIR/phate.",
    )
    parser.add_argument("--knn", type=int, default=30)
    parser.add_argument("--decay", type=float, default=40.0)
    parser.add_argument("--n-landmark", type=int, default=10_000)
    parser.add_argument("--t", default="auto", help="PHATE diffusion time or 'auto'.")
    parser.add_argument("--n-jobs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metadata-chunk-size", type=int, default=1_000_000)
    parser.add_argument("--stage-vmin", type=float, default=4.0)
    parser.add_argument("--stage-vmax", type=float, default=26.0)
    parser.add_argument("--cmap", default="plasma")
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.65)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--skip-celltype-plot",
        action="store_true",
        help="Skip the 193-category dominant-celltype plot.",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Require and reuse existing PHATE coordinates without fitting.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute PHATE even if a compatible coordinate checkpoint exists.",
    )
    return parser.parse_args()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def save_npy_atomic(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.writing-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, values)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_inputs(root: Path) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    embedding_path = root / "microcell_embeddings.npy"
    metadata_path = root / "microcells.parquet"
    assignment_path = root / "cell_to_microcell.npy"
    for path in (embedding_path, metadata_path, assignment_path):
        if not path.exists():
            raise FileNotFoundError(path)

    embedding = np.load(embedding_path, mmap_mode="r")
    metadata = (
        pd.read_parquet(metadata_path).sort_values("microcell_id").reset_index(drop=True)
    )
    assignment = np.load(assignment_path, mmap_mode="r")
    if embedding.ndim != 2:
        raise ValueError(f"microcell embeddings must be 2D, got {embedding.shape}")
    if len(metadata) != len(embedding):
        raise ValueError(f"metadata rows {len(metadata):,} != embeddings {len(embedding):,}")
    expected_ids = np.arange(len(metadata), dtype=np.int64)
    if not np.array_equal(metadata.microcell_id.to_numpy(), expected_ids):
        raise ValueError("microcell_id must be contiguous and aligned with embedding rows")
    if assignment.ndim != 1 or len(assignment) != int(metadata.n_cells.sum()):
        raise ValueError("cell_to_microcell length does not match sum(microcells.n_cells)")
    if int(assignment.min()) != 0 or int(assignment.max()) != len(metadata) - 1:
        raise ValueError("cell_to_microcell contains missing or out-of-range microcell IDs")
    if not np.isfinite(embedding).all():
        raise ValueError("microcell embeddings contain non-finite values")
    return embedding, metadata, assignment


def aggregate_true_stage(
    stage_csv: Path,
    assignment: np.ndarray,
    expected_sizes: np.ndarray,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not stage_csv.exists():
        raise FileNotFoundError(stage_csv)
    n_microcells = len(expected_sizes)
    stage_sum = np.zeros(n_microcells, dtype=np.float64)
    stage_count = np.zeros(n_microcells, dtype=np.int64)
    offset = 0
    log("Aggregating cell-level true_stage into microcells")
    for frame in pd.read_csv(
        stage_csv, usecols=["idx", "true_stage"], chunksize=chunk_size
    ):
        size = len(frame)
        end = offset + size
        if end > len(assignment):
            raise ValueError("predicted_stage.csv has more rows than cell_to_microcell")
        indices = frame.idx.to_numpy(dtype=np.int64, copy=False)
        if size and (indices[0] != offset or indices[-1] != end - 1):
            raise ValueError(f"predicted_stage.csv idx is not contiguous at row {offset:,}")
        groups = np.asarray(assignment[offset:end], dtype=np.int64)
        stages = pd.to_numeric(frame.true_stage, errors="coerce").to_numpy(dtype=np.float64)
        valid = np.isfinite(stages)
        stage_sum += np.bincount(
            groups[valid], weights=stages[valid], minlength=n_microcells
        )
        stage_count += np.bincount(groups[valid], minlength=n_microcells)
        offset = end
        log(f"Aggregated true stage for {offset:,}/{len(assignment):,} cells")
    if offset != len(assignment):
        raise ValueError(
            f"predicted_stage.csv has {offset:,} rows; expected {len(assignment):,}"
        )
    if not np.array_equal(stage_count, expected_sizes):
        bad = int(np.count_nonzero(stage_count != expected_sizes))
        raise ValueError(
            f"true_stage is missing or misaligned for {bad:,} microcells; "
            "refusing to produce a partially aggregated annotation"
        )
    return stage_sum / stage_count, stage_count


def fit_config(args: argparse.Namespace, embedding: np.ndarray, root: Path) -> dict:
    t_value: str | int
    if str(args.t).lower() == "auto":
        t_value = "auto"
    else:
        t_value = int(args.t)
        if t_value < 1:
            raise ValueError("--t must be 'auto' or a positive integer")
    return {
        "input_dir": str(root),
        "embedding_shape": list(embedding.shape),
        "knn": args.knn,
        "decay": args.decay,
        "n_landmark": min(args.n_landmark, len(embedding)),
        "t": t_value,
        "knn_dist": "cosine",
        "n_pca": None,
        "n_components": 2,
        "seed": args.seed,
    }


def get_phate_coordinates(
    embedding: np.ndarray,
    output_dir: Path,
    config: dict,
    args: argparse.Namespace,
) -> np.ndarray:
    coordinate_path = output_dir / "microcell_phate.npy"
    config_path = output_dir / "phate_fit_config.json"
    if coordinate_path.exists() and config_path.exists() and not args.overwrite:
        old_config = json.loads(config_path.read_text())
        if old_config != config:
            raise ValueError(
                "Existing PHATE checkpoint has different settings. Use another "
                "--output-dir or pass --overwrite."
            )
        coordinates = np.load(coordinate_path)
        if coordinates.shape != (len(embedding), 2):
            raise ValueError(f"saved coordinates have unexpected shape {coordinates.shape}")
        log(f"Reusing PHATE coordinate checkpoint: {coordinate_path}")
        return coordinates
    if args.plots_only:
        raise FileNotFoundError(
            "--plots-only requires compatible microcell_phate.npy and phate_fit_config.json"
        )

    try:
        import phate
    except ImportError as error:
        raise RuntimeError("PHATE is not installed: pip install phate") from error
    operator = phate.PHATE(
        n_components=2,
        knn=config["knn"],
        decay=config["decay"],
        n_landmark=config["n_landmark"],
        t=config["t"],
        n_pca=None,
        knn_dist="cosine",
        mds_dist="euclidean",
        n_jobs=args.n_jobs,
        random_state=args.seed,
        verbose=1,
    )
    log(
        f"Computing cosine PHATE for all {len(embedding):,} microcells; "
        f"landmarks={config['n_landmark']:,}"
    )
    coordinates = np.asarray(
        operator.fit_transform(np.asarray(embedding, dtype=np.float32)),
        dtype=np.float32,
    )
    if coordinates.shape != (len(embedding), 2) or not np.isfinite(coordinates).all():
        raise RuntimeError(f"PHATE returned invalid coordinates: {coordinates.shape}")
    save_npy_atomic(coordinate_path, coordinates)
    atomic_json(config_path, config)
    log(f"Saved PHATE coordinate checkpoint: {coordinate_path}")
    return coordinates


def discrete_palette(n: int) -> list[str]:
    names = ("tab20", "tab20b", "tab20c", "Set1", "Set2", "Set3", "Dark2", "Paired")
    result: list[str] = []
    seen: set[str] = set()
    for name in names:
        cmap = plt.get_cmap(name)
        for value in cmap.colors:
            color = matplotlib.colors.to_hex(value).upper()
            if color not in seen and color not in {"#FFFFFF", "#000000", "#B0B0B0"}:
                seen.add(color)
                result.append(color)
                if len(result) == n:
                    return result
    if len(result) < n:
        hues = np.linspace(0, 1, n - len(result), endpoint=False)
        result.extend(matplotlib.colors.to_hex(plt.cm.hsv(value)) for value in hues)
    return result[:n]


def existing_palette(root: Path, column: str, categories: list[str]) -> list[str]:
    path = root / f"microcell_umap_{column}_colors.csv"
    if path.exists():
        table = pd.read_csv(path, keep_default_na=False)
        if {column, "color"}.issubset(table.columns):
            mapping = table.drop_duplicates(column).set_index(column).color
            if all(category in mapping.index for category in categories):
                return mapping.reindex(categories).astype(str).tolist()
    return discrete_palette(len(categories))


def save_continuous_plot(
    coordinates: np.ndarray,
    values: np.ndarray,
    output_stem: Path,
    title: str,
    color_label: str,
    args: argparse.Namespace,
    order: np.ndarray,
) -> None:
    valid = np.isfinite(values) & np.isfinite(coordinates).all(axis=1)
    draw = order[valid[order]]
    figure, axis = plt.subplots(figsize=(11, 9), constrained_layout=True)
    points = axis.scatter(
        coordinates[draw, 0],
        coordinates[draw, 1],
        c=np.asarray(values)[draw],
        s=args.point_size,
        alpha=args.alpha,
        cmap=args.cmap,
        vmin=args.stage_vmin,
        vmax=args.stage_vmax,
        linewidths=0,
        rasterized=True,
    )
    figure.colorbar(points, ax=axis, label=color_label)
    axis.set(title=title, xlabel="PHATE1", ylabel="PHATE2")
    figure.savefig(output_stem.with_suffix(".png"), dpi=args.dpi, facecolor="white")
    figure.savefig(output_stem.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)
    log(f"Saved {output_stem.with_suffix('.png')}")


def save_categorical_plot(
    coordinates: np.ndarray,
    labels: pd.Series,
    output_stem: Path,
    title: str,
    palette: list[str],
    args: argparse.Namespace,
    order: np.ndarray,
    figsize: tuple[int, int],
    legend_fontsize: float,
) -> None:
    string_labels = labels.fillna("Unknown").astype(str).to_numpy()
    categories = sorted(np.unique(string_labels))
    if len(palette) != len(categories):
        raise ValueError("palette length does not match categorical labels")
    color_map = dict(zip(categories, palette))
    colors = np.asarray([color_map[value] for value in string_labels], dtype=object)
    figure, axis = plt.subplots(figsize=figsize, constrained_layout=True)
    axis.scatter(
        coordinates[order, 0],
        coordinates[order, 1],
        c=colors[order],
        s=args.point_size,
        alpha=args.alpha,
        linewidths=0,
        rasterized=True,
    )
    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="", markersize=5,
            markerfacecolor=color_map[value], markeredgecolor="none", label=value,
        )
        for value in categories
    ]
    axis.legend(
        handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left",
        frameon=False, fontsize=legend_fontsize,
    )
    axis.set(title=title, xlabel="PHATE1", ylabel="PHATE2")
    figure.savefig(output_stem.with_suffix(".png"), dpi=args.dpi, facecolor="white", bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(figure)
    log(f"Saved {output_stem.with_suffix('.png')}")


def main() -> None:
    args = parse_args()
    root = args.input_dir.resolve()
    output_dir = (args.output_dir or root / "phate").resolve()
    if args.knn < 2 or args.n_landmark < 2 or args.n_jobs < 1:
        raise ValueError("knn and n-landmark must be >=2; n-jobs must be >=1")
    if args.stage_vmax <= args.stage_vmin:
        raise ValueError("stage-vmax must be greater than stage-vmin")
    output_dir.mkdir(parents=True, exist_ok=True)

    embedding, metadata, assignment = validate_inputs(root)
    log(f"Validated {len(embedding):,} microcells x {embedding.shape[1]} dimensions")
    true_stage, true_stage_count = aggregate_true_stage(
        args.predicted_stage,
        assignment,
        metadata.n_cells.to_numpy(dtype=np.int64),
        args.metadata_chunk_size,
    )
    config = fit_config(args, embedding, root)
    coordinates = get_phate_coordinates(embedding, output_dir, config, args)

    result = metadata.copy()
    result["mean_predicted_stage"] = pd.to_numeric(result.mean_stage, errors="raise")
    result["mean_true_stage"] = true_stage
    result["true_stage_cell_count"] = true_stage_count
    result["PHATE1"] = coordinates[:, 0]
    result["PHATE2"] = coordinates[:, 1]
    result.to_parquet(output_dir / "microcell_phate_coordinates.parquet", index=False)

    order = np.random.default_rng(args.seed).permutation(len(metadata))
    save_continuous_plot(
        coordinates,
        result.mean_true_stage.to_numpy(),
        output_dir / "microcell_phate_by_mean_true_stage",
        "Microcells — mean true stage",
        "Mean true stage",
        args,
        order,
    )
    save_continuous_plot(
        coordinates,
        result.mean_predicted_stage.to_numpy(),
        output_dir / "microcell_phate_by_mean_predicted_stage",
        "Microcells — mean predicted stage",
        "Mean predicted stage",
        args,
        order,
    )

    lineage_categories = sorted(metadata.lineage.fillna("Unknown").astype(str).unique())
    save_categorical_plot(
        coordinates,
        metadata.lineage,
        output_dir / "microcell_phate_by_lineage",
        "Microcells — lineage",
        existing_palette(root, "lineage", lineage_categories),
        args,
        order,
        figsize=(14, 10),
        legend_fontsize=8,
    )
    if not args.skip_celltype_plot:
        celltype_categories = sorted(
            metadata.dominant_celltype.fillna("Unknown").astype(str).unique()
        )
        save_categorical_plot(
            coordinates,
            metadata.dominant_celltype,
            output_dir / "microcell_phate_by_dominant_celltype",
            "Microcells — dominant cell type",
            existing_palette(root, "celltype", celltype_categories),
            args,
            order,
            figsize=(18, 12),
            legend_fontsize=5,
        )

    run_config = {
        **config,
        "predicted_stage": str(args.predicted_stage.resolve()),
        "output_dir": str(output_dir),
        "n_jobs": args.n_jobs,
        "stage_color_range": [args.stage_vmin, args.stage_vmax],
        "cmap": args.cmap,
        "point_size": args.point_size,
        "alpha": args.alpha,
        "true_stage_source": "predicted_stage.csv:true_stage aggregated by cell_to_microcell.npy",
        "predicted_stage_source": "microcells.parquet:mean_stage",
    }
    atomic_json(output_dir / "run_config.json", run_config)
    atomic_json(output_dir / "complete.json", {"complete": True})
    log(f"Done: {output_dir}")


if __name__ == "__main__":
    main()
