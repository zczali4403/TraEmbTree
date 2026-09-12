#!/usr/bin/env python3
"""Sweep weights on stage-correlated dimensions before cosine GPU UMAP."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from diagnose_microcell_stage_subspace import (
    configure_gpu,
    run_umap,
    save_lineage,
    save_stage,
)
from plot_microcell_phate import aggregate_true_stage, validate_inputs


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
    parser.add_argument(
        "--dimensions", type=int, nargs="+", default=[49, 91, 16, 87, 0, 52]
    )
    parser.add_argument("--weights", type=float, nargs="+", default=[1, 2, 4, 8])
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


def weight_label(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def main():
    args = parse_args()
    root = args.input_dir.resolve()
    output = (args.output_dir or root / "stage_dimension_weight_sweep").resolve()
    output.mkdir(parents=True, exist_ok=True)
    embedding_disk, metadata, assignment = validate_inputs(root)
    dimensions = np.asarray(args.dimensions, dtype=np.int64)
    if len(np.unique(dimensions)) != len(dimensions):
        raise ValueError("--dimensions contains duplicates")
    if np.any(dimensions < 0) or np.any(dimensions >= embedding_disk.shape[1]):
        raise ValueError("--dimensions contains an out-of-range index")
    if any(not np.isfinite(value) or value <= 0 for value in args.weights):
        raise ValueError("all --weights must be finite and positive")

    true_stage, _ = aggregate_true_stage(
        args.predicted_stage,
        assignment,
        metadata.n_cells.to_numpy(dtype=np.int64),
        args.chunk_size,
    )
    predicted_stage = pd.to_numeric(metadata.mean_stage, errors="raise").to_numpy(float)
    lineage = metadata.lineage.fillna("Unknown").astype(str).to_numpy()
    raw_embedding = np.asarray(embedding_disk, dtype=np.float32)
    order = np.random.default_rng(args.seed).permutation(len(metadata))
    cp = configure_gpu(args.gpu)

    completed = []
    for weight in args.weights:
        label = weight_label(weight)
        prefix = f"top{len(dimensions):02d}_weight_alpha{label}"
        coordinate_path = output / f"{prefix}_umap.npy"
        if coordinate_path.exists() and not args.overwrite:
            coordinates = np.load(coordinate_path)
            if coordinates.shape != (len(metadata), 2):
                raise ValueError(f"invalid saved coordinates: {coordinate_path}")
            print(f"reusing: {coordinate_path}", flush=True)
        else:
            weighted = np.array(raw_embedding, dtype=np.float32, copy=True)
            weighted[:, dimensions] *= weight
            norms = np.linalg.norm(weighted, axis=1)
            if np.any(norms == 0) or not np.isfinite(norms).all():
                raise ValueError(f"invalid embedding norm for alpha={weight:g}")
            weighted /= norms[:, None]
            print(
                f"GPU UMAP alpha={weight:g}; dimensions={dimensions.tolist()}", flush=True
            )
            coordinates = run_umap(
                weighted,
                args.n_neighbors,
                args.min_dist,
                args.spread,
                args.seed,
                cp,
            )
            if coordinates.shape != (len(metadata), 2) or not np.isfinite(coordinates).all():
                raise ValueError(f"invalid UMAP coordinates for alpha={weight:g}")
            np.save(coordinate_path, coordinates)
            del weighted
            cp.get_default_memory_pool().free_all_blocks()

        table = pd.DataFrame(
            {
                "microcell_id": metadata.microcell_id,
                "mean_true_stage": true_stage,
                "mean_predicted_stage": predicted_stage,
                "lineage": lineage,
                "UMAP1": coordinates[:, 0],
                "UMAP2": coordinates[:, 1],
            }
        )
        table.to_parquet(output / f"{prefix}_umap.parquet", index=False)
        title = f"All 100 dimensions; top-{len(dimensions)} weight alpha={weight:g}"
        save_stage(
            coordinates,
            true_stage,
            output / f"{prefix}_true_stage.png",
            title + " — mean true stage",
            "Mean true stage",
            args,
            order,
        )
        save_stage(
            coordinates,
            predicted_stage,
            output / f"{prefix}_predicted_stage.png",
            title + " — mean predicted stage",
            "Mean predicted stage",
            args,
            order,
        )
        save_lineage(
            coordinates,
            lineage,
            output / f"{prefix}_lineage.png",
            title + " — lineage",
            args,
            order,
        )
        completed.append({"alpha": weight, "coordinate_file": coordinate_path.name})
        print(f"completed alpha={weight:g}", flush=True)

    config = {
        "input_dir": str(root),
        "predicted_stage": str(args.predicted_stage.resolve()),
        "output_dir": str(output),
        "dimensions_zero_based": dimensions.tolist(),
        "weights": args.weights,
        "operation": "multiply selected raw-z dimensions, then L2 normalize",
        "metric": "cosine via L2 normalization and CAGRA inner product",
        "n_neighbors": args.n_neighbors,
        "min_dist": args.min_dist,
        "spread": args.spread,
        "gpu": args.gpu,
        "seed": args.seed,
        "stage_color_range": [args.stage_vmin, args.stage_vmax],
        "completed": completed,
    }
    (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "complete.json").write_text('{"complete": true}\n')
    print(f"done: {output}", flush=True)


if __name__ == "__main__":
    main()
