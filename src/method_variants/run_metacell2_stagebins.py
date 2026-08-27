#!/usr/bin/env python3
"""Run MetaCell2 independently in every lineage x fixed-width stage stratum."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np


DEFAULT_INPUT = "/mnt/input/sc_cz/Concord/data/all_final_corrected_0630.h5ad"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage-bin-width", type=float, default=2.0)
    parser.add_argument(
        "--stage-origin",
        type=float,
        default=0.0,
        help="Bin boundaries are origin + k * width (default: 0, 2, 4, ...).",
    )
    parser.add_argument(
        "--lineage",
        action="append",
        default=None,
        help="Run only this lineage; repeat the option for multiple lineages.",
    )
    parser.add_argument("--min-stratum-cells", type=int, default=12)
    parser.add_argument("--target-cells", type=int, default=48)
    parser.add_argument("--min-cells", type=int, default=12)
    parser.add_argument("--target-umis", type=int, default=160_000)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--random-seed", type=int, default=123456)
    parser.add_argument("--read-block-rows", type=int, default=50_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values],
        dtype=object,
    )


def read_categorical(group: h5py.Group) -> tuple[np.ndarray, np.ndarray]:
    return decode(group["categories"][:]), np.asarray(group["codes"][:], dtype=np.int32)


def safe_name(value: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def number_text(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def enumerate_strata(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.stage_bin_width <= 0:
        raise ValueError("--stage-bin-width must be positive")
    with h5py.File(args.input, "r") as handle:
        lineage_names, lineage_codes = read_categorical(handle["obs/lineage"])
        stage_names, stage_codes = read_categorical(handle["obs/stage"])

    stages_by_category = np.asarray([float(value) for value in stage_names])
    stages = np.full(stage_codes.size, np.nan, dtype=np.float64)
    valid_stage = stage_codes >= 0
    stages[valid_stage] = stages_by_category[stage_codes[valid_stage]]
    bin_index = np.floor((stages - args.stage_origin) / args.stage_bin_width)

    requested = set(args.lineage) if args.lineage else None
    unknown = requested - set(lineage_names) if requested else set()
    if unknown:
        raise ValueError(f"unknown lineage(s): {', '.join(sorted(unknown))}")

    strata: list[dict[str, object]] = []
    for lineage_code, lineage in enumerate(lineage_names):
        if requested is not None and lineage not in requested:
            continue
        mask = (lineage_codes == lineage_code) & np.isfinite(stages)
        indices, counts = np.unique(bin_index[mask].astype(np.int64), return_counts=True)
        for index, count in zip(indices, counts):
            lower = args.stage_origin + int(index) * args.stage_bin_width
            upper = lower + args.stage_bin_width
            strata.append(
                {
                    "lineage": str(lineage),
                    "stage_min": float(lower),
                    "stage_max": float(upper),
                    "cells": int(count),
                    "eligible": int(count) >= args.min_stratum_cells,
                }
            )
    return strata


def expected_run_dir(output_dir: Path, item: dict[str, object]) -> Path:
    lineage = safe_name(str(item["lineage"]))
    lower = str(float(item["stage_min"]))
    upper = str(float(item["stage_max"]))
    return output_dir / f"{lineage}_stage_{lower}_{upper}"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    strata = enumerate_strata(args)
    plan_path = output_dir / "strata_plan.json"
    plan_path.write_text(json.dumps(strata, indent=2), encoding="utf-8")

    eligible = [item for item in strata if bool(item["eligible"])]
    excluded = [item for item in strata if not bool(item["eligible"])]
    print(
        f"planned {len(eligible)} eligible strata; excluded {len(excluded)} strata "
        f"with <{args.min_stratum_cells} cells",
        flush=True,
    )

    worker = Path(__file__).with_name("run_metacell2.py")
    failures: list[dict[str, object]] = []
    completed = 0
    skipped = 0
    for position, item in enumerate(eligible, start=1):
        run_dir = expected_run_dir(output_dir, item)
        summary_path = run_dir / "summary.json"
        label = (
            f"{item['lineage']} [{number_text(float(item['stage_min']))},"
            f"{number_text(float(item['stage_max']))})"
        )
        if summary_path.exists() and not args.overwrite:
            print(f"[{position}/{len(eligible)}] skip completed {label}", flush=True)
            skipped += 1
            continue

        command = [
            sys.executable,
            str(worker),
            "--input",
            args.input,
            "--output-dir",
            str(output_dir),
            "--lineage",
            str(item["lineage"]),
            "--stage-min",
            number_text(float(item["stage_min"])),
            "--stage-max",
            number_text(float(item["stage_max"])),
            "--target-cells",
            str(args.target_cells),
            "--min-cells",
            str(args.min_cells),
            "--target-umis",
            str(args.target_umis),
            "--threads",
            str(args.threads),
            "--random-seed",
            str(args.random_seed),
            "--read-block-rows",
            str(args.read_block_rows),
        ]
        if args.overwrite:
            command.append("--overwrite")
        print(f"[{position}/{len(eligible)}] run {label} ({item['cells']:,} cells)", flush=True)
        if args.dry_run:
            print(" ".join(command), flush=True)
            continue
        result = subprocess.run(command, check=False)
        if result.returncode == 0:
            completed += 1
            continue
        failure = dict(item)
        failure["returncode"] = result.returncode
        failures.append(failure)
        (output_dir / "failed_strata.json").write_text(
            json.dumps(failures, indent=2), encoding="utf-8"
        )
        if not args.continue_on_error:
            raise SystemExit(result.returncode)

    batch_summary = {
        "stage_bin_width": args.stage_bin_width,
        "stage_origin": args.stage_origin,
        "eligible_strata": len(eligible),
        "excluded_sparse_strata": len(excluded),
        "completed_this_run": completed,
        "skipped_completed": skipped,
        "failed": len(failures),
    }
    (output_dir / "batch_summary.json").write_text(
        json.dumps(batch_summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(batch_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
