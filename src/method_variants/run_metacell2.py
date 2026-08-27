#!/usr/bin/env python3
"""Run MetaCell2 on a lineage/stage subset of a very large H5AD file.

The input file contains >15 million cells, so loading even its backed AnnData
object can consume substantial memory through ``obs``. This script reads only
the requested observations and CSR rows with h5py, then runs the official
MetaCell2 divide-and-conquer pipeline. Cell-type labels are never accessed.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
import metacells as mc


DEFAULT_INPUT = "/mnt/input/sc_cz/Concord/data/all_final_corrected_0630.h5ad"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MetaCell2 without using cell-type labels."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--lineage",
        required=True,
        help="Exact lineage name, e.g. Liver. Use --list-lineages to inspect names.",
    )
    parser.add_argument("--stage-min", type=float, default=None, help="Inclusive")
    parser.add_argument("--stage-max", type=float, default=None, help="Exclusive")
    parser.add_argument("--target-cells", type=int, default=48)
    parser.add_argument("--min-cells", type=int, default=12)
    parser.add_argument("--target-umis", type=int, default=160_000)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--random-seed", type=int, default=123456)
    parser.add_argument(
        "--read-block-rows",
        type=int,
        default=50_000,
        help="Rows per HDF5 extraction block; lower this if extraction RAM is high.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list-lineages", action="store_true")
    return parser.parse_args()


def text_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values],
        dtype=object,
    )


def categorical_values(group: h5py.Group) -> tuple[np.ndarray, np.ndarray]:
    categories = text_array(group["categories"][:])
    codes = np.asarray(group["codes"][:], dtype=np.int32)
    return categories, codes


def read_string_rows(node: h5py.Dataset | h5py.Group, rows: np.ndarray) -> np.ndarray:
    if isinstance(node, h5py.Group):
        categories = text_array(node["categories"][:])
        codes = np.asarray(node["codes"][rows], dtype=np.int64)
        result = np.full(rows.size, "", dtype=object)
        valid = codes >= 0
        result[valid] = categories[codes[valid]]
        return result
    return text_array(node[rows])


def choose_rows(handle: h5py.File, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    categories, codes = categorical_values(handle["obs/lineage"])
    matches = np.flatnonzero(categories == args.lineage)
    if matches.size != 1:
        available = ", ".join(categories.tolist())
        raise ValueError(f"unknown lineage {args.lineage!r}; available: {available}")
    mask = codes == int(matches[0])

    stage_categories, stage_codes = categorical_values(handle["obs/stage"])
    numeric_categories = np.asarray([float(value) for value in stage_categories])
    stages = np.full(stage_codes.size, np.nan, dtype=np.float32)
    valid = stage_codes >= 0
    stages[valid] = numeric_categories[stage_codes[valid]]
    if args.stage_min is not None:
        mask &= stages >= args.stage_min
    if args.stage_max is not None:
        mask &= stages < args.stage_max
    rows = np.flatnonzero(mask).astype(np.int64)
    return rows, stages[rows]


def extract_csr_rows(
    handle: h5py.File, rows: np.ndarray, block_rows: int
) -> sp.csr_matrix:
    x_group = handle["X"]
    shape = tuple(int(value) for value in x_group.attrs["shape"])
    indptr_ds = x_group["indptr"]
    indices_ds = x_group["indices"]
    data_ds = x_group["data"]
    pieces: list[sp.csr_matrix] = []

    # Scan row blocks, but read expression values only for blocks containing
    # requested cells. Selected rows remain in original input order.
    for row_start in range(0, shape[0], block_rows):
        left = np.searchsorted(rows, row_start, side="left")
        row_stop = min(row_start + block_rows, shape[0])
        right = np.searchsorted(rows, row_stop, side="left")
        if left == right:
            continue
        local_rows = rows[left:right] - row_start
        ptr = np.asarray(indptr_ds[row_start : row_stop + 1], dtype=np.int64)
        value_start, value_stop = int(ptr[0]), int(ptr[-1])
        ptr -= value_start
        block = sp.csr_matrix(
            (
                np.asarray(data_ds[value_start:value_stop]),
                np.asarray(indices_ds[value_start:value_stop], dtype=np.int32),
                ptr,
            ),
            shape=(row_stop - row_start, shape[1]),
        )
        pieces.append(block[local_rows])
        print(f"extracted {right:,}/{rows.size:,} cells", flush=True)

    if not pieces:
        return sp.csr_matrix((0, shape[1]), dtype=np.int32)
    matrix = sp.vstack(pieces, format="csr")
    if np.any(matrix.data < 0) or np.any(matrix.data != np.rint(matrix.data)):
        raise ValueError("X is not a non-negative integer UMI-count matrix")
    # MetaCell2 expects integer-valued UMI counts stored as float32.
    matrix.data = matrix.data.astype(np.float32, copy=False)
    matrix.sort_indices()
    return matrix


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def add_summary_annotations(mdata: AnnData, adata: AnnData, lineage: str) -> None:
    groups = np.asarray(adata.obs["metacell"], dtype=np.int64)
    stages = np.asarray(adata.obs["stage"], dtype=float)
    mdata.obs["lineage"] = lineage
    means = np.full(mdata.n_obs, np.nan)
    minima = np.full(mdata.n_obs, np.nan)
    maxima = np.full(mdata.n_obs, np.nan)
    for group in range(mdata.n_obs):
        values = stages[groups == group]
        if values.size:
            means[group] = np.mean(values)
            minima[group] = np.min(values)
            maxima[group] = np.max(values)
    mdata.obs["mean_stage"] = means
    mdata.obs["min_stage"] = minima
    mdata.obs["max_stage"] = maxima


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    with h5py.File(input_path, "r") as handle:
        lineages, lineage_codes = categorical_values(handle["obs/lineage"])
        if args.list_lineages:
            counts = np.bincount(lineage_codes[lineage_codes >= 0], minlength=len(lineages))
            for lineage, count in zip(lineages, counts):
                print(f"{lineage}\t{count}")
            return

        rows, stages = choose_rows(handle, args)
        if rows.size < args.min_cells:
            raise ValueError(f"only {rows.size} selected cells; need at least {args.min_cells}")
        print(f"selected {rows.size:,} cells", flush=True)
        matrix = extract_csr_rows(handle, rows, args.read_block_rows)
        genes = read_string_rows(handle["var/gene"], np.arange(matrix.shape[1]))
        cell_ids = read_string_rows(handle["obs/cell_id"], rows)

    suffix = safe_name(args.lineage)
    if args.stage_min is not None or args.stage_max is not None:
        suffix += f"_stage_{args.stage_min if args.stage_min is not None else 'min'}"
        suffix += f"_{args.stage_max if args.stage_max is not None else 'max'}"
    run_dir = output_dir / suffix
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{run_dir} is not empty; use --overwrite")
    run_dir.mkdir(parents=True, exist_ok=True)

    adata = AnnData(
        X=matrix,
        obs=pd.DataFrame(
            {"original_row": rows, "lineage": args.lineage, "stage": stages},
            index=pd.Index(cell_ids, name="cell_id"),
        ),
        var=pd.DataFrame(index=pd.Index(genes, name="gene")),
    )
    adata.var_names_make_unique()
    # No dataset-specific lateral-gene blacklist was supplied.
    adata.var["lateral_gene"] = False
    mc.ut.set_processors_count(args.threads)
    print(f"running MetaCell2 {mc.__version__} with {args.threads} threads", flush=True)
    mc.pl.divide_and_conquer_pipeline(
        adata,
        target_metacell_size=args.target_cells,
        min_metacell_size=args.min_cells,
        target_metacell_umis=args.target_umis,
        random_seed=args.random_seed,
    )
    mdata = mc.pl.collect_metacells(adata, random_seed=args.random_seed)
    add_summary_annotations(mdata, adata, args.lineage)

    mdata.write_h5ad(run_dir / "metacells.h5ad", compression="gzip")
    assignments = pd.DataFrame(
        {
            "cell_id": adata.obs_names,
            "original_row": adata.obs["original_row"].to_numpy(),
            "lineage": args.lineage,
            "stage": adata.obs["stage"].to_numpy(),
            "metacell": adata.obs["metacell"].to_numpy(),
        }
    )
    with gzip.open(run_dir / "cell_assignments.tsv.gz", "wt") as stream:
        assignments.to_csv(stream, sep="\t", index=False)

    grouped = np.asarray(adata.obs["metacell"], dtype=int) >= 0
    summary = {
        "input": str(input_path),
        "lineage": args.lineage,
        "stage_min_inclusive": args.stage_min,
        "stage_max_exclusive": args.stage_max,
        "selected_cells": int(adata.n_obs),
        "grouped_cells": int(np.sum(grouped)),
        "outlier_cells": int(np.sum(~grouped)),
        "metacells": int(mdata.n_obs),
        "genes": int(mdata.n_vars),
        "target_cells": args.target_cells,
        "min_cells": args.min_cells,
        "target_umis": args.target_umis,
        "random_seed": args.random_seed,
        "metacells_version": mc.__version__,
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
