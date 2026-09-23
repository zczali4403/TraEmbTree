#!/usr/bin/env python3
"""Recover original-cell membership for contracted trajectory-tree nodes.

The mapping is reconstructed as

    cell row -> microcell -> source temporal node -> contracted node.

Rows are streamed to Parquet in chunks so the complete 15-million-cell run does
not need to be materialized as one pandas DataFrame.  Cell IDs are included when
``cell_ids.npy`` is supplied or can be found through the saved run configs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--contracted-dir", type=Path, required=True,
        help="Contracted-tree directory containing source_to_contracted_nodes.parquet.")
    parser.add_argument(
        "--node-dir", type=Path,
        help="Trajectory-node directory; inferred from contracted run_config.json.")
    parser.add_argument(
        "--microcell-dir", type=Path,
        help="Microcell directory; inferred from trajectory-node run_config.json.")
    parser.add_argument(
        "--cell-ids", type=Path,
        help="Row-aligned cell_ids.npy. If unavailable, output contains cell_index only.")
    parser.add_argument(
        "--node-id", type=int, nargs="+", default=None,
        help="Contracted node IDs to extract. Omit to export every contracted node.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output Parquet file.")
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--compression", default="zstd",
                        choices=("zstd", "snappy", "gzip", "none"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def read_config(directory: Path) -> dict:
    path = directory / "run_config.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def config_path(config: dict, key: str) -> Path | None:
    value = config.get(key)
    if value in (None, ""):
        return None
    return Path(value).expanduser().resolve()


def require_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def load_row_aligned_array(path: Path, description: str):
    """Memory-map ordinary NPY arrays and reject unsafe object arrays."""
    try:
        values = np.load(path, mmap_mode="r", allow_pickle=False)
    except ValueError as error:
        raise ValueError(
            f"{description} must be a non-object .npy array so it can be read safely: "
            f"{path}") from error
    if values.ndim != 1:
        raise ValueError(f"{description} must be one-dimensional: {path}")
    return values


def resolve_inputs(args):
    contracted_dir = args.contracted_dir.expanduser().resolve()
    contracted_config = read_config(contracted_dir)

    node_dir = (args.node_dir.expanduser().resolve() if args.node_dir else
                config_path(contracted_config, "node_dir"))
    if node_dir is None or not node_dir.is_dir():
        raise FileNotFoundError(
            "Could not locate the trajectory-node directory; pass --node-dir")

    node_config = read_config(node_dir)
    microcell_dir = (
        args.microcell_dir.expanduser().resolve() if args.microcell_dir else
        config_path(node_config, "input_dir"))
    if microcell_dir is None or not microcell_dir.is_dir():
        raise FileNotFoundError(
            "Could not locate the microcell directory; pass --microcell-dir")

    cell_ids_path = args.cell_ids.expanduser().resolve() if args.cell_ids else None
    if cell_ids_path is None:
        microcell_config = read_config(microcell_dir)
        csr_dir = config_path(microcell_config, "csr_dir")
        candidate = csr_dir / "cell_ids.npy" if csr_dir is not None else None
        if candidate is not None and candidate.is_file():
            cell_ids_path = candidate
    if cell_ids_path is not None:
        require_file(cell_ids_path, "cell ID array")

    return contracted_dir, node_dir, microcell_dir, cell_ids_path


def build_mapping(contracted_dir: Path, node_dir: Path):
    source_path = require_file(
        contracted_dir / "source_to_contracted_nodes.parquet",
        "source-to-contracted mapping")
    assignment_path = require_file(
        node_dir / "microcell_node_assignments.parquet",
        "microcell-to-source-node mapping")

    source = pd.read_parquet(
        source_path, columns=["source_node_id", "contracted_node_id"])
    if source.empty or source.source_node_id.duplicated().any():
        raise ValueError("source_node_id must be unique and nonempty")
    source_ids = source.source_node_id.to_numpy(dtype=np.int64)
    contracted_ids = source.contracted_node_id.to_numpy(dtype=np.int64)
    if np.any(source_ids < 0) or np.any(contracted_ids < 0):
        raise ValueError("Source and contracted node IDs must be nonnegative")
    source_to_contracted = np.full(source_ids.max() + 1, -1, dtype=np.int64)
    source_to_contracted[source_ids] = contracted_ids

    assignment = pd.read_parquet(
        assignment_path, columns=["microcell_id", "node_id"])
    if assignment.empty or assignment.microcell_id.duplicated().any():
        raise ValueError("microcell_id must be unique and nonempty")
    microcell_ids = assignment.microcell_id.to_numpy(dtype=np.int64)
    node_ids = assignment.node_id.to_numpy(dtype=np.int64)
    if np.any(microcell_ids < 0):
        raise ValueError("microcell_id must be nonnegative")
    microcell_to_source = np.full(microcell_ids.max() + 1, -1, dtype=np.int64)
    microcell_to_source[microcell_ids] = node_ids

    microcell_to_contracted = np.full(len(microcell_to_source), -1, dtype=np.int64)
    retained = ((microcell_to_source >= 0)
                & (microcell_to_source < len(source_to_contracted)))
    microcell_to_contracted[retained] = source_to_contracted[
        microcell_to_source[retained]]
    return microcell_to_source, microcell_to_contracted, np.unique(contracted_ids)


def output_schema(include_cell_id: bool) -> pa.Schema:
    fields = [pa.field("cell_index", pa.int64())]
    if include_cell_id:
        fields.append(pa.field("cell_id", pa.string()))
    fields.extend([
        pa.field("microcell_id", pa.int64()),
        pa.field("source_node_id", pa.int64()),
        pa.field("contracted_node_id", pa.int64()),
    ])
    return pa.schema(fields)


def main(argv=None):
    args = parse_args(argv)
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be positive")

    contracted_dir, node_dir, microcell_dir, cell_ids_path = resolve_inputs(args)
    cell_to_microcell_path = require_file(
        microcell_dir / "cell_to_microcell.npy", "cell-to-microcell mapping")
    cell_to_microcell = load_row_aligned_array(
        cell_to_microcell_path, "cell-to-microcell mapping")
    cell_ids = (load_row_aligned_array(cell_ids_path, "cell ID array")
                if cell_ids_path is not None else None)
    if cell_ids is not None and len(cell_ids) != len(cell_to_microcell):
        raise ValueError(
            f"cell_ids has {len(cell_ids):,} rows but cell_to_microcell has "
            f"{len(cell_to_microcell):,}")
    if cell_ids is None:
        print("Warning: cell_ids.npy was not found; exporting cell_index only.",
              file=sys.stderr, flush=True)

    (microcell_to_source, microcell_to_contracted,
     available_nodes) = build_mapping(contracted_dir, node_dir)
    if len(cell_to_microcell):
        min_microcell = int(np.min(cell_to_microcell))
        max_microcell = int(np.max(cell_to_microcell))
        if min_microcell < 0 or max_microcell >= len(microcell_to_source):
            raise ValueError(
                "cell_to_microcell.npy contains IDs absent from "
                "microcell_node_assignments.parquet")

    selected = None
    if args.node_id is not None:
        requested = np.unique(np.asarray(args.node_id, dtype=np.int64))
        missing = requested[~np.isin(requested, available_nodes)]
        if len(missing):
            raise ValueError(
                f"Unknown contracted node IDs: {missing.tolist()}; valid range is "
                f"{int(available_nodes.min())}..{int(available_nodes.max())}")
        selected = np.zeros(int(available_nodes.max()) + 1, dtype=bool)
        selected[requested] = True

    output = args.output.expanduser().resolve()
    if output.suffix.lower() != ".parquet":
        raise ValueError("--output must end in .parquet")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; use --overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.building-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary}")

    schema = output_schema(cell_ids is not None)
    compression = None if args.compression == "none" else args.compression
    writer = pq.ParquetWriter(temporary, schema, compression=compression)
    written = 0
    excluded = 0
    try:
        for start in range(0, len(cell_to_microcell), args.chunk_size):
            stop = min(start + args.chunk_size, len(cell_to_microcell))
            microcell = np.asarray(cell_to_microcell[start:stop], dtype=np.int64)
            source_node = microcell_to_source[microcell]
            contracted_node = microcell_to_contracted[microcell]
            keep = contracted_node >= 0
            excluded += int(np.count_nonzero(~keep))
            if selected is not None:
                in_range = (contracted_node >= 0) & (contracted_node < len(selected))
                chosen = np.zeros(len(contracted_node), dtype=bool)
                chosen[in_range] = selected[contracted_node[in_range]]
                keep &= chosen
            local = np.flatnonzero(keep)
            if not len(local):
                continue
            arrays = {
                "cell_index": np.arange(start, stop, dtype=np.int64)[local],
                "microcell_id": microcell[local],
                "source_node_id": source_node[local],
                "contracted_node_id": contracted_node[local],
            }
            if cell_ids is not None:
                arrays["cell_id"] = np.asarray(cell_ids[start:stop])[local].astype(str)
            table = pa.Table.from_pydict(arrays, schema=schema)
            writer.write_table(table)
            written += len(local)
            print(
                f"processed {stop:,}/{len(cell_to_microcell):,}; "
                f"written {written:,}", flush=True)
        writer.close()
        if output.exists():
            output.unlink()
        temporary.rename(output)
    except Exception:
        writer.close()
        print(f"Failed; partial output kept at {temporary}", file=sys.stderr)
        raise

    result = {
        "output": str(output),
        "rows_written": written,
        "input_cells": int(len(cell_to_microcell)),
        "cells_without_contracted_node": excluded,
        "selected_contracted_node_ids": (
            None if args.node_id is None else sorted(set(args.node_id))),
        "cell_id_included": cell_ids is not None,
    }
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
