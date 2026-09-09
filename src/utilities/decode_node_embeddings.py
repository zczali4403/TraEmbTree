#!/usr/bin/env python3
"""Decode TraEmb trajectory-node embeddings into ZINB mean expression.

This utility sends each node embedding directly through the decoder from the
same checkpoint that produced the cell embeddings. It does not run the encoder.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.lib.format import open_memmap


DEFAULT_TRAEMB_DIR = Path("/mnt/input/sc_cz/Concord/TraEmb")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--embedding-path", type=Path, required=True,
                        help="Contracted or uncontracted node_embeddings.npy")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="TraEmb checkpoint that produced the latent embedding space")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--node-table", type=Path, default=None,
                        help="Aligned tree_nodes.parquet or nodes.parquet; inferred beside embedding when omitted")
    parser.add_argument("--traemb-dir", type=Path, default=DEFAULT_TRAEMB_DIR,
                        help="TraEmb source directory containing pl_module.py")
    parser.add_argument("--config", type=Path, required=True,
                        help="TraEmb experiment config matching the checkpoint")
    parser.add_argument("--gene-names", type=Path, default=None,
                        help="Optional .npy, .csv, .tsv, or one-name-per-line gene-name file")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def load_config(path, traemb_dir):
    path = path.resolve()
    sys.path.insert(0, str(traemb_dir))
    spec = importlib.util.spec_from_file_location("traemb_decoder_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "get_config"):
        raise AttributeError(f"Config does not define get_config(): {path}")
    return module.get_config()


def load_checkpoint_state(path):
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format: {path}")
    return state


def decoder_dimensions(state):
    weights = []
    prefix = "model.net.decoder."
    for name, value in state.items():
        if name.startswith(prefix) and name.endswith(".weight") and value.ndim == 2:
            suffix = name[len(prefix):-len(".weight")]
            numeric_parts = tuple(int(part) for part in suffix.split(".") if part.isdigit())
            weights.append((numeric_parts, name, value))
    if not weights:
        raise ValueError("Checkpoint has no model.net.decoder linear weights")
    weights.sort(key=lambda item: item[0])
    return int(weights[0][2].shape[1]), int(weights[-1][2].shape[0])


def build_module(cfg, state, latent_dim, output_dim, traemb_dir):
    if int(cfg.model.latent_dim) != latent_dim:
        raise ValueError(
            f"Config latent_dim={int(cfg.model.latent_dim)}, checkpoint decoder input={latent_dim}")
    if not bool(cfg.model.get("use_decoder", False)):
        raise ValueError("Config has model.use_decoder=False")
    cfg.model.input_dim = output_dim
    cfg.model.num_domains = max(1, int(cfg.model.get("num_domains", 0)))
    cfg.model.num_lineages = max(1, int(cfg.model.get("num_lineages", 0)))
    cfg.n_stages = 1
    cfg.stage_values = [0.0]

    sys.path.insert(0, str(traemb_dir))
    from pl_module import StageLineageModule

    module = StageLineageModule(cfg, vocab={})
    incompatible = module.load_state_dict(state, strict=False)
    decoder_prefix = "model.net.decoder."
    missing = [key for key in incompatible.missing_keys if key.startswith(decoder_prefix)]
    unexpected = [key for key in incompatible.unexpected_keys if key.startswith(decoder_prefix)]
    if missing or unexpected:
        raise ValueError(
            f"Decoder checkpoint mismatch; missing={missing[:10]}, unexpected={unexpected[:10]}")
    return module


def load_node_table(path, embedding_path, n_nodes):
    if path is None:
        candidate = embedding_path.parent / "tree_nodes.parquet"
        path = candidate if candidate.is_file() else None
    if path is None:
        return pd.DataFrame({"node_id": np.arange(n_nodes, dtype=np.int64)}), None

    path = path.resolve()
    table = pd.read_parquet(path)
    if "node_type" in table.columns:
        table = table[table.node_type.eq("temporal_node")]
    if "node_id" not in table.columns:
        raise ValueError("Node table must contain node_id")
    table = table.sort_values("node_id").reset_index(drop=True)
    ids = table.node_id.to_numpy(dtype=np.int64)
    if len(table) != n_nodes:
        raise ValueError(
            f"Node table has {len(table)} real nodes, but embedding has {n_nodes} rows")
    if not np.array_equal(ids, np.arange(n_nodes, dtype=np.int64)):
        raise ValueError("Real node IDs must be contiguous from zero and align with embedding rows")
    preferred = [
        "node_id", "lineage", "dominant_celltype", "celltype_purity",
        "n_cells", "n_metacells", "cell_weighted_mean_stage",
        "min_stage", "max_stage", "n_source_nodes", "source_node_ids",
    ]
    return table[[column for column in preferred if column in table.columns]].copy(), path


def load_gene_names(path, output_dim):
    if path is None:
        return np.asarray([f"g_{index}" for index in range(output_dim)], dtype=str)
    path = path.resolve()
    suffix = path.suffix.lower()
    if suffix == ".npy":
        names = np.load(path, allow_pickle=False).astype(str)
    elif suffix in {".csv", ".tsv"}:
        table = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
        if table.shape[1] == 0:
            raise ValueError(f"Gene-name table contains no columns: {path}")
        names = table.iloc[:, 0].astype(str).to_numpy()
    else:
        names = np.asarray(
            [line.strip() for line in path.read_text().splitlines() if line.strip()], dtype=str)
    names = np.asarray(names).reshape(-1)
    if len(names) != output_dim:
        raise ValueError(
            f"Gene-name count {len(names)} does not match decoder output {output_dim}")
    return names


def main(argv=None):
    args = parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    embedding_path = args.embedding_path.resolve()
    checkpoint_path = args.checkpoint.resolve()
    traemb_dir = args.traemb_dir.resolve()
    config_path = args.config if args.config.is_absolute() else traemb_dir / args.config
    config_path = config_path.resolve()
    output = args.output_dir.resolve()
    for path in (embedding_path, checkpoint_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not (traemb_dir / "pl_module.py").is_file():
        raise FileNotFoundError(traemb_dir / "pl_module.py")
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output}; use --overwrite")
        shutil.rmtree(output)
    temporary = output.with_name(f".{output.name}.building")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    try:
        embeddings = np.load(embedding_path, mmap_mode="r")
        if embeddings.ndim != 2 or not np.issubdtype(embeddings.dtype, np.number):
            raise ValueError("Embedding must be a numeric 2D NumPy array")
        n_nodes, embedding_dim = map(int, embeddings.shape)
        if n_nodes == 0:
            raise ValueError("Embedding contains no nodes")

        state = load_checkpoint_state(checkpoint_path)
        latent_dim, output_dim = decoder_dimensions(state)
        if embedding_dim != latent_dim:
            raise ValueError(
                f"Embedding dimension {embedding_dim} does not match decoder input {latent_dim}")
        cfg = load_config(config_path, traemb_dir)
        module = build_module(cfg, state, latent_dim, output_dim, traemb_dir)
        module.eval()
        module.freeze()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        module.to(device)

        nodes, node_table_path = load_node_table(args.node_table, embedding_path, n_nodes)
        genes = load_gene_names(args.gene_names, output_dim)
        decoded_path = temporary / "node_decoded_expression.npy"
        decoded = open_memmap(
            decoded_path, mode="w+", dtype=np.float32, shape=(n_nodes, output_dim))
        decoder = module.model.net.decoder

        with torch.inference_mode():
            for start in range(0, n_nodes, args.batch_size):
                end = min(start + args.batch_size, n_nodes)
                values = np.asarray(embeddings[start:end], dtype=np.float32)
                if not np.isfinite(values).all():
                    raise ValueError(f"Non-finite embedding values in rows {start}:{end}")
                z = torch.from_numpy(values).to(device, non_blocking=True)
                mu = torch.nn.functional.softplus(decoder(z))
                decoded[start:end] = mu.float().cpu().numpy()
                print(f"decoded {end:,}/{n_nodes:,} nodes", flush=True)
        decoded.flush()
        del decoded

        nodes.to_csv(temporary / "node_decoded_nodes.csv", index=False)
        pd.DataFrame({"gene_index": np.arange(output_dim), "gene": genes}).to_csv(
            temporary / "node_decoded_genes.csv", index=False)
        run_config = {
            "embedding_path": str(embedding_path),
            "node_table": str(node_table_path) if node_table_path else None,
            "checkpoint": str(checkpoint_path),
            "traemb_config": str(config_path),
            "n_nodes": n_nodes,
            "latent_dim": latent_dim,
            "n_genes": output_dim,
            "batch_size": args.batch_size,
            "device": str(device),
            "decoded_quantity": "ZINB mean mu = softplus(decoder(node_embedding))",
            "gene_names_source": str(args.gene_names.resolve()) if args.gene_names else "generated g_0..g_N",
        }
        (temporary / "run_config.json").write_text(
            json.dumps(run_config, indent=2, ensure_ascii=False) + "\n")
        os.replace(temporary, output)
        print(f"saved: {output / 'node_decoded_expression.npy'}")
        print(f"shape: ({n_nodes}, {output_dim})")
    except BaseException:
        print(f"failed; partial output kept at {temporary}", flush=True)
        raise


if __name__ == "__main__":
    main()
