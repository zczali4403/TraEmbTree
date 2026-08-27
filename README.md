# Lineage- and Stage-Constrained Landmark Tree Construction

This repository implements a scalable workflow for representing large single-cell embedding datasets with landmark nodes and constructing temporally ordered developmental trees. The method preserves lineage identity and developmental-stage structure while reducing millions of cells to a compact, interpretable representation.

## Method

The workflow comprises four stages:

1. **Lineage-stage stratification.** Cells are partitioned into non-overlapping strata defined by lineage and fixed-width developmental-stage intervals.
2. **Microcell construction.** Within each stratum, two-pass balanced partitioning of cosine k-nearest-neighbor graphs produces compact microcells. Each microcell is represented by the mean embedding of its member cells.
3. **Adaptive landmark allocation.** Landmark quotas are allocated across eligible strata according to equal-microcell-weight marginal improvement in cosine-space coverage. Allocation terminates when the best available marginal gain falls below a specified threshold.
4. **Stage-monotone tree construction.** Landmark nodes are ordered by mean developmental stage. Each node is connected to its nearest eligible earlier parent, preferentially within the same lineage.

The adaptive allocation procedure determines the number of landmark nodes from the geometry of the microcell embeddings rather than requiring a fixed node count.

## Reference configuration

| Parameter | Value |
|---|---:|
| Embedding dimension | 100 |
| Stage-bin width | 2 |
| Microcell target size | approximately 500 cells |
| Allocation kNN | 30 |
| Regrouping kNN | 64 |
| Minimum cells per stratum | 50 |
| Minimum marginal gain | 0.001 |
| Random seed | 42 |

Under this configuration, 217 eligible lineage-stage strata produced 1,439 landmark nodes. Fourteen sparsely supported strata comprising 329 cells were excluded from landmark allocation.

## Installation

Python 3.10 is recommended. The exact package versions used for the reference analysis are specified in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The reference environment used a GPU-enabled FAISS build. For CPU-only systems, replace `faiss-gpu` with a compatible `faiss-cpu` release.

## Input data

The primary workflow expects:

- a NumPy embedding matrix with shape `(n_cells, 100)`;
- row-aligned cell identifiers;
- a metadata table containing `cell`, `celltype`, `lineage_final`, `stage`, and `dataset`;
- a CSR or index source compatible with the selection scripts, when required by the input representation.

Input paths are supplied explicitly through command-line arguments. Source data are not distributed with the software.

## Workflow

### 1. Construct stage-constrained microcells

```bash
python src/workflow/select_landmark_nodes_mc2_lineage_stagebin_marginal.py \
  --embeddings /path/to/embeddings.npy \
  --csr-dir /path/to/csr_or_index_source \
  --metadata /path/to/metadata.csv \
  --output-dir /path/to/stagebin_microcells \
  --stage-bin-width 2 \
  --overwrite
```

### 2. Construct an adaptive set of landmark nodes

```bash
python src/workflow/remerge_stagebin_microcells_adaptive.py \
  --input-dir /path/to/stagebin_microcells \
  --output-dir /path/to/adaptive_landmark_nodes \
  --min-marginal-gain 0.001 \
  --min-cells-per-stratum 50 \
  --allocation-knn 30 \
  --allocation-density-power 1.0 \
  --regroup-knn 64 \
  --faiss-threads 32 \
  --overwrite
```

Strata below the minimum support threshold do not participate in landmark allocation. Their microcells remain in `microcells.parquet` with `landmark_node_id = -1`, and their provenance is recorded in `excluded_sparse_strata.parquet`.

### 3. Construct the developmental tree

```bash
python src/workflow/build_landmark_tree.py \
  --nodes /path/to/adaptive_landmark_nodes/nodes.parquet \
  --embeddings /path/to/adaptive_landmark_nodes/node_embeddings.npy \
  --metadata /path/to/metadata.csv \
  --output-dir /path/to/landmark_tree \
  --stage-column mean_stage \
  --metric cosine \
  --overwrite
```

This step exports node and edge tables, GraphML, run metadata, and global tree visualizations colored by lineage, dominant cell type, and cell-type purity.

### 4. Generate lineage-specific visualizations

```bash
python src/visualization/plot_each_lineage_tree.py \
  --tree-dir /path/to/landmark_tree \
  --width 14 \
  --height 14 \
  --dpi 240
```

Each lineage receives a categorical cell-type visualization and a continuous cell-type-purity visualization.

## Outputs

The adaptive landmark-node directory contains:

| File | Description |
|---|---|
| `nodes.parquet` | Landmark-node annotations and quality metrics |
| `node_embeddings.npy` | Cell-count-weighted landmark embeddings |
| `node_celltype_composition.parquet` | Cell-type composition of each node |
| `microcells.parquet` | Microcell annotations and node assignments |
| `microcell_to_node.npy` | Microcell-to-node mapping; `-1` denotes exclusion |
| `cell_to_node.npy` | Cell-to-node mapping; `-1` denotes exclusion |
| `excluded_sparse_strata.parquet` | Audit table for filtered strata |
| `run_config.json` | Complete run configuration |

The tree directory contains:

| File | Description |
|---|---|
| `tree_nodes.parquet` | Node annotations and tree coordinates |
| `tree_edges.parquet` | Directed parent-child edges and distances |
| `landmark_tree.graphml` | Interoperable directed graph |
| `tree_summary.json` | Structural and quality summary |
| `tree_by_*.png`, `tree_by_*.pdf` | Global tree visualizations |
| `trees_by_lineage/` | Lineage-specific visualizations |

## Source code

- `src/workflow/`: reference microcell construction, adaptive or fixed-count landmark construction, and stage-monotone tree construction
- `src/visualization/`: global supporting visualizations, lineage-specific trees, cell-type-purity plots, and microcell UMAPs
- `src/utilities/`: annotation maintenance and data-preparation utilities
- `src/method_variants/`: earlier and comparative landmark-selection strategies retained for methodological evaluation

## Reproducibility

- Randomized operations use a configurable seed (`42` by default).
- Each output directory records its complete command-line configuration.
- Landmark and microcell identifiers are contiguous for included objects.
- Excluded cells and microcells are represented explicitly by assignment `-1`.
- The tree contains exactly `n_nodes - 1` directed edges.
- Parent-child edges are constrained to be non-decreasing in developmental stage.

## License

A project license should be selected before public release.
