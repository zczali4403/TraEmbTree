# TraEmb metacell developmental tree

This repository provides a reproducible workflow for constructing lineage-aware developmental trees from TraEmb cell embeddings.

Cell-type labels are excluded from microcell construction, Leiden clustering, graph construction, tree extraction, and node contraction. They are joined only after assignments have been fixed and are used for annotation and evaluation.

## Workflow overview

```text
cell embeddings + lineage + predicted stage
                    |
                    v
       cohesive lineage-stage microcells
                    |
                    v
       lineage-local Leiden states
                    |
                    v
          temporal trajectory nodes
                    |
                    v
        stage-directed Markov node DAG
                    |
                    v
       rooted maximum-weight tree
                    |
                    v
       nonbranching-path contraction
```

Known lineage is a hard stratum. Predicted stage is used for fixed-width microcell strata, temporal subdivision, edge direction, early-root selection, and contraction span limits.

## Repository layout

```text
src/
├── workflow/       Core microcell, node, tree, and contraction stages
├── utilities/      Purity evaluation and embedding decoding
└── visualization/  UMAP, PHATE, diffusion-map, and tree visualizations
results/            Local generated outputs; ignored by Git except its README
```

The retained 0914 run, including exact inputs, parameters, statistics, purity, and decoded-expression outputs, is documented in [results/README.md](results/README.md).

## Environment

Install the main dependencies pinned in `requirements.txt`:

```bash
conda activate <environment>
python -m pip install -r requirements.txt
```

Optional utilities additionally use SciPy, `igraph`, `leidenalg`, PyTorch, PHATE, and, for GPU UMAP, CuPy/RAPIDS with `rapids-singlecell`. Run commands from the repository root.

## Input contract

### Cell embeddings

`embeddings.npy` must be a finite, nonzero, two-dimensional NumPy array with one row per cell.

### CSR cell index directory

The directory supplied through `--csr-dir` must contain:

- `cell_ids.npy`: cell IDs aligned row-for-row with the embeddings;
- `lineage_ids.npy`: integer lineage assignments aligned with the embeddings;
- `metadata.json`: at least the ordered `lineages` list;
- `stage_ids.npy` and the ordered `stages` list when original rather than predicted stage is used.

### Predicted stage

The optional predicted-stage CSV must contain `idx`, `cell_id`, and `predicted_stage`. The loader accepts shuffled rows but requires exactly one finite prediction per cell and validates both the embedding/CSR row index and cell ID.

### Cell metadata

The annotation CSV must contain `cell` and `celltype`. Cell type is joined only after microcell assignments have been fixed.

## Workflow

The examples use generic outputs under `results/`. Replace input paths with local data locations.

### 1. Construct cohesive microcells

Cells are separated by lineage and fixed-width stage bins. Two embedding-only cohesive partition passes then construct microcells inside every nonempty stratum.

```bash
python src/workflow/build_cohesive_microcells.py \
  --embeddings /path/to/embeddings.npy \
  --csr-dir /path/to/csr_directory \
  --metadata /path/to/cell_metadata.csv \
  --predicted-stage /path/to/predicted_stage.csv \
  --output-dir results/microcells \
  --stage-bin-width 3 \
  --stage-bin-origin 0 \
  --pile-size 20000 \
  --microcell-size 200 \
  --microcell-min-size 50 \
  --microcell-max-size 400 \
  --microcell-method cohesive \
  --microcell-radius-factor 2 \
  --knn 30 \
  --regroup-knn 64 \
  --faiss-threads 32
```

`--microcell-min-size` is a compactness-checked merge target, not a hard lower bound. Isolated cells and small groups remain when no safe local merge exists.

Principal outputs:

- `cell_to_microcell.npy`: cell-row to microcell assignment;
- `microcell_embeddings.npy`: mean embedding per microcell;
- `microcells.parquet`: size, stage, lineage, compactness, and annotation summaries;
- `microcell_celltype_composition.parquet`: post-hoc cell-type composition;
- `microcell_grouping_summary.json` and `run_config.json`: diagnostics and provenance.

### 2. Discover lineage-local temporal nodes

Cosine kNN and Leiden clustering are computed independently within each lineage using microcell embeddings. By default, each state is divided into fixed-width predicted-stage bins. An optional adaptive method instead finds valleys in a cell-count-weighted stage KDE, merges undersupported adjacent segments, and recursively splits broad segments at feasible cell-weighted medians. Nodes below the final support threshold are excluded without consulting cell type.

```bash
python src/workflow/discover_leiden_trajectory_nodes.py \
  --input-dir results/microcells \
  --output-dir results/trajectory_nodes \
  --knn 30 \
  --leiden-resolution 0.5 \
  --leiden-iterations 2 \
  --stage-bin-width 3 \
  --min-node-cells 50 \
  --threads 32 \
  --seed 42
```

Adaptive temporal subdivision avoids global stage-bin boundaries:

```bash
python src/workflow/discover_leiden_trajectory_nodes.py \
  --input-dir results/microcells \
  --output-dir results/trajectory_nodes_kde \
  --knn 30 \
  --leiden-resolution 0.5 \
  --leiden-iterations 2 \
  --temporal-split-method kde \
  --stage-kde-bandwidth 0.5 \
  --stage-min-peak-distance 1.0 \
  --stage-valley-ratio 0.6 \
  --max-node-stage-span 3 \
  --min-segment-metacells 3 \
  --min-node-cells 50 \
  --threads 32 \
  --seed 42
```

The maximum-span rule is a support-constrained safeguard: a broad segment is split only when both sides retain the requested minimum cells and metacells. `temporal_split_diagnostics.parquet` records KDE peaks, accepted valleys, small-segment merges, median splits, and final segment counts for every Leiden state.

Principal outputs:

- `nodes.parquet` and `node_embeddings.npy`: retained temporal nodes;
- `states.parquet`: lineage-local Leiden states;
- `microcell_node_assignments.parquet`: assignments and audit IDs;
- `filtered_small_nodes.parquet`: excluded low-support nodes;
- `temporal_split_diagnostics.parquet`: per-state adaptive-split audit table when KDE is used;
- composition tables, `summary.json`, `run_config.json`, and `complete.json`.

Excluded microcells retain `raw_node_id` and receive `node_id=-1`. Retained nodes are renumbered contiguously from zero.

### 3. Build the lineage-aware Markov tree

An exact cosine kNN graph is built independently for each lineage. Predicted stage directs real edges from earlier to later, and early nodes connect to a lineage virtual root. Markov quantities are calculated on the candidate DAG before a rooted maximum-weight tree is extracted.

```bash
python src/workflow/build_markov_node_tree.py \
  --input-dir results/trajectory_nodes \
  --microcell-dir results/microcells \
  --output-dir results/markov_tree \
  --knn 10 \
  --mutual-knn-bonus 1.5 \
  --stage-column cell_weighted_mean_stage \
  --root-stage-window 0.5 \
  --dpi 220
```

Principal outputs:

- `candidate_markov_edges.parquet`: directed candidate graph;
- `node_fate_probabilities.parquet`: terminal fate probabilities;
- `tree_nodes.parquet` and `tree_edges.parquet`: extracted tree;
- `branch_segments.parquet`: compressed paths for reporting;
- lineage summaries, plots, `summary.json`, and `run_config.json`.

Nearest-earlier-node fallback edges ensure reachability when a node has no valid incoming edge in its directed kNN neighborhood.

### 4. Contract redundant tree nodes

By default, only nodes along nonbranching paths can merge. Lineage roots and anchor-to-anchor edges are protected. Merge decisions use embedding distance and stage only.

```bash
python src/workflow/contract_markov_tree_nodes.py \
  --tree-dir results/markov_tree \
  --node-dir results/trajectory_nodes \
  --output-dir results/contracted_tree \
  --max-cosine-distance 0.15 \
  --max-stage-span 8 \
  --dpi 220
```

Optional topology-simplifying sibling merging runs after path contraction. Two siblings may merge when they share a real temporal-node parent, have cosine distance at most 0.20 and mean-stage gap at most 1.0, and at least one is a leaf. Direct children of lineage virtual roots are not merged. A second path-contraction pass then removes any new nonbranching chains created by sibling merging.

```bash
python src/workflow/contract_markov_tree_nodes.py \
  --tree-dir results/markov_tree \
  --node-dir results/trajectory_nodes \
  --output-dir results/contracted_tree_with_siblings \
  --max-cosine-distance 0.15 \
  --max-stage-span 8 \
  --merge-siblings \
  --sibling-max-cosine-distance 0.20 \
  --sibling-max-stage-gap 1.0 \
  --dpi 220
```

Principal outputs:

- `tree_nodes.parquet` and `tree_edges.parquet`: final topology;
- `node_embeddings.npy`: cell-count-weighted contracted embeddings;
- `source_to_contracted_nodes.parquet`: exact old-to-new mapping;
- `sibling_merge_history.parquet`: ordered sibling-merge audit trail when enabled;
- `node_celltype_composition.parquet`: post-contraction annotation;
- plots, `summary.json`, and `run_config.json`.

The script verifies that branch topology, total cell count, and total microcell count are preserved. Markov probabilities are not recomputed after contraction.

Regenerate plots without rerunning contraction:

```bash
python src/workflow/contract_markov_tree_nodes.py \
  --tree-dir results/markov_tree \
  --output-dir results/contracted_tree \
  --plots-only \
  --lineage-node-size 45 \
  --lineage-node-label-size 5 \
  --dpi 220
```

Per-lineage tree points are labeled with contracted `node_id`.

## Evaluation

```bash
python src/utilities/evaluate_microcell_purity.py \
  --input-dir results/microcells \
  --output-dir results/microcells/purity_evaluation \
  --overwrite

python src/utilities/evaluate_node_purity.py \
  --input-dir results/contracted_tree \
  --output-dir results/contracted_tree/purity_evaluation \
  --overwrite
```

Purity is an external evaluation using post-hoc annotations, not an optimization target.

## Decode contracted-node embeddings

```bash
python src/utilities/decode_node_embeddings.py \
  --embedding-path results/contracted_tree/node_embeddings.npy \
  --node-table results/contracted_tree/tree_nodes.parquet \
  --checkpoint /path/to/model.ckpt \
  --traemb-dir /path/to/TraEmb \
  --config /path/to/config.py \
  --gene-names /path/to/gene_names.txt \
  --output-dir results/contracted_tree/decoder_expression
```

The expression matrix is aligned by contiguous `node_id`. `node_decoded_nodes.csv` annotates rows and `node_decoded_genes.csv` annotates columns.

## Visualization

```bash
python src/visualization/plot_microcell_umap_scanpy.py \
  --input-dir results/microcells \
  --n-neighbors 30 \
  --min-dist 0.25 \
  --seed 42 \
  --dpi 220
```

Additional scripts provide RAPIDS UMAP, PHATE, diffusion maps, stage-subspace diagnostics, contracted-node UMAP, and sampled-cell tree views. Run a script with `--help` for its complete interface.

For sampled-cell tree plots, point density reflects independent per-node sampling rather than population abundance.

## Reproducibility and output safety

- Workflow stages record configuration and summary JSON files with their outputs.
- Most stages refuse to overwrite existing output directories.
- Atomic-build directories preserve partial output after failure.
- Fixed seeds control randomized operations where applicable.
- Generated arrays, tables, figures, and result directories are excluded from Git; source and documentation remain tracked.
