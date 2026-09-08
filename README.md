# TraEmb metacell developmental tree

This repository contains the retained workflow for constructing a developmental tree from TraEmb embeddings without using cell-type labels during grouping, clustering, graph construction, or tree contraction.

## Inputs

- Cell embeddings: `/mnt/input/sc_cz/Concord/eval/2026_09_03/embeddings.npy`
- Predicted developmental stage: `/mnt/input/sc_cz/Concord/eval/2026_09_03/predicted_stage.csv`
- Cell metadata: `/mnt/input/sc_cz/Concord/data/all_lineage_260829_liver_reanno.csv`
- CSR cell index directory: `/scratch/amlt_code/traemb_csr_0829`

`predicted_stage.csv` must contain `idx`, `cell_id`, and `predicted_stage`. The loader verifies both the embedding/CSR row index and cell ID before using a prediction.

## Environment

The workflow is run in the `train` conda environment:

```bash
conda activate train
pip install -r requirements.txt
```

Run commands from the repository root:

```bash
cd /mnt/input/sc_cz/Concord/eval/2026_08_19/TraEmbTree
```

## Workflow

### 1. Construct cohesive metacells

Cells are first separated by lineage and fixed-width predicted-stage bins. Within each stratum, two embedding-only cohesive partition passes construct metacells. Cell type is joined only after all assignments have been fixed.

```bash
python src/workflow/build_cohesive_microcells.py \
  --embeddings /mnt/input/sc_cz/Concord/eval/2026_09_03/embeddings.npy \
  --csr-dir /scratch/amlt_code/traemb_csr_0829 \
  --metadata /mnt/input/sc_cz/Concord/data/all_lineage_260829_liver_reanno.csv \
  --predicted-stage /mnt/input/sc_cz/Concord/eval/2026_09_03/predicted_stage.csv \
  --output-dir microcells_0903_predicted_stage \
  --stage-bin-width 2 \
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

Current result: 15,203,034 cells in 816,475 metacells.

### 2. Discover lineage-local trajectory nodes

Cosine kNN and Leiden clustering are computed independently inside each of the 19 known lineages using metacell embeddings only. Each Leiden state is then subdivided by predicted-stage bins of width 2.

```bash
python src/workflow/discover_leiden_trajectory_nodes.py \
  --input-dir microcells_0903_predicted_stage \
  --output-dir leiden_nodes_0903_by_lineage_r05_stage2 \
  --knn 30 \
  --leiden-resolution 0.5 \
  --leiden-iterations 2 \
  --stage-bin-width 2 \
  --threads 32 \
  --seed 42
```

Current result: 297 Leiden states and 3,141 temporal nodes.

### 3. Build the lineage-aware Markov node tree

For each lineage, an exact cosine kNN graph is built from the 3,141 node embeddings. Predicted stage orients every real edge from earlier to later. Embedding distance determines edge weight; cell type is excluded. Markov transition probabilities and terminal fate probabilities are calculated before extracting one rooted tree.

```bash
python src/workflow/build_markov_node_tree.py \
  --input-dir leiden_nodes_0903_by_lineage_r05_stage2 \
  --microcell-dir microcells_0903_predicted_stage \
  --output-dir markov_tree_0903_nodeknn_k10 \
  --knn 10 \
  --mutual-knn-bonus 1.5 \
  --stage-column cell_weighted_mean_stage \
  --dpi 220
```

Current result: 19,178 candidate edges, 3,141 temporal nodes, and one tree containing all 19 lineage subtrees.

### 4. Contract redundant nodes after tree construction

Only consecutive degree-two nodes on nonbranching paths can merge. Lineage roots, branch points, and terminal nodes are fixed anchors. A candidate node joins the current group only when its cosine distance to the weighted group center is at most 0.08 and the resulting cell-level stage span is at most 6. Cell type is aggregated after contraction and never affects merging.

```bash
python src/workflow/contract_markov_tree_nodes.py \
  --tree-dir markov_tree_0903_nodeknn_k10 \
  --node-dir leiden_nodes_0903_by_lineage_r05_stage2 \
  --output-dir markov_tree_0903_nodeknn_k10_contracted_d008_stage6 \
  --max-cosine-distance 0.08 \
  --max-stage-span 6 \
  --dpi 220
```

Current result: 3,141 nodes contract to 1,912. The 19 lineage roots, 351 branch points, and 493 terminal nodes are unchanged. `source_to_contracted_nodes.parquet` records the exact old-to-new node mapping.

The contracted output represents the final tree topology. Markov probabilities are not recomputed after contraction.

## Evaluation and visualization

Evaluate metacell purity:

```bash
python src/utilities/evaluate_microcell_purity.py \
  --input-dir microcells_0903_predicted_stage \
  --output-dir microcells_0903_predicted_stage/purity_evaluation \
  --overwrite
```

Evaluate pre-contraction trajectory-node purity:

```bash
python src/utilities/evaluate_node_purity.py \
  --input-dir leiden_nodes_0903_by_lineage_r05_stage2 \
  --output-dir leiden_nodes_0903_by_lineage_r05_stage2/purity_evaluation \
  --overwrite
```

Compute two- and three-dimensional metacell UMAPs:

```bash
python src/visualization/plot_microcell_umap_scanpy.py \
  --input-dir microcells_0903_predicted_stage \
  --n-neighbors 30 \
  --min-dist 0.25 \
  --seed 42 \
  --dpi 220
```

Represent the uncontracted tree using stratified samples of its underlying cells. Each cell uses its own predicted stage on the horizontal axis; the tree topology is unchanged.

```bash
python src/visualization/plot_markov_tree_sampled_cells.py \
  --tree-dir markov_tree_0903_nodeknn_k10 \
  --node-dir leiden_nodes_0903_by_lineage_r05_stage2 \
  --microcell-dir microcells_0903_predicted_stage \
  --predicted-stage /mnt/input/sc_cz/Concord/eval/2026_09_03/predicted_stage.csv \
  --metadata /mnt/input/sc_cz/Concord/data/all_lineage_260829_liver_reanno.csv \
  --cells-per-node 150 \
  --seed 42 \
  --dpi 220
```

Because cells are sampled separately within every node, point density in the sampled-cell figure does not represent population abundance.

## Retained source files

```text
src/workflow/build_cohesive_microcells.py
src/workflow/cohesive_microcells.py
src/workflow/predicted_stage.py
src/workflow/discover_leiden_trajectory_nodes.py
src/workflow/build_markov_node_tree.py
src/workflow/contract_markov_tree_nodes.py
src/utilities/evaluate_microcell_purity.py
src/utilities/evaluate_node_purity.py
src/visualization/plot_microcell_umap_scanpy.py
src/visualization/plot_markov_tree_sampled_cells.py
```
