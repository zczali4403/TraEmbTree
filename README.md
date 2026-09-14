# TraEmb metacell developmental tree

This repository contains the retained workflow for constructing a developmental tree from TraEmb embeddings without using cell-type labels during grouping, clustering, graph construction, or tree contraction.

## Inputs and annotation provenance

- Cell embeddings: `/mnt/input/sc_cz/Concord/eval/2026_09_03/embeddings.npy`
- Predicted developmental stage: `/mnt/input/sc_cz/Concord/eval/2026_09_03/predicted_stage.csv`
- Cell metadata: `/mnt/input/sc_cz/Concord/data/all_lineage_260829_liver_reanno.csv`
- CSR cell index directory: `/scratch/amlt_code/traemb_csr_0829`

`predicted_stage.csv` must contain `idx`, `cell_id`, and `predicted_stage`. The loader verifies both the embedding/CSR row index and cell ID before using a prediction.

Annotation provenance: the retained microcell run records `all_lineage_260829.csv` in its run configuration. Liver cell-type annotations were updated afterwards without changing microcell assignments or embeddings; pre-reannotation Parquet tables are retained with the suffix `.before_liver_reannotation_20260906_170745`. Cell type remains post-hoc annotation only.

## Environment

The workflow is run in the `train` conda environment:

```bash
conda activate train
pip install -r requirements.txt
```

The pinned requirements cover the main workflow. Optional utilities also use SciPy, `igraph`, `leidenalg`, PyTorch, PHATE, and, for GPU UMAP, CuPy/RAPIDS with `rapids-singlecell`; these are supplied by the retained `train` environment.

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

Cosine kNN and Leiden clustering are computed independently inside each of the 19 known lineages using metacell embeddings only. Each Leiden state is then subdivided by predicted-stage bins of width 3. Temporal nodes with fewer than `--min-node-cells` cells are excluded from downstream node outputs without using cell type. Their original IDs and assignments remain in audit columns and tables.

```bash
python src/workflow/discover_leiden_trajectory_nodes.py \
  --input-dir microcells_0903_predicted_stage \
  --output-dir leiden_nodes_0903_by_lineage_r05_stage3_min50 \
  --knn 30 \
  --leiden-resolution 0.5 \
  --leiden-iterations 2 \
  --stage-bin-width 3 \
  --min-node-cells 50 \
  --threads 32 \
  --seed 42
```

The output includes `filtered_small_nodes.parquet`; excluded metacells retain `raw_node_id` and receive `node_id=-1`. Retained nodes are renumbered contiguously for all downstream scripts. `summary.json` reports both raw and retained node counts and the excluded cell fraction.

Current result: 297 embedding states produce 2,134 raw temporal nodes. After the 50-cell support filter, 1,617 nodes remain. The 517 excluded nodes contain 8,404 cells, or 0.0553% of all cells.

### 3. Build the lineage-aware Markov node tree

For each lineage, an exact cosine kNN graph is built from the node embeddings. Predicted stage orients every real edge from earlier to later. All nodes within `--root-stage-window` of the lineage's earliest retained stage connect directly to its virtual root; incoming real-node edges to this early root set are removed. Embedding distance determines edge weight; cell type is excluded. Markov transition probabilities and terminal fate probabilities are calculated before extracting one rooted tree.

```bash
python src/workflow/build_markov_node_tree.py \
  --input-dir leiden_nodes_0903_by_lineage_r05_stage3_min50 \
  --microcell-dir microcells_0903_predicted_stage \
  --output-dir markov_tree_0903_r05_stage3_min50_nodeknn_k10_rootw05 \
  --knn 10 \
  --mutual-knn-bonus 1.5 \
  --stage-column cell_weighted_mean_stage \
  --root-stage-window 0.5 \
  --dpi 220
```

Current result: 10,147 candidate edges and 1,617 temporal nodes. With 19 lineage virtual roots and one global virtual root, the complete tree has 1,637 nodes and 1,636 edges. The candidate graph contains 46 nearest-earlier-node fallback edges; all 46 are selected into the tree.

The Markov summary counts absorbing sinks in the full candidate DAG; this is a different quantity from the extracted tree leaf count reported during contraction.

### 4. Contract redundant nodes after tree construction

Only consecutive degree-two nodes on nonbranching paths can merge. Lineage roots, branch points, and terminal nodes are fixed anchors. A candidate node joins the current group only when its cosine distance to the weighted group center is at most 0.10 and the resulting cell-level stage span is at most 6. Cell type is aggregated after contraction and never affects merging.

```bash
python src/workflow/contract_markov_tree_nodes.py \
  --tree-dir markov_tree_0903_r05_stage3_min50_nodeknn_k10_rootw05 \
  --node-dir leiden_nodes_0903_by_lineage_r05_stage3_min50 \
  --output-dir markov_tree_0903_r05_stage3_min50_nodeknn_k10_rootw05_contracted_d010_stage6 \
  --max-cosine-distance 0.10 \
  --max-stage-span 6 \
  --dpi 220
```

Current result: 1,617 temporal nodes contract to 1,179, removing 438 redundant nodes. The extracted-tree topology is unchanged: 73 early-root temporal nodes, 185 branch points, and 320 terminal leaves are preserved. The 73 early-root nodes attach to the 19 lineage virtual roots. `source_to_contracted_nodes.parquet` records the exact old-to-new node mapping.

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
  --input-dir leiden_nodes_0903_by_lineage_r05_stage3_min50 \
  --output-dir leiden_nodes_0903_by_lineage_r05_stage3_min50/purity_evaluation \
  --overwrite
```

Evaluate contracted-node purity:

```bash
python src/utilities/evaluate_node_purity.py \
  --input-dir markov_tree_0903_r05_stage3_min50_nodeknn_k10_rootw05_contracted_d010_stage6 \
  --output-dir markov_tree_0903_r05_stage3_min50_nodeknn_k10_rootw05_contracted_d010_stage6/purity_evaluation \
  --overwrite
```

Current cell-weighted cell-type purities are 0.8636 for microcells, 0.7024 before contraction, and 0.6927 after contraction. Cell-type labels are evaluation annotations, not grouping or tree inputs.

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
  --tree-dir markov_tree_0903_r05_stage3_min50_nodeknn_k10_rootw05 \
  --node-dir leiden_nodes_0903_by_lineage_r05_stage3_min50 \
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
src/utilities/decode_node_embeddings.py
src/visualization/plot_microcell_umap_scanpy.py
src/visualization/plot_microcell_umap_rsc.py
src/visualization/plot_contracted_nodes_umap_rsc.py
src/visualization/plot_microcell_phate.py
src/visualization/plot_microcell_diffusion_map.py
src/visualization/diagnose_microcell_stage_subspace.py
src/visualization/sweep_microcell_stage_dimension_weight.py
src/visualization/plot_markov_tree_sampled_cells.py
```
