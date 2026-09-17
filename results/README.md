# Retained 0914 reference run

This directory contains the generated artifacts from the retained 0914 TraEmb developmental-tree run. The files are local analysis outputs and are excluded from Git; this document is the tracked record of the run.

## Output directories

```text
results/
├── microcells_0914_predicted_stage3/
├── leiden_nodes_0914_r05_stage3_min50/
├── markov_tree_0914_r05_stage3_min50_nodeknn_k10_rootw05/
└── markov_tree_0914_r05_stage3_min50_nodeknn_k10_rootw05_contracted_d015_stage8/
```

## Input provenance

- Cell embeddings: `/mnt/input/sc_cz/Concord/eval/2026_09_03/embeddings.npy`
- Predicted developmental stage: `/mnt/input/sc_cz/Concord/eval/2026_09_03/predicted_stage.csv`
- Cell metadata: `/mnt/input/sc_cz/Concord/data/all_lineage_260829_liver_reanno.csv`
- CSR cell index directory: `/scratch/amlt_code/traemb_csr_0829`

The predicted-stage loader validated every embedding/CSR row index against its cell ID. Cell-type annotations were joined only after assignments were fixed and were not used for grouping, graph construction, or tree contraction.

## Summary

| Quantity | Value |
|---|---:|
| Input cells | 15,203,034 |
| Embedding dimensions | 100 |
| Lineages | 19 |
| Metacells | 792,038 |
| Leiden embedding states | 283 |
| Raw temporal nodes | 2,117 |
| Retained temporal nodes | 1,613 |
| Candidate Markov edges | 10,075 |
| Contracted temporal nodes | 924 |
| Contracted-tree branch points | 197 |
| Contracted-tree terminal leaves | 315 |

## 1. Cohesive metacells

Output: `microcells_0914_predicted_stage3/`

Important parameters:

| Parameter | Value |
|---|---:|
| Predicted-stage bin width | 3 |
| Pile size | 20,000 |
| Target metacell size | 200 |
| Minimum merge target | 50 |
| Maximum metacell size | 400 |
| Cohesive radius factor | 2 |
| kNN | 30 |
| Regroup kNN | 64 |

The run produced 792,038 metacells across 199 nonempty lineage-stage strata. Median size was 16 cells, 84,392 groups were singletons, and 80.65% of cells belonged to groups smaller than 50 cells.

The minimum size is a compactness-checked merge target, not a hard lower bound. Small or singleton groups are retained when no safe local merge exists.

## 2. Lineage-local temporal nodes

Output: `leiden_nodes_0914_r05_stage3_min50/`

Important parameters:

| Parameter | Value |
|---|---:|
| kNN | 30 |
| Leiden resolution | 0.5 |
| Leiden iterations | 2 |
| Temporal bin width | 3 |
| Minimum supporting cells | 50 |
| Random seed | 42 |

The 283 embedding states produced 2,117 raw temporal nodes. Filtering removed 504 nodes containing 8,993 cells, or 0.0592% of all cells, leaving 1,613 retained nodes. Excluded metacells retain `raw_node_id` and use `node_id=-1` in the filtered assignment table.

## 3. Markov node tree

Output: `markov_tree_0914_r05_stage3_min50_nodeknn_k10_rootw05/`

Important parameters:

| Parameter | Value |
|---|---:|
| Node kNN | 10 |
| Mutual-kNN bonus | 1.5 |
| Root-stage window | 0.5 |
| Stage column | `cell_weighted_mean_stage` |

The stage-directed candidate DAG contains 10,075 edges, including 6,014 mutual-kNN edges and 41 nearest-earlier-node fallback edges. All 41 fallback edges were selected into the extracted tree.

The complete extracted tree contains 1,613 temporal nodes, 19 lineage virtual roots, one global virtual root, and 1,632 edges. The candidate DAG has 85 absorbing sinks; this is distinct from the number of leaves in the extracted tree.

## 4. Tree contraction

Output: `markov_tree_0914_r05_stage3_min50_nodeknn_k10_rootw05_contracted_d015_stage8/`

Important parameters:

| Parameter | Value |
|---|---:|
| Maximum cosine distance | 0.15 |
| Maximum cell-level stage span | 8 |

Contraction reduced 1,613 temporal nodes to 924, removing 689 redundant nodes. The branch topology was preserved: 66 early-root temporal nodes, 197 branch points, and 315 terminal leaves were unchanged. A contracted node contains at most five source nodes; the median is two.

`source_to_contracted_nodes.parquet` records the exact source-to-contracted mapping. Markov probabilities were not recomputed after contraction.

## Cell-type purity

Cell-type labels were used only for post-hoc evaluation.

| Object | Cell-weighted purity |
|---|---:|
| Metacells | 0.8618 |
| Temporal nodes before contraction | 0.7050 |
| Contracted nodes | 0.6919 |

All 15,203,034 metacell annotations were matched. Contracted-node evaluation covers 15,194,041 cells after the 8,993-cell temporal-node support filter.

## Decoded expression

The contracted-node embeddings were decoded into expression estimates under:

```text
markov_tree_0914_r05_stage3_min50_nodeknn_k10_rootw05_contracted_d015_stage8/
└── decoder_expression/
    ├── node_decoded_expression.npy
    ├── node_decoded_genes.csv
    ├── node_decoded_nodes.csv
    └── run_config.json
```

The decoder produced a 924 × 6,737 matrix. Values are the ZINB mean parameter `mu = softplus(decoder(node_embedding))`.

Rows of `node_decoded_expression.npy` align directly with the contiguous `node_id` values in `node_decoded_nodes.csv`. The numbered points in `trees_by_lineage_celltype/` use the same `node_id`.

## Interpretation notes

- Known lineage is a hard stratum throughout this workflow; cell type is never a grouping or edge-selection input.
- Predicted stage is used for metacell stratification, temporal-node subdivision, edge direction, and contraction span limits.
- Point density in sampled-cell tree plots does not represent abundance because cells are sampled independently within every node.
- Paths stored in `run_config.json` record the locations used when the run was executed. They are provenance fields and were not rewritten when the output directories were moved under `results/`.
