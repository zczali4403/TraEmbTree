# Retained 0914 adaptive temporal-node reference run

This directory contains local artifacts from the retained 0914 TraEmb developmental-tree analysis. Generated files are excluded from Git; this README is the tracked record of the retained configuration and its principal diagnostics.

## Output directories

```text
results/
|-- microcells_0914_predicted_stage3/
|-- leiden_nodes_0914_r05_kde_bw01_peak05_valley08_span5_min50/
|-- markov_tree_0914_r05_kde_bw01_peak05_valley08_span5_min50_nodeknn_k10_rootw10/
`-- markov_tree_0914_r05_kde_bw01_peak05_valley08_span5_min50_nodeknn_k10_rootw10_contracted_d04_meanstage5_sibling_d04_gap3/
```

The previous fixed-stage-bin baseline is retained separately:

```text
results/
|-- leiden_nodes_0914_r05_stage3_min50/
|-- markov_tree_0914_r05_stage3_min50_nodeknn_k10_rootw05/
`-- markov_tree_0914_r05_stage3_min50_nodeknn_k10_rootw05_contracted_d04_stage8_sibling_d04_gap2/
```

## Input provenance

- Cell embeddings: `/mnt/input/sc_cz/Concord/eval/2026_09_03/embeddings.npy`
- Predicted developmental stage: `/mnt/input/sc_cz/Concord/eval/2026_09_03/predicted_stage.csv`
- Cell metadata: `/mnt/input/sc_cz/Concord/data/all_lineage_260829_liver_reanno.csv`
- CSR cell index directory: `/scratch/amlt_code/traemb_csr_0829`

The predicted-stage loader validated every embedding/CSR row index against its cell ID. Known lineage remained a hard stratum. Cell-type annotations were joined only after assignments were fixed and were not used for microcell grouping, Leiden clustering, temporal segmentation, graph construction, tree extraction, or contraction.

## Summary

| Quantity | Value |
|---|---:|
| Input cells | 15,203,034 |
| Embedding dimensions | 100 |
| Lineages | 19 |
| Metacells | 792,038 |
| Leiden embedding states | 285 |
| Temporal nodes before contraction | 2,099 |
| Candidate Markov edges | 12,871 |
| Contracted temporal nodes | 824 |
| Final branch nodes | 187 |
| Final terminal nodes | 312 |
| Sibling-merge events | 72 |
| Contracted-node cell-weighted purity | 0.6819 |

## 1. Cohesive metacells

Output: `microcells_0914_predicted_stage3/`

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

The run contains 792,038 metacells across 199 nonempty lineage-stage strata. Median size is 16 cells, 84,392 groups are singletons, and every input cell is assigned. `microcell_min_size=50` is a compactness-checked merge target rather than a hard lower bound.

## 2. Lineage-local adaptive temporal nodes

Output: `leiden_nodes_0914_r05_kde_bw01_peak05_valley08_span5_min50/`

| Parameter | Value |
|---|---:|
| kNN | 30 |
| Leiden resolution | 0.5 |
| Leiden iterations | 2 |
| Temporal split method | KDE |
| KDE bandwidth | 0.1 |
| Minimum KDE peak distance | 0.5 |
| Maximum valley ratio | 0.8 |
| Maximum node stage span | 5 |
| Minimum supporting cells | 50 |
| Random seed | 42 |
| Threads | 20 |

Leiden clustering is embedding-only and is run independently within each lineage. Within each Leiden state, temporal subdivision:

1. splits at accepted valleys in the cell-count-weighted predicted-stage KDE;
2. merges adjacent temporal segments containing fewer than 50 cells;
3. recursively splits remaining broad segments at feasible cell-weighted medians when their member-metacell mean-stage span exceeds 5.

Segment support is determined only by represented cell count; no minimum metacell count is imposed. A weighted-median split is accepted only when both resulting sides retain at least 50 cells.

The 285 embedding states produced 2,099 temporal nodes. No node, metacell, or cell was removed by the final support filter.

`temporal_split_diagnostics.parquet` records 3,088 accepted KDE valleys, 1,446 low-support adjacent-segment merges, and 172 weighted-median split operations across 75 states. Of the 1,814 effective subdivisions beyond the 285 states, 1,642 (90.5%) came from net KDE segmentation and 172 (9.5%) from span-constrained median splitting.

## 3. Stage-directed Markov node tree

Output: `markov_tree_0914_r05_kde_bw01_peak05_valley08_span5_min50_nodeknn_k10_rootw10/`

| Parameter | Value |
|---|---:|
| Node kNN | 10 |
| Mutual-kNN bonus | 1.5 |
| Root-stage window | 1.0 |
| Stage column | `cell_weighted_mean_stage` |
| Distance temperature | lineage median, selected automatically |
| Fate-probability output threshold | 0.0001 |

The candidate DAG contains 12,871 directed edges, including 8,085 mutual-kNN edges and 57 nearest-earlier-node fallback edges. All 57 fallback edges were selected into the extracted tree.

The extracted tree contains 2,099 temporal nodes, 19 lineage roots, one global root, and 2,118 edges. The candidate DAG has 113 absorbing sinks, and the extracted tree has 640 reported branch segments. `root_stage_window=1.0` connects nodes within one stage unit after each lineage minimum directly to the lineage root.

## 4. Mean-stage path contraction and sibling merging

Output: `markov_tree_0914_r05_kde_bw01_peak05_valley08_span5_min50_nodeknn_k10_rootw10_contracted_d04_meanstage5_sibling_d04_gap3/`

| Parameter | Value |
|---|---:|
| Maximum path cosine distance | 0.4 |
| Maximum path node-mean-stage span | 5 |
| Sibling merging | enabled |
| Maximum sibling cosine distance | 0.4 |
| Maximum sibling mean-stage gap | 3.0 |
| Mixed display threshold | 0.0 (disabled) |

For path contraction, `max_stage_span=5` is the maximum difference among source nodes' `cell_weighted_mean_stage` values. It does not constrain the complete member-metacell stage envelope.

Only nonbranching paths are eligible for path contraction. Structural anchors remain protected. Sibling merging combines eligible siblings sharing a real temporal-node parent when they satisfy the cosine and mean-stage thresholds and at least one is a leaf. Direct children of lineage roots remain protected. A second path pass removes nonbranching chains created by sibling merging.

Contraction reduced 2,099 temporal nodes to 824. Initial path contraction removed 1,000 nodes, 72 sibling merges simplified eligible local topology, and the second path pass removed another 203 nodes. Branch nodes decreased from 237 to 187 and terminal nodes from 384 to 312 while preserving 84 early-root temporal nodes. A final contracted node contains at most 12 source nodes; the median is two.

`source_to_contracted_nodes.parquet` records the exact mapping and `sibling_merge_history.parquet` records every sibling merge. Markov probabilities were not recomputed after contraction. The path mean-stage span applies only to path contraction; sibling merging uses its separate mean-stage-gap criterion.

## 5. Cell-type annotation and purity

Cell-type labels are used only for post-hoc annotation and evaluation. The retained plots use `mixed_celltype_purity_threshold=0`, so every temporal node is colored by its plurality cell type and low-purity nodes are not relabeled as `Mixed`.

| Quantity | Value |
|---|---:|
| Evaluated temporal nodes | 824 |
| Evaluated cells | 15,203,034 |
| Mean node purity | 0.7393 |
| Median node purity | 0.7795 |
| Q10 node purity | 0.4117 |
| Q90 node purity | 0.9953 |
| Cell-weighted node purity | 0.6819 |

Purity is the largest known-celltype count divided by the total annotated-cell count in each contracted node. These values come from `node_celltype_composition.parquet` and do not influence clustering, edge selection, or merging.

## Principal audit files

- `run_config.json`: exact arguments and input paths for each stage;
- `summary.json`: stage-level counts and validation summaries;
- `temporal_split_diagnostics.parquet`: KDE valleys, low-support merges, median splits, and final segments;
- `candidate_markov_edges.parquet`: candidate edges and selected-tree flags;
- `tree_nodes.parquet` and `tree_edges.parquet`: extracted or contracted topology;
- `source_to_contracted_nodes.parquet`: source-to-contracted mapping;
- `sibling_merge_history.parquet`: ordered sibling-merge history;
- `node_celltype_composition.parquet`: post-hoc node composition.

## Interpretation notes

- Known lineage is a hard stratum throughout the workflow.
- Cell type is annotation and evaluation only; it is never a grouping, temporal-splitting, edge-selection, or merge input.
- Microcell construction uses fixed-width stage strata; adaptive KDE subdivision is applied after lineage-local Leiden clustering.
- Adaptive temporal support is cell-count based. A broad segment can remain wider than the configured threshold when no feasible split preserves the minimum cell count on both sides.
- Path contraction uses node-level cell-weighted mean-stage span. Final member-metacell stage envelopes can be wider.
- Sibling merging intentionally changes eligible local topology and is fully audited.
- `Mixed` display is disabled in the retained plots; colors always indicate the plurality cell type.
- Paths in `run_config.json` are provenance fields and record the locations used when each stage was executed.
