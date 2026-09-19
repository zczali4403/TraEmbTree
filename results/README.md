# Retained 0914 adaptive temporal-node reference run

This directory contains local artifacts from the retained 0914 TraEmb developmental-tree analysis. Generated files are excluded from Git; this README is the tracked record of the retained configuration and its principal diagnostics.

## Output directories

```text
results/
├── microcells_0914_predicted_stage3/
├── leiden_nodes_0914_r05_kde_bw05_span4_min50/
├── markov_tree_0914_r05_kde_bw05_span4_min50_nodeknn_k10_rootw10/
└── markov_tree_0914_r05_kde_bw05_span4_min50_nodeknn_k10_rootw10_contracted_d04_meanstage5_sibling_d04_gap2/
```

The previous fixed-stage-bin baseline is retained separately:

```text
results/
├── leiden_nodes_0914_r05_stage3_min50/
├── markov_tree_0914_r05_stage3_min50_nodeknn_k10_rootw05/
└── markov_tree_0914_r05_stage3_min50_nodeknn_k10_rootw05_contracted_d04_stage8_sibling_d04_gap2/
```

## Input provenance

- Cell embeddings: `/mnt/input/sc_cz/Concord/eval/2026_09_03/embeddings.npy`
- Predicted developmental stage: `/mnt/input/sc_cz/Concord/eval/2026_09_03/predicted_stage.csv`
- Cell metadata: `/mnt/input/sc_cz/Concord/data/all_lineage_260829_liver_reanno.csv`
- CSR cell index directory: `/scratch/amlt_code/traemb_csr_0829`

The predicted-stage loader validated every embedding/CSR row index against its cell ID. Known lineage remained a hard stratum. Cell-type annotations were joined only after assignments were fixed and were not used for microcell grouping, Leiden clustering, temporal splitting, graph construction, tree extraction, or contraction.

## Summary

| Quantity | Value |
|---|---:|
| Input cells | 15,203,034 |
| Embedding dimensions | 100 |
| Lineages | 19 |
| Metacells | 792,038 |
| Leiden embedding states | 281 |
| Temporal nodes before contraction | 2,076 |
| Candidate Markov edges | 12,652 |
| Contracted temporal nodes | 823 |
| Final branch nodes | 161 |
| Final terminal nodes | 326 |
| Sibling-merge events | 47 |
| Contracted-node cell-weighted purity | 0.6857 |

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

The run contains 792,038 metacells across 199 nonempty lineage-stage strata. Median size is 16 cells, 84,392 groups are singletons, and every input cell is assigned.

`microcell_min_size=50` is a compactness-checked merge target rather than a hard lower bound. Small or singleton groups remain when no safe local merge exists.

## 2. Lineage-local adaptive temporal nodes

Output: `leiden_nodes_0914_r05_kde_bw05_span4_min50/`

Important parameters:

| Parameter | Value |
|---|---:|
| kNN | 30 |
| Leiden resolution | 0.5 |
| Leiden iterations | 2 |
| Temporal split method | KDE |
| KDE bandwidth | 0.5 |
| Minimum KDE peak distance | 1.0 |
| Maximum valley ratio | 0.6 |
| Maximum node stage span | 4 |
| Minimum segment metacells | 3 |
| Minimum supporting cells | 50 |
| Random seed | 42 |

Leiden clustering is embedding-only and is run independently within each lineage. Within every Leiden state, temporal subdivision proceeds in three steps:

1. split at accepted valleys in the cell-count-weighted predicted-stage KDE;
2. merge adjacent temporal segments that do not meet the minimum cell or metacell support;
3. recursively split remaining broad segments at feasible cell-weighted medians when their member-microcell mean-stage span exceeds 4.

The 281 embedding states produced 2,076 temporal nodes. No node, metacell, or cell was removed by the final support filter.

`temporal_split_diagnostics.parquet` records the full audit trail per state. Across this run it records 1,447 accepted KDE valleys, 432 low-support adjacent-segment merges, and 780 support-constrained weighted-median split operations. At least one weighted-median split occurred in 241 states.

## 3. Stage-directed Markov node tree

Output: `markov_tree_0914_r05_kde_bw05_span4_min50_nodeknn_k10_rootw10/`

Important parameters:

| Parameter | Value |
|---|---:|
| Node kNN | 10 |
| Mutual-kNN bonus | 1.5 |
| Root-stage window | 1.0 |
| Stage column | `cell_weighted_mean_stage` |
| Distance temperature | lineage median, selected automatically |
| Fate-probability output threshold | 0.0001 |

The lineage-local candidate DAG contains 12,652 directed edges, including 7,989 mutual-kNN edges and 45 nearest-earlier-node fallback edges. All 45 fallback edges were selected into the extracted tree.

The complete extracted tree contains 2,076 temporal nodes, 19 lineage virtual roots, one global virtual root, and 2,095 edges. The candidate DAG has 111 absorbing sinks. Absorbing sinks in the candidate DAG are distinct from terminal leaves in the extracted or contracted tree.

`root_stage_window=1.0` connects every temporal node no more than one stage unit after its lineage minimum directly to the lineage virtual root. Cell type does not participate in root or parent selection.

## 4. Mean-stage path contraction and sibling merging

Output: `markov_tree_0914_r05_kde_bw05_span4_min50_nodeknn_k10_rootw10_contracted_d04_meanstage5_sibling_d04_gap2/`

Important parameters:

| Parameter | Value |
|---|---:|
| Maximum path cosine distance | 0.4 |
| Maximum path node-mean-stage span | 5 |
| Sibling merging | enabled |
| Maximum sibling cosine distance | 0.4 |
| Maximum sibling mean-stage gap | 2.0 |

For path contraction, `max_stage_span=5` means the maximum difference among the source nodes' `cell_weighted_mean_stage` values. It does not constrain the stored `max_stage - min_stage` envelope across all member microcells in the contracted node.

Only nonbranching paths are eligible for path contraction. Structural anchors and anchor-to-anchor edges remain protected. Optional sibling merging then combines eligible siblings that share a real temporal-node parent, satisfy the cosine and mean-stage thresholds, and include at least one leaf. Direct children of lineage virtual roots remain protected. A second path pass cleans up nonbranching paths created by sibling merging.

Contraction reduced 2,076 temporal nodes to 823. The 47 sibling merges simplified the topology from 200 to 161 branch nodes and from 373 to 326 terminal nodes while preserving 126 early-root temporal nodes. A final contracted node contains at most 18 source nodes; the median is two.

`source_to_contracted_nodes.parquet` records the exact source-to-contracted mapping, and `sibling_merge_history.parquet` records every sibling merge. Markov probabilities were not recomputed after contraction; the output represents the contracted tree topology.

The configured path mean-stage span applies only to path contraction. Sibling merging uses its separate mean-stage-gap criterion, so the final source-node mean-stage span of a sibling-merged group can exceed 5.

## 5. Cell-type purity

Cell-type labels are used only for post-hoc evaluation. The contracted-node purity report is stored under:

```text
markov_tree_0914_r05_kde_bw05_span4_min50_nodeknn_k10_rootw10_contracted_d04_meanstage5_sibling_d04_gap2/
└── purity_evaluation/
```

| Quantity | Value |
|---|---:|
| Evaluated temporal nodes | 823 |
| Evaluated cells | 15,203,034 |
| Mean node purity | 0.7379 |
| Median node purity | 0.7762 |
| Cell-weighted node purity | 0.6857 |
| Annotation coverage | 1.0000 |

Purity is defined as the dominant known-celltype count divided by the number of known annotated cells in each node. All cells have matched known annotations in this evaluation.

## Principal audit files

- `run_config.json`: exact arguments and input paths for each workflow stage;
- `summary.json`: stage-level counts and validation summaries;
- `temporal_split_diagnostics.parquet`: KDE valleys, low-support merges, median splits, and final segments per Leiden state;
- `candidate_markov_edges.parquet`: all stage-directed candidate edges and selected-tree flags;
- `tree_nodes.parquet` and `tree_edges.parquet`: extracted or contracted topology;
- `source_to_contracted_nodes.parquet`: source-to-contracted node mapping;
- `sibling_merge_history.parquet`: ordered sibling-merge history;
- `node_celltype_composition.parquet`: post-hoc node composition.

## Interpretation notes

- Known lineage is a hard stratum throughout the workflow.
- Cell type is annotation and evaluation only; it is never a grouping, temporal-splitting, edge-selection, or merge input.
- Microcell construction still uses fixed-width stage strata; adaptive KDE subdivision is applied after lineage-local Leiden clustering.
- The KDE maximum-span split is support constrained. Sparse broad segments can remain wider than the configured threshold when no feasible split preserves both minimum cells and minimum metacells.
- Path contraction uses node-level cell-weighted mean-stage span. Final member-microcell stage envelopes can be wider.
- Sibling merging intentionally changes eligible local topology and is fully audited.
- Paths in `run_config.json` are provenance fields and record the locations used when each stage was executed.
