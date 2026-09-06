# Lineage- and Stage-Constrained Landmark Tree Construction

This repository implements a scalable workflow for representing large single-cell embedding datasets with landmark nodes and constructing temporally ordered developmental trees. The method preserves lineage identity and developmental-stage structure while reducing millions of cells to a compact, interpretable representation.

## Method

The workflow comprises four stages:

1. **Lineage-stage stratification.** Cells are partitioned into non-overlapping strata defined by lineage and fixed-width developmental-stage intervals.
2. **Microcell construction.** Within each stratum, two-pass locally distance-constrained region growing on cosine k-nearest-neighbor graphs produces microcells. Group sizes can vary; small groups are retained when no safe merge is available. The legacy balanced partition remains available with `--microcell-method balanced`. Each microcell is represented by the mean embedding of its member cells.
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

The historical reference used the legacy balanced partition. Under this configuration, 217 eligible lineage-stage strata produced 1,439 landmark nodes. Fourteen sparsely supported strata comprising 329 cells were excluded from landmark allocation.

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

This command only constructs microcells and aggregates their annotations. It
has no landmark-count or allocation parameters. `--target-nodes`,
`--min-nodes-per-stratum`, `--quota-power`, `--allocation-knn`, and
`--allocation-density-power` are no longer accepted by this command.
`--regroup-knn` remains because it forms coherent piles for the second
microcell pass.

Its outputs are `microcells.parquet`, `microcell_embeddings.npy`,
`cell_to_microcell.npy`, `microcell_celltype_composition.parquet`,
`microcell_grouping_summary.json`, and `run_config.json`. Node tables and
cell-to-node mappings are created only by the separate remerge commands.
The script filename is retained for compatibility with existing paths.

### Use model-predicted stage instead of original stage

Pass `--predicted-stage /mnt/input/sc_cz/Concord/eval/2026_09_03/predicted_stage.csv`
to the microcell construction command. The CSV must contain `idx`, `cell_id`,
and `predicted_stage`. It may be shuffled: `idx` identifies the embedding/CSR
row and its `cell_id` must match exactly. Duplicate indices, missing cells,
non-finite predictions and ID mismatches raise errors. The complete input CSV
is validated even when `--max-cells` selects a test prefix.

In this mode neither `stage_ids.npy` nor the original stage values in CSR
metadata are used. The CSV columns `true_stage` and `stage_id` are ignored.
Predictions determine lineage-time bins and every `mean_stage`, `stage_std`,
`min_stage` and `max_stage` summary. Values are used without clipping or rank
normalization; `--stage-bin-width` is measured in predicted-stage units.

Run landmark remerging on these newly generated microcells. The tree command
continues to use `--stage-column mean_stage`, now the mean predicted time.
`stage_source` in node/microcell tables and run configuration records the time
source; tree summaries record both the selected column and available sources.
Omitting `--predicted-stage` preserves original-stage behavior.

These predictions provide a supervised model time coordinate, not an
independently inferred unsupervised pseudotime. Changing the coordinate does
not itself guarantee higher cell-type purity. Rebuild microcells to change
hard time-bin membership; relabeling an existing tree cannot do this.

### Microcell grouping options

The default `--microcell-method cohesive` uses embeddings only for grouping.
Cell-type annotations are computed after assignments are complete and do not
influence group growth or merging. Existing lineage and stage stratification is
retained. Landmark allocation and merging are unchanged.

- `--microcell-size`: preferred growth size (default 500).
- `--microcell-min-size`: attempt safe merges below this size (default 200); this
  is a soft minimum, so isolated or incompatible small groups are retained.
- `--microcell-max-size`: hard maximum for cohesive groups (default 1000).
- `--microcell-radius-factor`: local cosine-distance multiplier (default 2).
  Smaller values are stricter and may produce many small groups.
- `--microcell-mutual-knn`: optionally require reciprocal neighbors; off by
  default because it can increase fragmentation.

Local distance scales use the median of up to five nearest non-self neighbors.
Growth checks distance to both the seed and current group center. Small-group
merges require graph adjacency, the size limit, and acceptable distances of
all merged members to the new center. Cells are never assigned merely to fill
capacity. These geometric checks do not guarantee cell-type purity.

For an initial smaller-group experiment, use `--microcell-size 200
--microcell-min-size 50 --microcell-max-size 400`. Compare against `balanced`
with the same input, seed and size parameters. Inspect `microcells.parquet`
(including `below_min_size`) and `microcell_grouping_summary.json` for small-group
and singleton counts, size distribution, and cell coverage alongside post-hoc
purity. Landmark-node purity is a separate measurement because subsequent
merging can combine multiple microcells. The historical 1,439-node result above
has not been reproduced with the new method.

Run synthetic regression checks with:

```bash
python -m unittest discover -s tests -v
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

## Evaluate microcell cell-type purity

After constructing microcells, run:

```bash
conda activate train
python src/utilities/evaluate_microcell_purity.py \
  --input-dir ./microcells_0903_predicted_stage
```

The script reads `microcells.parquet` and
`microcell_celltype_composition.parquet` without modifying assignments. Results
are saved to `purity_evaluation/` inside the input directory (override with
`--output-dir`). Existing nonempty output directories require `--overwrite`.

Outputs include `summary.json`, `per_microcell.csv`, `by_lineage.csv`, and
`by_dominant_celltype.csv`, plus PNG/PDF figures for purity distribution,
purity versus size, lineage comparisons, threshold coverage, group sizes and
annotation coverage. Lineage plots are paginated at 30 lineages per figure.

Primary purity is the dominant **known** cell-type count divided by the number
of known annotated cells. Blank, `Unknown`, `Unkown`, and `Unassigned` labels are excluded
(case insensitive); configure other placeholders with `--unknown-labels`.
Groups without known annotations have undefined purity and are excluded from
purity means, while remaining in coverage and size statistics. The original
convention including unknown labels is also saved per group as
`purity_including_unknown`.

`known_cell_weighted_purity` weights by known annotated cell counts.
`dominant_fraction_all_cells` uses all member cells as the denominator, including
unknown and unmatched cells. Threshold coverage measures the cells belonging to
passing groups, not classification accuracy. Dominant-celltype summaries are
not per-celltype recall. All microcells in the table are evaluated, including
those excluded from subsequent landmark allocation.

Compare purity together with annotation coverage, size distributions and small
group frequency (`--small-size`, default 50). No known annotations produces null
summary purity rather than a misleading zero or perfect score.

When all cells have known annotations, plots automatically omit the annotation
coverage panel and show only one purity-coverage curve. Plot labels use
"Cell-type purity" and "Cells". Unknown/unmatched counts remain in the CSV/JSON
checks; incomplete annotations automatically restore the panel and second
curve. `summary.json` includes `annotation_complete`. The filename
`size_and_annotations` is retained in both modes so `--overwrite` replaces old
plots rather than leaving an obsolete annotation figure behind.

## Evaluate merged landmark purity

```bash
conda activate train
python src/utilities/evaluate_landmark_purity.py \
  --input-dir ./landmarks_0903_predicted_stage \
  --dpi 240
```

This independent entry point reads `nodes.parquet` and
`node_celltype_composition.parquet`. It recomputes purity from cell-type counts,
not averages of microcell purities. It shares the microcell evaluator's metrics,
PNG/PDF plots, unknown-label checks and automatic plot simplification.

Outputs go to `landmarks_0903_predicted_stage/purity_evaluation/` by default:
`per_node.csv` (with `node_id`), `summary.json` (with `n_nodes` and
`evaluable_nodes`), `by_lineage.csv`, `by_dominant_celltype.csv`, and the same
five plot categories with landmark-node labels. `--output-dir`, `--small-size`,
`--unknown-labels`, and `--overwrite` work as in the microcell evaluator.

Coverage uses cells represented by the included landmark nodes as its
denominator. Cells from strata excluded during landmark allocation are not
included; account for this when comparing against the pre-merge evaluation.

## Recover a UMAP H5AD export

If UMAP coordinates and plots were saved but the final H5AD export failed,
rebuild the H5AD without recomputing UMAP:

```bash
python src/visualization/plot_microcell_umap_scanpy.py \
  --input-dir ./microcells_0903_predicted_stage \
  --restore-h5ad
```

This validates saved coordinate IDs, sizes, times and lineages against the
microcell inputs, then restores embedding, annotations, UMAP coordinates and
saved categorical palettes. The restored H5AD does not contain the original
neighbor graph or UMAP parameter metadata; this is recorded in
`uns['umap_recovery']`. Existing PNG/PDF files are untouched. All H5AD writes
now use a temporary file and replace the target only after successful export.
The string observation index is named `obs_id`, distinct from integer
`microcell_id`, to avoid AnnData's duplicate-name serialization conflict.
