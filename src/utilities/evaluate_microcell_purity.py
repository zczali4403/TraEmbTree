#!/usr/bin/env python3
"""Post-hoc microcell purity evaluation; never changes clustering assignments."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def evaluate(micro, composition, unknown_labels=('unknown', 'unkown', 'unassigned', '')):
    required = {'microcell_id', 'n_cells', 'lineage'}
    if not required.issubset(micro):
        raise ValueError(f'microcells table requires {sorted(required)}')
    required = {'microcell_id', 'celltype', 'cell_count'}
    if not required.issubset(composition):
        raise ValueError(f'composition table requires {sorted(required)}')
    m = micro[['microcell_id', 'n_cells', 'lineage']].copy().set_index('microcell_id')
    if m.empty or m.index.has_duplicates or m.index.isna().any():
        raise ValueError('microcell IDs must be unique, nonmissing and nonempty')
    c = composition.copy()
    for values in (m.n_cells, c.cell_count):
        numbers = pd.to_numeric(values, errors='raise').to_numpy(dtype=float)
        if not np.isfinite(numbers).all() or np.any(numbers <= 0) or np.any(numbers != np.floor(numbers)):
            raise ValueError('cell counts must be finite positive integers')
    m['n_cells'] = m.n_cells.astype('int64')
    c['cell_count'] = c.cell_count.astype('int64')
    if not c.microcell_id.isin(m.index).all():
        raise ValueError('composition contains unknown microcell IDs')
    c['celltype'] = c.celltype.fillna('').astype(str).str.strip()
    c = c.groupby(['microcell_id', 'celltype'], as_index=False).cell_count.sum()
    unknown = {str(x).strip().casefold() for x in unknown_labels} | {''}
    c['is_unknown'] = c.celltype.str.casefold().isin(unknown)
    known = c[~c.is_unknown].copy()
    m['matched_cells'] = c.groupby('microcell_id').cell_count.sum().reindex(m.index, fill_value=0)
    if (m.matched_cells > m.n_cells).any():
        raise ValueError('annotated counts exceed n_cells; check duplicate annotations')
    m['known_cells'] = known.groupby('microcell_id').cell_count.sum().reindex(m.index, fill_value=0)
    m['unknown_cells'] = m.matched_cells - m.known_cells
    m['unmatched_cells'] = m.n_cells - m.matched_cells
    m['known_fraction'] = m.known_cells / m.n_cells
    m['purity_including_unknown'] = c.groupby('microcell_id').cell_count.max().reindex(m.index) / m.matched_cells.replace(0, np.nan)
    dominant = known.sort_values(['microcell_id', 'cell_count', 'celltype'], ascending=[True, False, True]).drop_duplicates('microcell_id').set_index('microcell_id')
    m['dominant_known_celltype'] = dominant.celltype.reindex(m.index).fillna('(no known annotation)')
    m['dominant_known_cells'] = dominant.cell_count.reindex(m.index, fill_value=0)
    m['purity'] = m.dominant_known_cells / m.known_cells.replace(0, np.nan)
    m['known_celltype_count'] = known.groupby('microcell_id').size().reindex(m.index, fill_value=0)
    totals = known.groupby('microcell_id').cell_count.transform('sum')
    fractions = known.cell_count / totals
    known['entropy_part'] = -fractions * np.log(fractions)
    m['celltype_entropy'] = known.groupby('microcell_id').entropy_part.sum().reindex(m.index)
    # Denominator is all member cells: this penalizes unknown/unmatched labels.
    m['dominant_fraction_all_cells'] = m.dominant_known_cells / m.n_cells
    return m.reset_index()


def summary(frame, small_size):
    v = frame[frame.known_cells > 0]
    total = int(frame.n_cells.sum())
    known = int(frame.known_cells.sum())
    result = dict(n_microcells=len(frame), n_cells=total, evaluable_microcells=len(v),
                  matched_cells=int(frame.matched_cells.sum()), known_cells=known,
                  unknown_cells=int(frame.unknown_cells.sum()), unmatched_cells=int(frame.unmatched_cells.sum()),
                  matched_fraction=float(frame.matched_cells.sum()/total), known_fraction=known/total,
                  mean_purity=float(v.purity.mean()) if len(v) else None,
                  median_purity=float(v.purity.median()) if len(v) else None,
                  p10_purity=float(v.purity.quantile(.1)) if len(v) else None,
                  known_cell_weighted_purity=float(v.dominant_known_cells.sum()/known) if known else None,
                  dominant_fraction_all_cells=float(frame.dominant_known_cells.sum()/total),
                  mean_entropy=float(v.celltype_entropy.mean()) if len(v) else None,
                  median_group_size=float(frame.n_cells.median()),
                  singleton_fraction=float(frame.n_cells.eq(1).mean()),
                  small_size_threshold=small_size,
                  small_group_fraction=float(frame.n_cells.lt(small_size).mean()),
                  small_group_cell_fraction=float(frame.loc[frame.n_cells.lt(small_size), 'n_cells'].sum()/total))
    for threshold in (.9, .95):
        high = v[v.purity >= threshold]
        key = f'purity_ge_{threshold:g}'
        result[key + '_evaluable_group_fraction'] = len(high)/len(v) if len(v) else None
        # This is coverage by groups, not the fraction of correctly classified cells.
        result[key + '_all_cell_coverage'] = float(high.n_cells.sum()/total)
        result[key + '_known_cell_coverage'] = float(high.known_cells.sum()/known) if known else None
    return result


def grouped_summary(frame, column, small_size):
    return pd.DataFrame([{column: str(key), **summary(group, small_size)}
                         for key, group in frame.groupby(column, dropna=False, sort=True)])


def plots(frame, by_lineage, output, dpi, unit='microcell'):
    plt.rcParams.update({'pdf.fonttype': 42, 'font.size': 10})
    valid = frame[frame.known_cells > 0]
    incomplete = bool(frame.unknown_cells.sum() or frame.unmatched_cells.sum())
    purity_label = 'Purity among known annotations' if incomplete else 'Cell-type purity'
    cell_label = 'Known cells' if incomplete else 'Cells'
    weighted_label = 'Weighted by known cells' if incomplete else 'Weighted by cells'

    def save(fig, name):
        fig.tight_layout()
        for extension in ('png', 'pdf'):
            fig.savefig(output / f'{name}.{extension}', dpi=dpi, bbox_inches='tight')
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(0, 1, 31)
    axes[0].hist(valid.purity, bins=bins, color='#4477AA')
    axes[0].set(xlabel=purity_label, ylabel=unit.capitalize()+'s', title=unit.capitalize()+' purity')
    axes[1].hist(valid.purity, bins=bins, weights=valid.known_cells, color='#228833')
    axes[1].set(xlabel=purity_label, ylabel=cell_label, title=weighted_label)
    save(fig, 'purity_distribution')

    fig, ax = plt.subplots(figsize=(7, 5))
    if len(valid):
        h = ax.hexbin(valid.n_cells, valid.purity, xscale='log', gridsize=45, mincnt=1, bins='log', cmap='viridis')
        fig.colorbar(h, ax=ax, label=unit.capitalize()+'s per bin')
    ax.set(xlabel=f'Cells per {unit} (log scale)', ylabel=purity_label, ylim=(-.02, 1.02))
    save(fig, 'purity_vs_size')

    # Paginate instead of silently omitting lineages in large datasets.
    ordered = by_lineage.sort_values('known_cell_weighted_purity', na_position='first')
    for page, lo in enumerate(range(0, len(ordered), 30), 1):
        part = ordered.iloc[lo:lo+30]
        fig, ax = plt.subplots(figsize=(10, max(4, .32*len(part)+1)))
        y = np.arange(len(part))
        ax.barh(y-.18, pd.to_numeric(part.mean_purity, errors="coerce"), height=.35, label=f'Mean per {unit}')
        ax.barh(y+.18, pd.to_numeric(part.known_cell_weighted_purity, errors="coerce"), height=.35, label=weighted_label)
        for row, value in enumerate(part.mean_purity):
            if pd.isna(value):
                ax.text(.02, row, "No known annotations", va="center", color="gray")
        ax.set_yticks(y, part.lineage)
        ax.set(xlim=(0, 1.05), xlabel=purity_label, title='Purity by lineage')
        ax.legend(loc='lower right')
        save(fig, f'purity_by_lineage_{page}')

    thresholds = np.linspace(0, 1, 101)
    total = frame.n_cells.sum()
    coverage = [valid.loc[valid.purity >= t, 'n_cells'].sum()/total for t in thresholds]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thresholds, coverage, label='Threshold on known-label purity' if incomplete else 'Cell coverage')
    if incomplete:
        conservative = [frame.loc[frame.dominant_fraction_all_cells >= t, 'n_cells'].sum()/total for t in thresholds]
        ax.plot(thresholds, conservative, label='Threshold on dominant / all cells', linestyle='--')
    ax.set(xlabel='Threshold', ylabel='All-cell coverage by passing groups', ylim=(0, 1.02), xlim=(0, 1))
    if incomplete:
        ax.legend()
    save(fig, 'purity_coverage')

    fig, axes = plt.subplots(1, 2 if incomplete else 1, figsize=(11, 4) if incomplete else (7, 4), squeeze=False)
    size_ax = axes[0, 0]
    size_ax.hist(np.log10(frame.n_cells), bins=35, color='#4477AA')
    size_ax.set(xlabel=f'log10(cells per {unit})', ylabel=unit.capitalize()+'s', title='Group size distribution')
    if incomplete:
        counts = [frame.known_cells.sum(), frame.unknown_cells.sum(), frame.unmatched_cells.sum()]
        axes[0, 1].bar(['Known', 'Unknown', 'Unmatched'], np.asarray(counts)/total, color=['#228833', '#CCBB44', '#CC6677'])
        axes[0, 1].set(ylabel='Fraction of all cells', ylim=(0, 1), title='Annotation coverage')
    # Keep the filename stable so --overwrite also replaces older two-panel figures.
    save(fig, 'size_and_annotations')


def main(kind='microcell'):
    if kind not in ('microcell', 'node'):
        raise ValueError('kind must be microcell or node')
    is_node = kind == 'node'
    id_column = 'node_id' if is_node else 'microcell_id'
    table_name = 'nodes.parquet' if is_node else 'microcells.parquet'
    composition_name = 'node_celltype_composition.parquet' if is_node else 'microcell_celltype_composition.parquet'
    count_column = 'n_nodes' if is_node else 'n_microcells'
    unit = 'trajectory node' if is_node else 'microcell'
    parser = argparse.ArgumentParser(description=f'Post-hoc {unit} purity evaluation and plots.')
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--unknown-labels', nargs='+', default=['Unknown', 'Unkown', 'Unassigned'], help='Case-insensitive labels excluded from primary purity. Blank labels are always excluded.')
    parser.add_argument('--small-size', type=int, default=50)
    parser.add_argument('--dpi', type=int, default=180)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.small_size < 1 or args.dpi < 1:
        parser.error('--small-size and --dpi must be positive')
    output = args.output_dir or args.input_dir / 'purity_evaluation'
    if output.resolve() == args.input_dir.resolve():
        parser.error('output-dir must differ from input-dir')
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        parser.error('output directory is nonempty; choose another or use --overwrite')
    table_path = args.input_dir / table_name
    if is_node and not table_path.exists():
        contracted_path = args.input_dir / "tree_nodes.parquet"
        if contracted_path.exists():
            table_path = contracted_path
            table_name = contracted_path.name
        else:
            raise FileNotFoundError(
                f"Neither nodes.parquet nor tree_nodes.parquet exists in {args.input_dir}")
    objects = pd.read_parquet(table_path)
    if is_node and "node_type" in objects:
        objects = objects[objects.node_type.eq("temporal_node")].copy()
    composition = pd.read_parquet(args.input_dir / composition_name)
    if id_column not in objects or id_column not in composition:
        raise ValueError(f'{table_name} and {composition_name} must contain {id_column}')
    # Adapt IDs internally; exported node tables always retain node_id.
    frame = evaluate(objects.rename(columns={id_column: 'microcell_id'}),
                     composition.rename(columns={id_column: 'microcell_id'}), args.unknown_labels)
    report = summary(frame, args.small_size)
    report['annotation_complete'] = report['unknown_cells'] == 0 and report['unmatched_cells'] == 0
    report.update(input_dir=str(args.input_dir.resolve()), unknown_labels=args.unknown_labels,
                  purity_definition='dominant known celltype count / known annotated cells',
                  object_type=kind,
                  coverage_definition=f'all cells belonging to passing {unit}s / all cells in {table_name}',
                  evaluation_scope=('All trajectory nodes in nodes.parquet'
                                    if is_node else 'All microcells in microcells.parquet'))
    renames = {'n_microcells': 'n_nodes', 'evaluable_microcells': 'evaluable_nodes'} if is_node else {}
    report = {renames.get(key, key): value for key, value in report.items()}
    output.mkdir(parents=True, exist_ok=True)
    frame.rename(columns={'microcell_id': id_column}).to_csv(output / f'per_{kind}.csv', index=False)
    by_lineage = grouped_summary(frame, 'lineage', args.small_size)
    by_lineage.rename(columns=renames).to_csv(output / 'by_lineage.csv', index=False)
    grouped_summary(frame, 'dominant_known_celltype', args.small_size).rename(columns=renames).to_csv(output / 'by_dominant_celltype.csv', index=False)
    # These type groups describe dominant labels, not per-celltype recall.
    (output / 'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    plots(frame, by_lineage, output, args.dpi, unit=unit)
    for name in (count_column, 'n_cells', 'known_fraction', 'mean_purity', 'median_purity',
                 'known_cell_weighted_purity', 'dominant_fraction_all_cells', 'median_group_size', 'singleton_fraction'):
        print(f'{name}: {report[name]}')
    print(f'Results: {output.resolve()}')


if __name__ == '__main__':
    main()
