#!/usr/bin/env python3
"""Compute and plot two- and three-dimensional microcell UMAPs using Scanpy."""
from __future__ import annotations

import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(key, '32')

import argparse
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import scanpy as sc


DEFAULT_3D_VIEWS = ((25, 0), (25, 90), (25, 180), (25, 270))


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--n-neighbors', type=int, default=30)
    parser.add_argument('--min-dist', type=float, default=0.25)
    parser.add_argument('--spread', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dpi', type=int, default=220)
    parser.add_argument('--point-size', type=float, default=8.0)
    parser.add_argument('--point-size-3d', type=float, default=1.0)
    parser.add_argument('--alpha-3d', type=float, default=0.8)
    parser.add_argument(
        '--restore-h5ad', action='store_true',
        help='Rebuild H5AD from saved 2D and optional 3D coordinates without recomputing neighbors, UMAP, or plots. Neighbor graph is not recovered.')
    return parser.parse_args()


def discrete_palette(n):
    names = ('tab20', 'tab20b', 'tab20c', 'Set1', 'Set2', 'Set3',
             'Dark2', 'Paired', 'Accent')
    rgb, seen = [], set()
    for name in names:
        cmap = plt.get_cmap(name)
        for value in cmap.colors:
            hex_color = matplotlib.colors.to_hex(value).upper()
            if hex_color not in seen and hex_color not in {'#FFFFFF', '#000000', '#B0B0B0'}:
                seen.add(hex_color)
                rgb.append(tuple(matplotlib.colors.to_rgb(hex_color)))
    if len(rgb) < n:
        levels = np.linspace(.12, .88, 7)
        candidates = np.asarray([
            (r, g, b) for r in levels for g in levels for b in levels
            if .25 < max(r, g, b) - min(r, g, b)
            and .18 < .2126 * r + .7152 * g + .0722 * b < .82
        ])
        chosen = np.asarray(rgb, float)
        while len(rgb) < n:
            distance = ((candidates[:, None, :] - chosen[None, :, :]) ** 2).sum(2).min(1)
            index = int(np.argmax(distance))
            value = tuple(candidates[index])
            rgb.append(value)
            chosen = np.vstack((chosen, value))
            candidates = np.delete(candidates, index, axis=0)
    return [matplotlib.colors.to_hex(value).upper() for value in rgb[:n]]


def save_scanpy_umap(adata, color, path, palette=None, cmap=None, title=None,
                      size=8, dpi=220, figsize=(12, 10)):
    figure = sc.pl.umap(
        adata, color=color, palette=palette, color_map=cmap,
        title=title or color, size=size, frameon=False,
        legend_loc='right margin', show=False, return_fig=True)
    figure.set_size_inches(*figsize)
    figure.savefig(path.with_suffix('.png'), dpi=dpi, bbox_inches='tight')
    figure.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(figure)
    print('Saved', path.with_suffix('.png'), flush=True)


def _save_3d_figure(figure, stem, dpi):
    figure.savefig(stem.with_suffix('.png'), dpi=dpi, bbox_inches='tight')
    figure.savefig(stem.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(figure)
    print('Saved', stem.with_suffix('.png'), flush=True)


def save_3d_continuous_views(coordinates, values, output_prefix, label,
                             point_size=1, alpha=.8, dpi=220, seed=42,
                             views=DEFAULT_3D_VIEWS, cmap='viridis'):
    coordinates = np.asarray(coordinates)
    values = np.asarray(values, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError('3D UMAP coordinates must have shape N x 3')
    if len(values) != len(coordinates) or not np.isfinite(coordinates).all() or not np.isfinite(values).all():
        raise ValueError('3D coordinates and continuous colors must be aligned and finite')
    order = np.random.default_rng(seed).permutation(len(coordinates))
    xyz, colors = coordinates[order], values[order]
    for elevation, azimuth in views:
        figure = plt.figure(figsize=(9, 8))
        axis = figure.add_subplot(111, projection='3d')
        points = axis.scatter(
            xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=point_size,
            cmap=cmap, alpha=alpha, linewidths=0, rasterized=True)
        axis.view_init(elev=elevation, azim=azimuth)
        axis.set(xlabel='UMAP1', ylabel='UMAP2', zlabel='UMAP3',
                 title=f'Microcells — {label}')
        figure.colorbar(points, ax=axis, shrink=.65, pad=.1, label=label)
        stem = output_prefix.with_name(
            f'{output_prefix.name}_elev{elevation}_azim{azimuth}')
        _save_3d_figure(figure, stem, dpi)


def save_3d_categorical_views(coordinates, labels, categories, palette,
                              output_prefix, title, point_size=1, alpha=.8,
                              dpi=220, seed=42, views=DEFAULT_3D_VIEWS,
                              figsize=(12, 9), legend_fontsize=7):
    coordinates = np.asarray(coordinates)
    labels = pd.Categorical(labels, categories=categories)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError('3D UMAP coordinates must have shape N x 3')
    if len(labels) != len(coordinates) or not np.isfinite(coordinates).all():
        raise ValueError('3D coordinates and categorical colors must be aligned and finite')
    if len(palette) != len(categories) or (labels.codes < 0).any():
        raise ValueError('Categorical palette does not cover all observations')
    order = np.random.default_rng(seed).permutation(len(coordinates))
    xyz = coordinates[order]
    rgba = np.asarray([matplotlib.colors.to_rgba(color) for color in palette])
    point_colors = rgba[labels.codes[order]]
    handles = [
        Line2D([0], [0], marker='o', linestyle='', markersize=5,
               markerfacecolor=palette[index], markeredgecolor='none', label=str(category))
        for index, category in enumerate(categories)
    ]
    for elevation, azimuth in views:
        figure = plt.figure(figsize=figsize)
        axis = figure.add_subplot(111, projection='3d')
        axis.scatter(
            xyz[:, 0], xyz[:, 1], xyz[:, 2], c=point_colors,
            s=point_size, alpha=alpha, linewidths=0, rasterized=True)
        axis.view_init(elev=elevation, azim=azimuth)
        axis.set(xlabel='UMAP1', ylabel='UMAP2', zlabel='UMAP3', title=title)
        axis.legend(handles=handles, bbox_to_anchor=(1.08, 1), loc='upper left',
                    frameon=False, fontsize=legend_fontsize)
        stem = output_prefix.with_name(
            f'{output_prefix.name}_elev{elevation}_azim{azimuth}')
        _save_3d_figure(figure, stem, dpi)


def save_all_3d_views(adata, root, lineage_palette, celltype_palette,
                      point_size, alpha, dpi, seed):
    coordinates = adata.obsm['X_umap_3d']
    save_3d_continuous_views(
        coordinates, adata.obs['stage'].to_numpy(),
        root / 'microcell_umap_3d_by_stage', 'mean stage',
        point_size=point_size, alpha=alpha, dpi=dpi, seed=seed)
    save_3d_categorical_views(
        coordinates, adata.obs['lineage'], adata.obs['lineage'].cat.categories,
        lineage_palette, root / 'microcell_umap_3d_by_lineage',
        'Microcells — lineage', point_size=point_size, alpha=alpha,
        dpi=dpi, seed=seed, figsize=(14, 10), legend_fontsize=8)
    save_3d_categorical_views(
        coordinates, adata.obs['celltype'], adata.obs['celltype'].cat.categories,
        celltype_palette, root / 'microcell_umap_3d_by_celltype',
        'Microcells — dominant cell type', point_size=point_size, alpha=alpha,
        dpi=dpi, seed=seed, figsize=(18, 12), legend_fontsize=5)


def write_h5ad_atomic(adata, path):
    """Publish only a complete H5AD; leave any previous output intact on failure."""
    temporary = path.with_name(f'.{path.stem}.writing-{os.getpid()}.h5ad')
    try:
        adata.write_h5ad(temporary, compression='gzip')
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validated_coordinates(path, adata, dimensions):
    coordinates = pd.read_parquet(path).sort_values('microcell_id')
    columns = ['microcell_id'] + [f'UMAP{index}' for index in range(1, dimensions + 1)]
    if not set(columns).issubset(coordinates.columns):
        raise ValueError(f'{path.name} must contain {columns}')
    if not np.array_equal(coordinates.microcell_id.to_numpy(), adata.obs.microcell_id.to_numpy()):
        raise ValueError(f'{path.name} and input microcell IDs do not match')
    for column in ('n_cells', 'mean_stage', 'lineage'):
        if column in coordinates and not np.array_equal(
                coordinates[column].to_numpy(), adata.obs[column].to_numpy()):
            raise ValueError(f'{path.name} has different {column}; check input provenance')
    values = coordinates[columns[1:]].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f'{path.name} must contain finite coordinates')
    return values


def restore_h5ad(adata, root):
    coordinate_2d = root / 'microcell_umap_coordinates.parquet'
    adata.obsm['X_umap'] = _validated_coordinates(coordinate_2d, adata, 2)
    coordinate_3d = root / 'microcell_umap_3d_coordinates.parquet'
    has_3d = coordinate_3d.exists()
    if has_3d:
        adata.obsm['X_umap_3d'] = _validated_coordinates(coordinate_3d, adata, 3)
    for column in ('lineage', 'celltype'):
        path = root / f'microcell_umap_{column}_colors.csv'
        if path.exists():
            palette = pd.read_csv(path, keep_default_na=False).set_index(column)['color']
            categories = adata.obs[column].cat.categories
            if not palette.index.is_unique or not categories.isin(palette.index).all():
                raise ValueError(f'saved {column} palette does not match categories')
            adata.uns[f'{column}_colors'] = palette.reindex(categories).to_numpy(dtype=str)
    adata.uns['umap_recovery'] = {
        'coordinate_source_2d': str(coordinate_2d),
        'coordinate_source_3d': str(coordinate_3d) if has_3d else None,
        'three_dimensional_coordinates_available': has_3d,
        'neighbor_graph_available': False,
        'note': 'Restored saved embedding coordinates and annotations; neighbor graph is not recovered.',
    }
    write_h5ad_atomic(adata, root / 'microcells_scanpy_umap.h5ad')
    print(f'Restored H5AD using saved 2D coordinates; 3D available={has_3d}. '
          'Neighbors were not recomputed.', flush=True)


def main():
    args = parse_args()
    root = args.input_dir.resolve()
    metadata = (pd.read_parquet(root / 'microcells.parquet')
                .sort_values('microcell_id').reset_index(drop=True))
    embedding = np.load(root / 'microcell_embeddings.npy')
    if len(metadata) != len(embedding):
        raise ValueError(f'metadata rows {len(metadata)} != embeddings {len(embedding)}')
    if not np.array_equal(metadata.microcell_id.to_numpy(), np.arange(len(metadata))):
        raise ValueError('microcell_id must be contiguous and aligned to embedding rows')
    observation = metadata.copy()
    observation.index = pd.Index(observation.microcell_id.astype(str).to_numpy(), name='obs_id')
    observation['stage'] = pd.to_numeric(observation['mean_stage'], errors='raise')
    observation['lineage'] = pd.Categorical(
        observation['lineage'].fillna('Unknown').astype(str),
        categories=sorted(observation['lineage'].fillna('Unknown').astype(str).unique()))
    observation['celltype'] = pd.Categorical(
        observation['dominant_celltype'].fillna('Unknown').astype(str),
        categories=sorted(observation['dominant_celltype'].fillna('Unknown').astype(str).unique()))
    adata = ad.AnnData(X=np.asarray(embedding, dtype=np.float32), obs=observation)
    if args.restore_h5ad:
        restore_h5ad(adata, root)
        return

    print(f'Computing Scanpy cosine neighbors: {adata.n_obs:,} x {adata.n_vars}; '
          f'k={args.n_neighbors}', flush=True)
    sc.pp.neighbors(
        adata, n_neighbors=args.n_neighbors, use_rep='X', metric='cosine',
        random_state=args.seed)
    print('Computing 2D Scanpy UMAP', flush=True)
    sc.tl.umap(
        adata, n_components=2, min_dist=args.min_dist, spread=args.spread,
        random_state=args.seed)
    print('Computing 3D Scanpy UMAP from the same neighbor graph', flush=True)
    sc.tl.umap(
        adata, n_components=3, min_dist=args.min_dist, spread=args.spread,
        random_state=args.seed, key_added='X_umap_3d')

    lineage_palette = discrete_palette(len(adata.obs['lineage'].cat.categories))
    celltype_palette = discrete_palette(len(adata.obs['celltype'].cat.categories))
    if 'Unknown' in adata.obs['lineage'].cat.categories:
        lineage_palette[list(adata.obs['lineage'].cat.categories).index('Unknown')] = '#B0B0B0'
    if 'Unknown' in adata.obs['celltype'].cat.categories:
        celltype_palette[list(adata.obs['celltype'].cat.categories).index('Unknown')] = '#B0B0B0'

    save_scanpy_umap(
        adata, 'stage', root / 'microcell_umap_by_stage', cmap='viridis',
        title='Microcells — mean stage', size=args.point_size, dpi=args.dpi)
    save_scanpy_umap(
        adata, 'lineage', root / 'microcell_umap_by_lineage', palette=lineage_palette,
        title='Microcells — lineage', size=args.point_size, dpi=args.dpi, figsize=(14, 10))
    save_scanpy_umap(
        adata, 'celltype', root / 'microcell_umap_by_celltype', palette=celltype_palette,
        title='Microcells — dominant cell type', size=args.point_size, dpi=args.dpi,
        figsize=(18, 12))
    save_all_3d_views(
        adata, root, lineage_palette, celltype_palette,
        args.point_size_3d, args.alpha_3d, args.dpi, args.seed)

    coordinates_2d = metadata.copy()
    coordinates_2d['UMAP1'] = adata.obsm['X_umap'][:, 0]
    coordinates_2d['UMAP2'] = adata.obsm['X_umap'][:, 1]
    coordinates_2d.to_parquet(root / 'microcell_umap_coordinates.parquet', index=False)
    coordinates_3d = metadata.copy()
    for index in range(3):
        coordinates_3d[f'UMAP{index + 1}'] = adata.obsm['X_umap_3d'][:, index]
    coordinates_3d.to_parquet(root / 'microcell_umap_3d_coordinates.parquet', index=False)
    np.save(root / 'microcell_umap_3d.npy', adata.obsm['X_umap_3d'])
    pd.DataFrame({
        'lineage': adata.obs['lineage'].cat.categories,
        'color': lineage_palette,
    }).to_csv(root / 'microcell_umap_lineage_colors.csv', index=False)
    pd.DataFrame({
        'celltype': adata.obs['celltype'].cat.categories,
        'color': celltype_palette,
    }).to_csv(root / 'microcell_umap_celltype_colors.csv', index=False)
    write_h5ad_atomic(adata, root / 'microcells_scanpy_umap.h5ad')
    print('Saved 2D/3D coordinates, plots, palettes, and AnnData in', root, flush=True)


if __name__ == '__main__':
    main()
