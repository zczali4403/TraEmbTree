#!/usr/bin/env python3
# Example:
#   OPENBLAS_NUM_THREADS=32 OMP_NUM_THREADS=32 \
#   /home/aiscuser/.conda/envs/train/bin/python plot_microcell_umap_scanpy.py \
#     --input-dir /mnt/input/sc_cz/Concord/eval/2026_08_19/landmark_nodes_mc2
"""Compute and plot microcell UMAP using Scanpy."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(key,'32')
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

def parse_args():
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--input-dir',type=Path,required=True)
    p.add_argument('--n-neighbors',type=int,default=30)
    p.add_argument('--min-dist',type=float,default=0.25)
    p.add_argument('--spread',type=float,default=1.0)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--dpi',type=int,default=220)
    p.add_argument('--point-size',type=float,default=8.0)
    return p.parse_args()

def discrete_palette(n):
    names=('tab20','tab20b','tab20c','Set1','Set2','Set3','Dark2','Paired','Accent')
    rgb=[];seen=set()
    for name in names:
        cmap=plt.get_cmap(name)
        for value in cmap.colors:
            hx=matplotlib.colors.to_hex(value).upper()
            if hx not in seen and hx not in {'#FFFFFF','#000000','#B0B0B0'}:
                seen.add(hx);rgb.append(tuple(matplotlib.colors.to_rgb(hx)))
    if len(rgb)<n:
        levels=np.linspace(.12,.88,7)
        candidates=np.asarray([(r,g,b) for r in levels for g in levels for b in levels
            if .25<max(r,g,b)-min(r,g,b) and .18<.2126*r+.7152*g+.0722*b<.82])
        chosen=np.asarray(rgb,float)
        while len(rgb)<n:
            distance=((candidates[:,None,:]-chosen[None,:,:])**2).sum(2).min(1)
            i=int(np.argmax(distance));value=tuple(candidates[i]);rgb.append(value)
            chosen=np.vstack((chosen,value));candidates=np.delete(candidates,i,axis=0)
    return [matplotlib.colors.to_hex(x).upper() for x in rgb[:n]]

def save_scanpy_umap(adata,color,path,palette=None,cmap=None,title=None,size=8,dpi=220,figsize=(12,10)):
    fig=sc.pl.umap(adata,color=color,palette=palette,color_map=cmap,title=title or color,
                   size=size,frameon=False,legend_loc='right margin',show=False,
                   return_fig=True)
    fig.set_size_inches(*figsize)
    fig.savefig(path.with_suffix('.png'),dpi=dpi,bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
    plt.close(fig)
    print('Saved',path.with_suffix('.png'),flush=True)

def main():
    a=parse_args();root=a.input_dir.resolve()
    meta=pd.read_parquet(root/'microcells.parquet').sort_values('microcell_id').reset_index(drop=True)
    x=np.load(root/'microcell_embeddings.npy')
    if len(meta)!=len(x):raise ValueError(f'metadata rows {len(meta)} != embeddings {len(x)}')
    obs=meta.copy();obs.index=obs.microcell_id.astype(str)
    obs['stage']=pd.to_numeric(obs['mean_stage'],errors='raise')
    obs['lineage']=pd.Categorical(obs['lineage'].fillna('Unknown').astype(str),categories=sorted(obs['lineage'].fillna('Unknown').astype(str).unique()))
    obs['celltype']=pd.Categorical(obs['dominant_celltype'].fillna('Unknown').astype(str),categories=sorted(obs['dominant_celltype'].fillna('Unknown').astype(str).unique()))
    adata=ad.AnnData(X=np.asarray(x,dtype=np.float32),obs=obs)
    print(f'Computing Scanpy cosine neighbors: {adata.n_obs:,} x {adata.n_vars}; k={a.n_neighbors}',flush=True)
    sc.pp.neighbors(adata,n_neighbors=a.n_neighbors,use_rep='X',metric='cosine',random_state=a.seed)
    print('Computing Scanpy UMAP',flush=True)
    sc.tl.umap(adata,min_dist=a.min_dist,spread=a.spread,random_state=a.seed)
    lineage_palette=discrete_palette(len(adata.obs['lineage'].cat.categories))
    celltype_palette=discrete_palette(len(adata.obs['celltype'].cat.categories))
    if 'Unknown' in adata.obs['lineage'].cat.categories:
        lineage_palette[list(adata.obs['lineage'].cat.categories).index('Unknown')]='#B0B0B0'
    if 'Unknown' in adata.obs['celltype'].cat.categories:
        celltype_palette[list(adata.obs['celltype'].cat.categories).index('Unknown')]='#B0B0B0'
    save_scanpy_umap(adata,'stage',root/'microcell_umap_by_stage',cmap='viridis',title='Microcells — mean stage',size=a.point_size,dpi=a.dpi)
    save_scanpy_umap(adata,'lineage',root/'microcell_umap_by_lineage',palette=lineage_palette,title='Microcells — lineage',size=a.point_size,dpi=a.dpi,figsize=(14,10))
    save_scanpy_umap(adata,'celltype',root/'microcell_umap_by_celltype',palette=celltype_palette,title='Microcells — dominant cell type',size=a.point_size,dpi=a.dpi,figsize=(18,12))
    coords=meta.copy();coords['UMAP1']=adata.obsm['X_umap'][:,0];coords['UMAP2']=adata.obsm['X_umap'][:,1]
    coords.to_parquet(root/'microcell_umap_coordinates.parquet',index=False)
    pd.DataFrame({'lineage':adata.obs['lineage'].cat.categories,'color':lineage_palette}).to_csv(root/'microcell_umap_lineage_colors.csv',index=False)
    pd.DataFrame({'celltype':adata.obs['celltype'].cat.categories,'color':celltype_palette}).to_csv(root/'microcell_umap_celltype_colors.csv',index=False)
    adata.write_h5ad(root/'microcells_scanpy_umap.h5ad',compression='gzip')
    print('Saved coordinates and AnnData in',root,flush=True)
if __name__=='__main__':main()
