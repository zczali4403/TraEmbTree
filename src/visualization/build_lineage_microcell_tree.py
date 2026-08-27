#!/usr/bin/env python3
# Example:
#   /home/aiscuser/.conda/envs/train/bin/python \
#     /mnt/input/sc_cz/Concord/eval/2026_08_19/build_lineage_microcell_tree.py \
#     --lineage Heart \
#     --output-dir /mnt/input/sc_cz/Concord/eval/2026_08_19/heart_microcell_tree \
#     --overwrite
"""Build a stage-monotone cosine or Euclidean tree for one lineage microcells."""
from __future__ import annotations
import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "lineage"

def parse_args():
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--lineage',required=True,help='Exact lineage name, e.g. Heart')
    p.add_argument('--microcells',type=Path,default=HERE/'landmark_nodes_mc2/microcells.parquet')
    p.add_argument('--embeddings',type=Path,default=HERE/'landmark_nodes_mc2/microcell_embeddings.npy')
    p.add_argument('--build-script',type=Path,default=HERE.parent/'workflow'/'build_landmark_tree.py')
    p.add_argument('--output-dir',type=Path,default=None)
    p.add_argument('--stage-column',default='mean_stage')
    p.add_argument('--metric',choices=('cosine','euclidean'),default='cosine')
    p.add_argument('--figure-width',type=float,default=30.0)
    p.add_argument('--figure-height',type=float,default=18.0)
    p.add_argument('--dpi',type=int,default=220)
    p.add_argument('--label-nodes',action='store_true')
    p.add_argument('--overwrite',action='store_true')
    return p.parse_args()

def main():
    a=parse_args()
    output=(a.output_dir or HERE/f'{safe_name(a.lineage)}_microcell_tree').resolve()
    cells=pd.read_parquet(a.microcells)
    required={'microcell_id','lineage','n_cells','dominant_celltype',a.stage_column}
    missing=required-set(cells.columns)
    if missing:raise ValueError(f'microcells table missing columns: {sorted(missing)}')
    available=sorted(cells['lineage'].fillna('').astype(str).unique())
    subset=cells[cells['lineage'].astype(str)==a.lineage].copy().sort_values('microcell_id').reset_index(drop=True)
    if subset.empty:raise ValueError(f'lineage {a.lineage!r} not found; available: {available}')
    source_ids=subset['microcell_id'].to_numpy(dtype=np.int64)
    source_emb=np.load(a.embeddings,mmap_mode='r')
    if source_emb.ndim!=2 or source_ids.min()<0 or source_ids.max()>=len(source_emb):
        raise ValueError(f'microcell IDs are incompatible with embedding shape {source_emb.shape}')
    subset.insert(0,'original_microcell_id',source_ids)
    subset=subset.drop(columns=['microcell_id'])
    subset.insert(0,'node_id',np.arange(len(subset),dtype=np.int32))
    with tempfile.TemporaryDirectory(prefix=f'{safe_name(a.lineage)}_microtree_') as td:
        tmp=Path(td); nodes=tmp/'nodes.parquet'; emb=tmp/'node_embeddings.npy'
        subset.to_parquet(nodes,index=False)
        np.save(emb,np.asarray(source_emb[source_ids],dtype=np.float32))
        cmd=[sys.executable,str(a.build_script),'--nodes',str(nodes),'--embeddings',str(emb),
             '--output-dir',str(output),'--stage-column',a.stage_column,'--metric',a.metric,
             '--figure-width',str(a.figure_width),'--figure-height',str(a.figure_height),'--dpi',str(a.dpi)]
        if a.label_nodes:cmd.append('--label-nodes')
        if a.overwrite:cmd.append('--overwrite')
        env=os.environ.copy()
        for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env.setdefault(key,'32')
        print(f'Building {a.lineage}: {len(subset):,} microcells; metric={a.metric} -> {output}',flush=True)
        subprocess.run(cmd,check=True,env=env)
if __name__=='__main__':main()
