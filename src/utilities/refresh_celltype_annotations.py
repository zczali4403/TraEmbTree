#!/usr/bin/env python3
"""Refresh one lineage's aggregate celltype annotations without reclustering."""
import argparse,math,shutil
from pathlib import Path
import numpy as np,pandas as pd

def parse():
 p=argparse.ArgumentParser()
 p.add_argument('--input-dir',type=Path,action='append',required=True)
 p.add_argument('--metadata',type=Path,required=True)
 p.add_argument('--lineage',default='Liver')
 p.add_argument('--csr-dir',type=Path,default=Path('/scratch/amlt_code/traemb_csr_0811'))
 p.add_argument('--chunk-size',type=int,default=500000)
 p.add_argument('--backup-tag',default='before_celltype_refresh')
 return p.parse_args()

def aggregate(assign_path,rows,cts,id_col):
 a=np.load(assign_path,mmap_mode='r');ids=np.asarray(a[rows],np.int32)
 z=pd.DataFrame({id_col:ids,'celltype':cts}).groupby([id_col,'celltype'],sort=False).size().rename('cell_count').reset_index()
 totals=z.groupby(id_col).cell_count.transform('sum');z['fraction']=z.cell_count/totals
 z=z.sort_values([id_col,'cell_count','celltype'],ascending=[True,False,True]);z['rank']=z.groupby(id_col).cumcount()+1
 records=[]
 for idx,g in z.groupby(id_col,sort=False):
  total=int(g.cell_count.sum());p=g.cell_count.to_numpy()/total
  records.append({id_col:int(idx),'dominant_celltype':str(g.iloc[0].celltype),'celltype_purity':float(g.iloc[0].fraction),'dominant_celltype_count':int(g.iloc[0].cell_count),'second_celltype':str(g.iloc[1].celltype) if len(g)>1 else '','celltype_entropy':float(-(p*np.log(p)).sum()),'celltype_types':len(g),'celltype_matched_cells':total})
 return z,pd.DataFrame(records)

def main():
 a=parse();parts=[]
 for df in pd.read_csv(a.metadata,usecols=['cell','celltype','lineage_final'],chunksize=a.chunk_size):
  z=df[df.lineage_final.astype(str)==a.lineage][['cell','celltype']]
  if len(z):parts.append(z)
 target=pd.concat(parts,ignore_index=True);print('metadata target rows',len(target),flush=True)
 cell_ids=np.load(a.csr_dir/'cell_ids.npy',mmap_mode='r');n=len(cell_ids);h=np.empty(n,np.uint64)
 for lo in range(0,n,a.chunk_size):
  hi=min(n,lo+a.chunk_size);h[lo:hi]=pd.util.hash_array(np.asarray(cell_ids[lo:hi],dtype=object),categorize=False).astype(np.uint64)
  if hi%1000000<a.chunk_size or hi==n:print('hashed',hi,'/',n,flush=True)
 order=np.argsort(h);hs=h[order];th=pd.util.hash_array(target.cell.astype(str).to_numpy(dtype=object),categorize=False).astype(np.uint64);pos=np.searchsorted(hs,th);ok=(pos<n)&(hs[np.minimum(pos,n-1)]==th)
 if int(ok.sum())!=len(target):raise RuntimeError(f'matched {ok.sum()}/{len(target)}')
 rows=order[pos];cts=target.celltype.fillna('Unknown').astype(str).to_numpy()
 for root in a.input_dir:
  root=root.resolve();specs=[('nodes.parquet','node_celltype_composition.parquet','cell_to_node.npy','node_id')]
  if (root/'cell_to_microcell.npy').exists():specs.append(('microcells.parquet','microcell_celltype_composition.parquet','cell_to_microcell.npy','microcell_id'))
  for table_name,comp_name,assign_name,id_col in specs:
   tp=root/table_name;cp=root/comp_name;table=pd.read_parquet(tp);comp=pd.read_parquet(cp);target_ids=table.loc[table.lineage.astype(str)==a.lineage,id_col]
   newcomp,summary=aggregate(root/assign_name,rows,cts,id_col)
   for path in (tp,cp):
    backup=path.with_name(path.stem+'.'+a.backup_tag+path.suffix)
    if not backup.exists():shutil.copy2(path,backup)
   comp=pd.concat([comp[~comp[id_col].isin(target_ids)],newcomp],ignore_index=True).sort_values([id_col,'rank'])
   sm=summary.set_index(id_col);idx=table[id_col].isin(target_ids);ids=table.loc[idx,id_col]
   for col in ['dominant_celltype','celltype_purity','dominant_celltype_count','second_celltype','celltype_entropy','celltype_types','celltype_matched_cells']:
    if col in table.columns or col in sm.columns:table.loc[idx,col]=sm.loc[ids,col].to_numpy()
   table.to_parquet(tp,index=False);comp.to_parquet(cp,index=False)
   print(root.name,table_name,table.loc[idx,'dominant_celltype'].value_counts().to_dict(),flush=True)
if __name__=='__main__':main()
