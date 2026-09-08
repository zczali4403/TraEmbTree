#!/usr/bin/env python3
"""Construct and annotate microcells within lineage x fixed-width time bins.

Two embedding-only partition passes produce microcells and cell assignments.
Cell types are aggregated only after grouping. The resulting metacells feed the
lineage-local Leiden trajectory-node workflow.
"""
from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v,"32")
import argparse, heapq, json, math, shutil, time
from collections import defaultdict, deque
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent

def log(x): print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {x}",flush=True)

def args_parser():
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--embeddings',type=Path,default=Path('/mnt/input/sc_cz/Concord/eval/2026_09_03/embeddings.npy'))
    p.add_argument('--csr-dir',type=Path,default=Path('/scratch/amlt_code/traemb_csr_0829'))
    p.add_argument('--metadata',type=Path,default=Path('/mnt/input/sc_cz/Concord/data/all_lineage_260829.csv'))
    p.add_argument('--output-dir',type=Path,default=HERE/'stagebin_microcells')
    p.add_argument('--predicted-stage',type=Path,default=None,help='CSV with idx, cell_id, predicted_stage; replaces original stage for binning and all stage summaries.')
    p.add_argument('--stage-bin-width',type=float,default=1.0,help='Width of fixed stage bins.')
    p.add_argument('--stage-bin-origin',type=float,default=0.0,help='Fixed origin anchoring stage-bin boundaries.')
    p.add_argument('--pile-size',type=int,default=20000)
    p.add_argument('--microcell-size',type=int,default=500)
    p.add_argument('--microcell-min-size',type=int,default=200)
    p.add_argument('--microcell-max-size',type=int,default=1000)
    p.add_argument('--microcell-method',choices=('cohesive','balanced'),default='cohesive',help='Embedding-only cohesive regions, or legacy balanced groups.')
    p.add_argument('--microcell-radius-factor',type=float,default=2.0,help='Local cosine-distance multiplier for cohesive grouping.')
    p.add_argument('--microcell-mutual-knn',action='store_true',help='Require reciprocal neighbors in cohesive grouping; may increase fragmentation.')
    p.add_argument('--knn',type=int,default=30)
    p.add_argument('--regroup-knn',type=int,default=64,help='Neighbors of microcell centers for forming second-pass piles.')
    p.add_argument('--metadata-chunk-size',type=int,default=500000)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--faiss-threads',type=int,default=32)
    p.add_argument('--max-cells',type=int,default=None,help='Testing only: use first N rows.')
    p.add_argument('--overwrite',action='store_true')
    return p.parse_args()

def normalize(x):
    z=np.asarray(x,dtype=np.float32).copy()
    z/=np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
    return z


def cosine_knn_edges(vectors,k):
    import faiss
    z=normalize(vectors); k=min(k+1,len(z))
    index=faiss.IndexFlatIP(z.shape[1]); index.add(z); sim,nei=index.search(z,k)
    src=np.repeat(np.arange(len(z),dtype=np.int32),k-1); dst=nei[:,1:].reshape(-1).astype(np.int32); score=sim[:,1:].reshape(-1)
    valid=(dst>=0)&(src!=dst)&np.isfinite(score)
    src,dst,score=src[valid],dst[valid],score[valid]
    lo=np.minimum(src,dst).astype(np.int64); hi=np.maximum(src,dst).astype(np.int64)
    key=lo*len(z)+hi; order=np.argsort(key); key=key[order]; keep=np.r_[True,key[1:]!=key[:-1]]
    return src[order][keep],dst[order][keep],score[order][keep]

def balanced_knn_region_partition(vectors,target_groups,k):
    """Balanced multi-source region growing on a cosine kNN graph."""
    n=len(vectors); target_groups=min(max(1,int(target_groups)),n)
    if target_groups==1:return np.zeros(n,dtype=np.int32)
    z=normalize(vectors)
    # Deterministic cosine farthest-point seeds; piles were already randomized.
    seeds=[0]; nearest=z@z[0]
    for _ in range(1,target_groups):
        candidate=int(np.argmin(nearest)); seeds.append(candidate)
        nearest=np.maximum(nearest,z@z[candidate])
    src,dst,score=cosine_knn_edges(z,k); adj=[[] for _ in range(n)]
    for a,b,w in zip(src,dst,score):
        a=int(a);b=int(b);w=float(w);adj[a].append((b,w));adj[b].append((a,w))
    capacities=np.full(target_groups,n//target_groups,dtype=np.int64);capacities[:n%target_groups]+=1
    labels=np.full(n,-1,dtype=np.int32);sizes=np.zeros(target_groups,dtype=np.int64);heap=[]
    for g,node in enumerate(seeds):
        labels[node]=g;sizes[g]=1
    for g,node in enumerate(seeds):
        for other,w in adj[node]:
            if labels[other]<0:heapq.heappush(heap,(-w,g,other))
    while heap:
        neg,g,node=heapq.heappop(heap)
        if labels[node]>=0 or sizes[g]>=capacities[g]:continue
        labels[node]=g;sizes[g]+=1
        for other,w in adj[node]:
            if labels[other]<0:heapq.heappush(heap,(-w,g,other))
    # Disconnected leftovers or regions blocked by capacity: closest seed with space.
    remaining=np.flatnonzero(labels<0)
    for lo in range(0,len(remaining),4096):
        rr=remaining[lo:lo+4096];sim=z[rr]@z[np.asarray(seeds)].T
        for row,node in zip(sim,rr):
            available=np.flatnonzero(sizes<capacities)
            g=int(available[np.argmax(row[available])]);labels[node]=g;sizes[g]+=1
    if np.any(labels<0) or len(np.unique(labels))!=target_groups:
        raise RuntimeError('balanced kNN partition failed to create all requested groups')
    return labels


def split_groups(indices,labels):
    order=np.argsort(labels,kind='stable'); labs=labels[order]; cuts=np.flatnonzero(np.r_[True,labs[1:]!=labs[:-1],True])
    return [indices[order[cuts[i]:cuts[i+1]]] for i in range(len(cuts)-1)]

def raw_means(emb,groups):
    out=np.empty((len(groups),emb.shape[1]),np.float32)
    for i,g in enumerate(groups): out[i]=np.asarray(emb[g],np.float32).mean(axis=0)
    return out

def coherent_piles(groups,centers,pile_size,k):
    if len(groups)==1:return [groups]
    src,dst,score=cosine_knn_edges(centers,min(k,len(groups)-1)); adj=[[] for _ in groups]
    for order_i in np.argsort(-score):
        a,b=int(src[order_i]),int(dst[order_i]); adj[a].append(b); adj[b].append(a)
    unused=set(range(len(groups))); piles=[]
    while unused:
        seed=min(unused); unused.remove(seed); chosen=[seed]; cells=len(groups[seed]); q=deque([seed])
        while q and cells<pile_size:
            a=q.popleft()
            for b in adj[a]:
                if b in unused:
                    unused.remove(b); chosen.append(b); q.append(b); cells+=len(groups[b])
                    if cells>=pile_size: break
        piles.append([groups[i] for i in chosen])
    return piles

def hash_strings(x):
    import pandas as pd
    return pd.util.hash_array(np.asarray(x,dtype=object),categorize=False).astype(np.uint64,copy=False)

def celltype_tables(metadata,cell_ids,assignment,n_microcells,chunk):
    import pandas as pd
    n=len(cell_ids); hashes=np.empty(n,np.uint64)
    for lo in range(0,n,chunk): hashes[lo:min(n,lo+chunk)]=hash_strings(cell_ids[lo:min(n,lo+chunk)])
    order=np.argsort(hashes); hs=hashes[order]; counts=defaultdict(int); matched=rows_seen=0
    for df in pd.read_csv(metadata,usecols=['cell','celltype'],chunksize=chunk):
        rows_seen+=len(df); h=hash_strings(df.cell.astype(str).to_numpy()); pos=np.searchsorted(hs,h); ok=pos<n; ok&=hs[np.minimum(pos,n-1)]==h
        rr=order[pos[ok]]; ct=df.loc[ok,'celltype'].fillna('Unknown').astype(str).to_numpy()
        valid=rr<len(assignment); rr=rr[valid]; ct=ct[valid]
        for micro_id,name in zip(np.asarray(assignment[rr]),ct): counts[(int(micro_id),name)]+=1
        matched+=len(rr); log(f'celltype matched {matched:,}/{rows_seen:,}')
    summary=[]; comp=[]
    by=defaultdict(list)
    for (micro_id,ct),c in counts.items():by[micro_id].append((ct,c))
    for micro_id in range(n_microcells):
        vals=sorted(by[micro_id],key=lambda x:(-x[1],x[0])); total=sum(x[1] for x in vals)
        for rank,(ct,c) in enumerate(vals,1):comp.append(dict(microcell_id=micro_id,celltype=ct,cell_count=c,fraction=c/total,rank=rank))
        dom,dc=vals[0] if vals else ('Unknown',0)
        summary.append(dict(microcell_id=micro_id,dominant_celltype=dom,celltype_purity=dc/total if total else np.nan,dominant_celltype_count=dc,celltype_types=len(vals),celltype_matched_cells=total))
    return summary,comp,matched


def partition_microcells(vectors, args):
    if args.microcell_method == 'balanced':
        count=min(len(vectors),max(1,int(round(len(vectors)/args.microcell_size))))
        if count==len(vectors):return np.arange(count,dtype=np.int32)
        return balanced_knn_region_partition(vectors,count,args.knn)
    from cohesive_microcells import cohesive_partition
    return cohesive_partition(vectors, args.microcell_size, args.microcell_min_size,
                              args.microcell_max_size, k=args.knn,
                              radius_factor=args.microcell_radius_factor,
                              mutual=args.microcell_mutual_knn)


def main():
    a=args_parser(); import faiss, pandas as pd
    if not np.isfinite(a.stage_bin_width) or a.stage_bin_width<=0:raise ValueError('stage-bin-width must be positive and finite')
    if a.microcell_method == 'cohesive':
        if not 1 <= a.microcell_min_size <= a.microcell_size <= a.microcell_max_size:
            raise ValueError('require 1 <= microcell-min-size <= microcell-size <= microcell-max-size')
        if a.knn < 1 or not np.isfinite(a.microcell_radius_factor) or a.microcell_radius_factor <= 0:
            raise ValueError('knn and microcell-radius-factor must be positive and finite')
    for name in ('pile_size','microcell_size','knn','regroup_knn','metadata_chunk_size','faiss_threads'):
        if getattr(a,name)<1:raise ValueError(f'{name} must be positive')
    if a.max_cells is not None and a.max_cells<1:raise ValueError('max-cells must be positive')
    faiss.omp_set_num_threads(a.faiss_threads)
    out=a.output_dir.resolve(); tmp=out.with_name(f'.{out.name}.building-{os.getpid()}')
    if out.exists() and not a.overwrite: raise FileExistsError(f'{out} exists; use --overwrite')
    if tmp.exists():shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        emb=np.load(a.embeddings,mmap_mode='r'); full_n,dim=emb.shape; n=min(full_n,a.max_cells) if a.max_cells else full_n
        if n<1:raise ValueError('embedding input must contain cells')
        md=json.loads((a.csr_dir/'metadata.json').read_text()); names=md['lineages']; lineage=np.load(a.csr_dir/'lineage_ids.npy',mmap_mode='r')[:n]
        all_cell_ids=np.load(a.csr_dir/'cell_ids.npy',mmap_mode='r'); cell_ids=all_cell_ids[:n]
        if len(all_cell_ids)!=full_n or len(lineage)!=n:
            raise ValueError('embedding and CSR cell/lineage rows are not aligned')
        if a.predicted_stage is not None:
            from predicted_stage import load_predicted_stage
            log(f'loading and validating predicted stage: {a.predicted_stage}')
            cell_stage=load_predicted_stage(a.predicted_stage,all_cell_ids,a.metadata_chunk_size)[:n]
            stage_source='predicted_stage'
        else:
            stage_id=np.load(a.csr_dir/'stage_ids.npy',mmap_mode='r')[:n]
            if len(stage_id)!=n:raise ValueError('stage_ids and embeddings are not row-aligned')
            stages=np.asarray(md['stages'],dtype=np.float64)
            cell_stage=stages[np.asarray(stage_id,dtype=np.int32)]
            stage_source='original_stage'
        bins=np.floor((cell_stage-a.stage_bin_origin)/a.stage_bin_width)
        if not np.isfinite(bins).all() or np.any(bins<np.iinfo(np.int32).min) or np.any(bins>np.iinfo(np.int32).max):
            raise ValueError('non-finite stages/origin or stage bins outside int32 range')
        stage_bin_id=bins.astype(np.int32)
        log(f'stage source={stage_source}; range=[{cell_stage.min():.6g}, {cell_stage.max():.6g}]')
        min_bin=int(stage_bin_id.min());n_bins=int(stage_bin_id.max())-min_bin+1
        stratum_key=np.asarray(lineage,dtype=np.int64)*n_bins+(np.asarray(stage_bin_id,dtype=np.int64)-min_bin)
        stratum_order=np.argsort(stratum_key,kind='stable'); sorted_key=stratum_key[stratum_order]
        unique_keys,starts,stratum_counts=np.unique(sorted_key,return_index=True,return_counts=True)
        micro_assignment=np.lib.format.open_memmap(tmp/'cell_to_microcell.npy',mode='w+',dtype=np.int32,shape=(n,)); micro_assignment[:]=-1
        all_micro_groups=[]; micro_lineages=[]; micro_constraint_stages=[]
        rng=np.random.default_rng(a.seed)
        log(f'input={n:,} x {dim}; nonempty lineage-stage-bin strata={len(unique_keys):,}; bin_width={a.stage_bin_width:g}')
        for stratum_index,(key,start,count) in enumerate(zip(unique_keys,starts,stratum_counts)):
            lid=int(key//n_bins);bid=int(key%n_bins)+min_bin;name=names[lid]
            bin_left=float(a.stage_bin_origin+bid*a.stage_bin_width);bin_right=float(bin_left+a.stage_bin_width);bin_label=f'[{bin_left:g},{bin_right:g})'
            ids=stratum_order[int(start):int(start+count)]
            log(f'stratum {stratum_index+1}/{len(unique_keys)}: lineage={name}; stage_bin={bin_label}; cells={len(ids):,}')
            perm=rng.permutation(ids); phase1=[]
            for pi,lo in enumerate(range(0,len(perm),a.pile_size)):
                pile=perm[lo:lo+a.pile_size]; lab=partition_microcells(emb[pile],a)
                phase1.extend(split_groups(pile,lab)); log(f'  phase1 pile {pi+1}/{math.ceil(len(perm)/a.pile_size)} -> total microcells={len(phase1):,}')
            centers=raw_means(emb,phase1); piles=coherent_piles(phase1,centers,a.pile_size,a.regroup_knn); phase2=[]
            for pi,parts in enumerate(piles):
                pile=np.concatenate(parts); lab=partition_microcells(emb[pile],a)
                phase2.extend(split_groups(pile,lab)); log(f'  phase2 pile {pi+1}/{len(piles)} -> total microcells={len(phase2):,}')
            for members in phase2:
                micro_id=len(all_micro_groups);micro_assignment[members]=micro_id;all_micro_groups.append(members)
                micro_lineages.append((lid,name));micro_constraint_stages.append((bid,bin_left,bin_right,bin_label))
            log(f'  retained {len(phase2):,} final microcells')
        if np.any(micro_assignment<0):raise RuntimeError(f'{np.count_nonzero(micro_assignment<0)} cells were not assigned to microcells')

        micro_emb=raw_means(emb,all_micro_groups); np.save(tmp/'microcell_embeddings.npy',micro_emb)

        log(f'aggregating cell types for {len(all_micro_groups):,} final microcells')
        mcts,mcomp,mmatched=celltype_tables(a.metadata,cell_ids,micro_assignment,len(all_micro_groups),a.metadata_chunk_size)
        mctmap={x['microcell_id']:x for x in mcts}; micro_rows=[]
        for micro_id,g in enumerate(all_micro_groups):
            raw=np.asarray(emb[g],np.float32);st=np.asarray(cell_stage[g],dtype=np.float64)
            center=normalize(micro_emb[micro_id:micro_id+1])[0]
            sim=(raw@center)/np.maximum(np.linalg.norm(raw,axis=1),1e-12)
            lid,name=micro_lineages[micro_id];bid,bin_left,bin_right,bin_label=micro_constraint_stages[micro_id];ct=dict(mctmap[micro_id]);ct.pop('microcell_id',None)
            row=dict(microcell_id=micro_id,stage_source=stage_source,lineage=name,lineage_id=lid,
                     constraint_stage_id=bid,constraint_stage=(bin_left+bin_right)/2,
                     stage_bin_id=bid,stage_bin_left=bin_left,stage_bin_right=bin_right,stage_bin_label=bin_label,
                     n_cells=len(g),mean_stage=float(st.mean()),stage_std=float(st.std()),min_stage=float(st.min()),max_stage=float(st.max()),
                     mean_cosine_distance=float((1-sim).mean()),p95_cosine_distance=float(np.quantile(1-sim,.95)))
            row.update(ct); micro_rows.append(row)
        micro_table=pd.DataFrame(micro_rows)
        micro_table['below_min_size']=micro_table.n_cells < a.microcell_min_size
        micro_table.to_parquet(tmp/'microcells.parquet',index=False)
        sizes=micro_table.n_cells.to_numpy()
        grouping_summary=dict(method=a.microcell_method,n_microcells=len(sizes),n_cells=n,
                              min_size=int(sizes.min()),median_size=float(np.median(sizes)),max_size=int(sizes.max()),
                              singleton_groups=int(np.count_nonzero(sizes==1)),
                              below_min_size_groups=int(np.count_nonzero(sizes<a.microcell_min_size)),
                              below_min_size_cell_fraction=float(sizes[sizes<a.microcell_min_size].sum()/n),
                              assigned_cell_fraction=float(np.count_nonzero(micro_assignment>=0)/n))
        (tmp/'microcell_grouping_summary.json').write_text(json.dumps(grouping_summary,indent=2)+'\n')
        log(f'microcell grouping summary: {grouping_summary}')
        micro_comp=pd.DataFrame(mcomp,columns=['microcell_id','celltype','cell_count','fraction','rank'])
        micro_comp.to_parquet(tmp/'microcell_celltype_composition.parquet',index=False)
        cfg={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()};cfg.update(stage_source=stage_source,stage_value_column='predicted_stage' if a.predicted_stage is not None else 'stage',n_cells=n,n_microcells=len(micro_rows),microcell_celltype_matched=mmatched,n_nonempty_lineage_stage_bin_strata=len(unique_keys),stage_constraint='fixed_width_stage_bin_hard',stage_bin_boundary='left_closed_right_open',method=f'embedding_only_{a.microcell_method}_cosine_knn_two_pass_lineage_fixed_stage_bin_microcells')
        (tmp/'run_config.json').write_text(json.dumps(cfg,indent=2,ensure_ascii=False)); del micro_assignment
        if out.exists():shutil.rmtree(out)
        tmp.rename(out);log(f'done: {out}')
    except Exception:
        log(f'failed; partial outputs kept: {tmp}');raise
if __name__=='__main__':main()
