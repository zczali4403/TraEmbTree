#!/usr/bin/env python3
# MC2-inspired microcells with structure-based marginal landmark allocation.
#
# Full-data example:
#   OPENBLAS_NUM_THREADS=32 OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 \
#   /home/aiscuser/.conda/envs/train/bin/python \
#     /mnt/input/sc_cz/Concord/eval/2026_08_19/select_landmark_nodes_mc2_lineage_stage_marginal.py \
#     --target-nodes 2000 \
#     --output-dir /mnt/input/sc_cz/Concord/eval/2026_08_19/landmark_nodes_mc2_lineage_stage_marginal \
#     --overwrite
#
# Quick small test:
#   .../python select_landmark_nodes_mc2_lineage_stage_marginal.py --max-cells 100000 \
#     --target-nodes 100 --output-dir landmark_nodes_mc2_marginal_test --overwrite
"""MC2-inspired TraEmb selection within hard lineage x exact-stage strata.

This is not the original gene/UMI-based Metacell-2 implementation. It borrows
its two-pass pile/refinement design, replacing expression similarity by cosine
kNN graphs. All final node coordinates are arithmetic means of original (not
L2-transformed) TraEmb vectors.
The two microcell passes are unchanged. Only the final per-stratum landmark
quota is allocated after all microcells exist, using normalized mean marginal
cosine-coverage gain over equally weighted microcell centers instead of
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
    p.add_argument('--metadata',type=Path,default=Path('/mnt/input/sc_cz/Concord/eval/2026_08_19/all_lineage_260811_liver_reanno.csv'))
    p.add_argument('--output-dir',type=Path,default=HERE/'landmark_nodes_mc2_lineage_stage_marginal')
    p.add_argument('--target-nodes',type=int,default=2000)
    p.add_argument('--min-nodes-per-stratum',type=int,default=1,help='Minimum nodes per nonempty lineage x stage stratum.')
    p.add_argument('--quota-power',type=float,default=.5)
    p.add_argument('--allocation-knn',type=int,default=30,help='Microcell neighbors used for density in marginal node allocation.')
    p.add_argument('--allocation-density-power',type=float,default=1.0,help='Density strength when proposing each additional node.')
    p.add_argument('--pile-size',type=int,default=20000)
    p.add_argument('--microcell-size',type=int,default=500)
    p.add_argument('--microcell-min-size',type=int,default=200)
    p.add_argument('--microcell-max-size',type=int,default=1000)
    p.add_argument('--knn',type=int,default=30)
    p.add_argument('--regroup-knn',type=int,default=64)
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

def allocate(counts,total,minimum,power):
    active=counts>0; n=int(active.sum())
    if total<n: raise ValueError(f'target-nodes must be >= nonempty strata ({n})')
    q=np.zeros(len(counts),dtype=int); base=min(minimum,total//n); q[active]=base
    left=total-int(q.sum()); w=np.where(active,counts.astype(float)**power,0); raw=left*w/w.sum()
    q+=np.floor(raw).astype(int)
    for i in np.argsort(-(raw-np.floor(raw)))[:total-int(q.sum())]: q[i]+=1
    return q

class DSU:
    def __init__(self,n,weights=None):
        self.p=np.arange(n); self.w=np.ones(n,dtype=np.int64) if weights is None else np.asarray(weights,dtype=np.int64).copy(); self.n=n
    def find(self,x):
        while self.p[x]!=x: self.p[x]=self.p[self.p[x]]; x=self.p[x]
        return int(x)
    def union(self,a,b,cap=None):
        a=self.find(a); b=self.find(b)
        if a==b or (cap is not None and self.w[a]+self.w[b]>cap): return False
        if self.w[a]<self.w[b]: a,b=b,a
        self.p[b]=a; self.w[a]+=self.w[b]; self.n-=1; return True
    def labels(self):
        roots=np.asarray([self.find(i) for i in range(len(self.p))]); _,lab=np.unique(roots,return_inverse=True); return lab.astype(np.int32)

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


def graph_partition(vectors,target_size,min_size,max_size,weights=None,target_groups=None,k=30):
    n=len(vectors); weights=np.ones(n,dtype=np.int64) if weights is None else np.asarray(weights,dtype=np.int64)
    if target_groups is None: target_groups=max(1,int(round(weights.sum()/target_size)))
    target_groups=min(target_groups,n)
    if target_groups==n: return np.arange(n,dtype=np.int32)
    # Cell-level pile partition: enforce balanced nonempty groups exactly.
    if np.all(weights==1):
        return balanced_knn_region_partition(vectors,target_groups,k)
    # Weighted microcell-to-landmark aggregation.
    src,dst,score=cosine_knn_edges(vectors,k); order=np.argsort(-score); dsu=DSU(n,weights)
    cap=max_size if max_size is not None else None
    for e in order:
        if dsu.n<=target_groups: break
        dsu.union(int(src[e]),int(dst[e]),cap)
    # Relax cap only when graph connectivity/size constraints prevent requested count.
    for e in order:
        if dsu.n<=target_groups: break
        dsu.union(int(src[e]),int(dst[e]),None)
    # kNN graph should be connected; if not, merge closest component centroids.
    while dsu.n>target_groups:
        lab=dsu.labels(); m=int(lab.max())+1; z=normalize(vectors)
        sums=np.zeros((m,z.shape[1]),np.float64); np.add.at(sums,lab,z*weights[:,None]); sums=normalize(sums)
        np.fill_diagonal((sim:=sums@sums.T),-np.inf); a,b=np.unravel_index(np.argmax(sim),sim.shape)
        ia=int(np.flatnonzero(lab==a)[0]); ib=int(np.flatnonzero(lab==b)[0]); dsu.union(ia,ib,None)
    labels=dsu.labels()
    # Attach undersized components to their closest component; only used locally.
    if min_size and target_groups>1:
        sizes=np.bincount(labels,weights=weights); small=np.flatnonzero(sizes<min_size)
        if len(small):
            z=normalize(vectors); sums=np.zeros((len(sizes),z.shape[1]),np.float64); np.add.at(sums,labels,z*weights[:,None]); cent=normalize(sums)
            for g in small:
                candidates=np.flatnonzero(sizes>=min_size)
                if not len(candidates): break
                to=int(candidates[np.argmax(cent[candidates]@cent[g])]); labels[labels==g]=to; sizes[to]+=sizes[g]; sizes[g]=0
            _,labels=np.unique(labels,return_inverse=True); labels=labels.astype(np.int32)
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

def celltype_tables(metadata,cell_ids,assignment,n_nodes,chunk):
    import pandas as pd
    n=len(cell_ids); hashes=np.empty(n,np.uint64)
    for lo in range(0,n,chunk): hashes[lo:min(n,lo+chunk)]=hash_strings(cell_ids[lo:min(n,lo+chunk)])
    order=np.argsort(hashes); hs=hashes[order]; counts=defaultdict(int); matched=rows_seen=0
    for df in pd.read_csv(metadata,usecols=['cell','celltype'],chunksize=chunk):
        rows_seen+=len(df); h=hash_strings(df.cell.astype(str).to_numpy()); pos=np.searchsorted(hs,h); ok=pos<n; ok&=hs[np.minimum(pos,n-1)]==h
        rr=order[pos[ok]]; ct=df.loc[ok,'celltype'].fillna('Unknown').astype(str).to_numpy()
        valid=rr<len(assignment); rr=rr[valid]; ct=ct[valid]
        for node,name in zip(np.asarray(assignment[rr]),ct): counts[(int(node),name)]+=1
        matched+=len(rr); log(f'celltype matched {matched:,}/{rows_seen:,}')
    summary=[]; comp=[]
    by=defaultdict(list)
    for (node,ct),c in counts.items():by[node].append((ct,c))
    for node in range(n_nodes):
        vals=sorted(by[node],key=lambda x:(-x[1],x[0])); total=sum(x[1] for x in vals)
        for rank,(ct,c) in enumerate(vals,1):comp.append(dict(node_id=node,celltype=ct,cell_count=c,fraction=c/total,rank=rank))
        dom,dc=vals[0] if vals else ('Unknown',0)
        summary.append(dict(node_id=node,dominant_celltype=dom,celltype_purity=dc/total if total else np.nan,dominant_celltype_count=dc,celltype_types=len(vals),celltype_matched_cells=total))
    return summary,comp,matched

def allocate_by_microcell_gain(center_sets,total,minimum,k,density_power):
    """Allocate landmark counts by normalized mean coverage gain over microcells."""
    n_strata=len(center_sets)
    if total<n_strata:raise ValueError(f'target-nodes={total} is below {n_strata} nonempty strata')
    if total>sum(len(x) for x in center_sets):raise ValueError('target-nodes exceeds total final microcells')
    states=[];heap=[]

    def next_proposal(state):
        if len(state['selected'])>=len(state['z']):return None
        score=state['nearest'].astype(np.float64)*np.power(state['density_rank'],density_power)
        score[np.asarray(state['selected'],dtype=np.int64)]=-np.inf
        candidate=int(np.argmax(score));distance=np.maximum(0.,1.-state['z']@state['z'][candidate]).astype(np.float32)
        gain=float(np.maximum(0.,state['nearest']-distance).mean())
        return candidate,distance,gain

    for centers in center_sets:
        z=normalize(centers);m=len(z)
        if m==1:density=np.ones(1,np.float32)
        else:
            kk=min(max(1,k),m-1);index=__import__('faiss').IndexFlatIP(z.shape[1]);index.add(z)
            similarity,_=index.search(z,kk+1)
            density=(1./np.maximum(np.maximum(0.,1.-similarity[:,1:]).mean(1),1e-8)).astype(np.float32)
        order=np.argsort(density,kind='stable');rank=np.empty(m,np.float32);rank[order]=(np.arange(m,dtype=np.float32)+1)/m
        first=int(np.argmax(density));nearest=np.maximum(0.,1.-z@z[first]).astype(np.float32)
        state=dict(z=z,density_rank=rank,selected=[first],nearest=nearest,proposal=None)
        while len(state['selected'])<min(max(1,minimum),m):
            proposal=next_proposal(state)
            if proposal is None:break
            candidate,distance,_=proposal;state['selected'].append(candidate);state['nearest']=np.minimum(state['nearest'],distance)
        states.append(state)

    allocated=sum(len(s['selected']) for s in states)
    if allocated>total:raise ValueError('minimum nodes per stratum exceeds target-nodes')
    for si,state in enumerate(states):
        state['proposal']=next_proposal(state)
        if state['proposal'] is not None:heapq.heappush(heap,(-state['proposal'][2],si,len(state['selected'])))
    while allocated<total:
        while heap:
            negative_gain,si,version=heapq.heappop(heap);state=states[si]
            if version==len(state['selected']) and state['proposal'] is not None:break
        else:raise RuntimeError('no stratum can accept remaining nodes')
        candidate,distance,gain=state['proposal'];state['selected'].append(candidate)
        state['nearest']=np.minimum(state['nearest'],distance);allocated+=1
        if allocated%25==0 or allocated==total:
            log(f'marginal node allocation {allocated:,}/{total:,}: stratum={si}; nodes={len(state["selected"])}; mean_gain={gain:.6g}')
        state['proposal']=next_proposal(state)
        if state['proposal'] is not None:heapq.heappush(heap,(-state['proposal'][2],si,len(state['selected'])))
    return np.asarray([len(s['selected']) for s in states],dtype=np.int32)

def main():
    a=args_parser(); import faiss, pandas as pd
    faiss.omp_set_num_threads(a.faiss_threads)
    out=a.output_dir.resolve(); tmp=out.with_name(f'.{out.name}.building-{os.getpid()}')
    if out.exists() and not a.overwrite: raise FileExistsError(f'{out} exists; use --overwrite')
    if tmp.exists():shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        emb=np.load(a.embeddings,mmap_mode='r'); full_n,dim=emb.shape; n=min(full_n,a.max_cells) if a.max_cells else full_n
        md=json.loads((a.csr_dir/'metadata.json').read_text()); names=md['lineages']; lineage=np.load(a.csr_dir/'lineage_ids.npy',mmap_mode='r')[:n]
        stage_id=np.load(a.csr_dir/'stage_ids.npy',mmap_mode='r')[:n]; cell_ids=np.load(a.csr_dir/'cell_ids.npy',mmap_mode='r')[:n]
        stages=np.asarray(md['stages'],dtype=np.float32)
        n_stages=len(stages); stratum_key=np.asarray(lineage,dtype=np.int32)*n_stages+np.asarray(stage_id,dtype=np.int32)
        stratum_order=np.argsort(stratum_key,kind='stable'); sorted_key=stratum_key[stratum_order]
        unique_keys,starts,stratum_counts=np.unique(sorted_key,return_index=True,return_counts=True)
        if a.target_nodes<len(unique_keys):
            raise ValueError(f'target-nodes={a.target_nodes} is below {len(unique_keys)} nonempty lineage x stage strata')
        assignment=np.lib.format.open_memmap(tmp/'cell_to_node.npy',mode='w+',dtype=np.int32,shape=(n,)); assignment[:]=-1
        micro_assignment=np.lib.format.open_memmap(tmp/'cell_to_microcell.npy',mode='w+',dtype=np.int32,shape=(n,)); micro_assignment[:]=-1
        all_node_groups=[]; node_lineages=[]; node_constraint_stages=[]
        all_micro_groups=[]; micro_lineages=[]; micro_constraint_stages=[]
        stratum_phase2=[];stratum_centers=[];stratum_weights=[];stratum_info=[];stratum_cell_counts=[];stratum_micro_starts=[]
        rng=np.random.default_rng(a.seed)
        log(f'input={n:,} x {dim}; nonempty lineage-stage strata={len(unique_keys):,}; final nodes={a.target_nodes}')
        for stratum_index,(key,start,count) in enumerate(zip(unique_keys,starts,stratum_counts)):
            lid=int(key//n_stages); sid=int(key%n_stages); name=names[lid]; stage_value=float(stages[sid])
            ids=stratum_order[int(start):int(start+count)]
            log(f'stratum {stratum_index+1}/{len(unique_keys)}: lineage={name}; stage={stage_value:g}; cells={len(ids):,}')
            perm=rng.permutation(ids); phase1=[]
            for pi,lo in enumerate(range(0,len(perm),a.pile_size)):
                pile=perm[lo:lo+a.pile_size]; lab=graph_partition(emb[pile],a.microcell_size,a.microcell_min_size,a.microcell_max_size,k=a.knn)
                phase1.extend(split_groups(pile,lab)); log(f'  phase1 pile {pi+1}/{math.ceil(len(perm)/a.pile_size)} -> total microcells={len(phase1):,}')
            centers=raw_means(emb,phase1); piles=coherent_piles(phase1,centers,a.pile_size,a.regroup_knn); phase2=[]
            for pi,parts in enumerate(piles):
                pile=np.concatenate(parts); lab=graph_partition(emb[pile],a.microcell_size,a.microcell_min_size,a.microcell_max_size,k=a.knn)
                phase2.extend(split_groups(pile,lab)); log(f'  phase2 pile {pi+1}/{len(piles)} -> total microcells={len(phase2):,}')
            micro_centers=raw_means(emb,phase2);weights=np.asarray([len(g) for g in phase2],dtype=np.int64)
            stratum_phase2.append(phase2);stratum_centers.append(micro_centers);stratum_weights.append(weights)
            stratum_info.append((lid,name,sid,stage_value));stratum_cell_counts.append(len(ids));stratum_micro_starts.append(len(all_micro_groups))
            for members in phase2:
                micro_id=len(all_micro_groups);micro_assignment[members]=micro_id;all_micro_groups.append(members)
                micro_lineages.append((lid,name));micro_constraint_stages.append((sid,stage_value))
            log(f'  retained {len(phase2):,} final microcells; landmark quota deferred')
        if np.any(micro_assignment<0):raise RuntimeError(f'{np.count_nonzero(micro_assignment<0)} cells were not assigned to microcells')

        quota=allocate_by_microcell_gain(stratum_centers,a.target_nodes,a.min_nodes_per_stratum,a.allocation_knn,a.allocation_density_power)
        micro_to_node=np.full(len(all_micro_groups),-1,dtype=np.int32);offset=0
        for stratum_index,(phase2,micro_centers,weights,info,cell_count,micro_start) in enumerate(zip(stratum_phase2,stratum_centers,stratum_weights,stratum_info,stratum_cell_counts,stratum_micro_starts)):
            lid,name,sid,stage_value=info;q=int(quota[stratum_index]);ideal=max(1,int(math.ceil(cell_count/q)))
            final_lab=graph_partition(micro_centers,ideal,0,2*ideal,weights=weights,target_groups=q,k=min(a.regroup_knn,max(1,len(phase2)-1)))
            if len(np.unique(final_lab))!=q:
                raise RuntimeError(f'{name}: final graph partition produced {len(np.unique(final_lab))}/{q} nodes')
            for i,members in enumerate(phase2):
                node=offset+int(final_lab[i]);micro_to_node[micro_start+i]=node;assignment[members]=node
            for g in range(q):
                members=np.concatenate([phase2[i] for i in np.flatnonzero(final_lab==g)])
                all_node_groups.append(members);node_lineages.append((lid,name));node_constraint_stages.append((sid,stage_value))
            offset+=q;log(f'  finalized stratum {stratum_index+1}/{len(quota)}: {len(phase2):,} microcells -> {q} nodes')
        if np.any(micro_to_node<0):raise RuntimeError(f'{np.count_nonzero(micro_to_node<0)} microcells were not assigned to nodes')
        if np.any(assignment<0):raise RuntimeError(f'{np.count_nonzero(assignment<0)} cells were not assigned to nodes')
        if np.any(micro_assignment<0):raise RuntimeError(f'{np.count_nonzero(micro_assignment<0)} cells were not assigned to microcells')
        node_emb=raw_means(emb,all_node_groups); np.save(tmp/'node_embeddings.npy',node_emb)
        micro_emb=raw_means(emb,all_micro_groups); np.save(tmp/'microcell_embeddings.npy',micro_emb)
        np.save(tmp/'microcell_to_node.npy',np.asarray(micro_to_node,dtype=np.int32))
        cts,comp,matched=celltype_tables(a.metadata,cell_ids,assignment,len(all_node_groups),a.metadata_chunk_size); ctmap={x['node_id']:x for x in cts}
        rows=[]
        for node,g in enumerate(all_node_groups):
            st=stages[np.asarray(stage_id[g],dtype=int)]; center=normalize(node_emb[node:node+1])[0]; sim=(np.asarray(emb[g],np.float32)@center)/np.maximum(np.linalg.norm(np.asarray(emb[g],np.float32),axis=1),1e-12)
            lid,name=node_lineages[node]; sid,constraint_stage=node_constraint_stages[node]
            row=dict(node_id=node,lineage=name,lineage_id=lid,constraint_stage_id=sid,constraint_stage=constraint_stage,n_cells=len(g),mean_stage=float(st.mean()),stage_std=float(st.std()),min_stage=float(st.min()),max_stage=float(st.max()),mean_cosine_distance=float((1-sim).mean()),p95_cosine_distance=float(np.quantile(1-sim,.95)))
            row.update(ctmap[node]);rows.append(row)
        pd.DataFrame(rows).to_parquet(tmp/'nodes.parquet',index=False); pd.DataFrame(comp).to_parquet(tmp/'node_celltype_composition.parquet',index=False)

        log(f'aggregating cell types for {len(all_micro_groups):,} final microcells')
        mcts,mcomp,mmatched=celltype_tables(a.metadata,cell_ids,micro_assignment,len(all_micro_groups),a.metadata_chunk_size)
        mctmap={x['node_id']:x for x in mcts}; micro_rows=[]
        for micro_id,g in enumerate(all_micro_groups):
            raw=np.asarray(emb[g],np.float32); st=stages[np.asarray(stage_id[g],dtype=int)]
            center=normalize(micro_emb[micro_id:micro_id+1])[0]
            sim=(raw@center)/np.maximum(np.linalg.norm(raw,axis=1),1e-12)
            lid,name=micro_lineages[micro_id]; sid,constraint_stage=micro_constraint_stages[micro_id]; ct=dict(mctmap[micro_id]); ct.pop('node_id',None)
            row=dict(microcell_id=micro_id,landmark_node_id=int(micro_to_node[micro_id]),lineage=name,lineage_id=lid,constraint_stage_id=sid,constraint_stage=constraint_stage,n_cells=len(g),mean_stage=float(st.mean()),stage_std=float(st.std()),min_stage=float(st.min()),max_stage=float(st.max()),mean_cosine_distance=float((1-sim).mean()),p95_cosine_distance=float(np.quantile(1-sim,.95)))
            row.update(ct); micro_rows.append(row)
        pd.DataFrame(micro_rows).to_parquet(tmp/'microcells.parquet',index=False)
        micro_comp=pd.DataFrame(mcomp).rename(columns={'node_id':'microcell_id'})
        micro_comp.to_parquet(tmp/'microcell_celltype_composition.parquet',index=False)
        cfg={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()};cfg.update(n_cells=n,n_nodes=len(rows),n_microcells=len(micro_rows),celltype_matched=matched,microcell_celltype_matched=mmatched,n_nonempty_lineage_stage_strata=len(unique_keys),stage_constraint='exact_stage_hard',allocation_objective='equal_microcell_weight_normalized_mean_marginal_cosine_coverage_gain',nodes_per_stratum=[int(x) for x in quota],method='embedding_only_mc2_inspired_cosine_knn_two_pass_lineage_exact_stage_marginal_microcell_allocation')
        (tmp/'run_config.json').write_text(json.dumps(cfg,indent=2,ensure_ascii=False)); del assignment,micro_assignment
        if out.exists():shutil.rmtree(out)
        tmp.rename(out);log(f'done: {out}')
    except Exception:
        log(f'failed; partial outputs kept: {tmp}');raise
if __name__=='__main__':main()
