#!/usr/bin/env python3
# 用法示例（在当前目录运行）：
#
#   OPENBLAS_NUM_THREADS=32 OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 \
#   /home/aiscuser/.conda/envs/train/bin/python \
#     /mnt/input/sc_cz/Concord/eval/2026_08_19/select_landmark_nodes.py \
#     --target-nodes 1000 \
#     --output-dir /mnt/input/sc_cz/Concord/eval/2026_08_19/landmark_nodes \
#     --overwrite
#
# 查看全部参数：
#   /home/aiscuser/.conda/envs/train/bin/python select_landmark_nodes.py --help

"""Select lineage-stratified landmark nodes from TraEmb embeddings.

Example
-------
OPENBLAS_NUM_THREADS=32 OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 \
/home/aiscuser/.conda/envs/train/bin/python select_landmark_nodes.py \
  --target-nodes 1000 --output-dir landmark_nodes --overwrite

Clustering reads the original TraEmb vectors and uses explicit cosine distance. Each
saved node coordinate is the arithmetic mean of the ORIGINAL, unmodified 100-D
embeddings of every assigned member. The node cell type is its dominant member
cell type and celltype_purity is the corresponding fraction.
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "32")

import argparse, json, math, shutil, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent

def log(s):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {s}", flush=True)

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--embeddings", type=Path, default=Path("/mnt/input/sc_cz/Concord/eval/2026_09_03/embeddings.npy"))
    p.add_argument("--csr-dir", type=Path, default=Path("/scratch/amlt_code/traemb_csr_0829"))
    p.add_argument("--metadata", type=Path, default=Path("/mnt/input/sc_cz/Concord/data/all_lineage_260829.csv"))
    p.add_argument("--output-dir", type=Path, default=HERE / "landmark_nodes")
    p.add_argument("--target-nodes", type=int, default=500)
    p.add_argument("--min-nodes-per-lineage", type=int, default=10)
    p.add_argument("--quota-power", type=float, default=0.5)
    p.add_argument("--fit-sample-per-lineage", type=int, default=300000)
    p.add_argument("--batch-size", type=int, default=65536)
    p.add_argument("--cosine-kmeans-iterations", type=int, default=20)
    p.add_argument("--chunk-size", type=int, default=250000)
    p.add_argument("--metadata-chunk-size", type=int, default=500000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()

def quotas(counts, total, minimum, power):
    n = len(counts)
    if total < n: raise ValueError(f"target-nodes must be >= number of lineages ({n})")
    minimum = min(minimum, total // n)
    q = np.full(n, minimum, dtype=int)
    left = total - int(q.sum())
    if left:
        w = np.asarray(counts, float) ** power; raw = left * w / w.sum()
        q += np.floor(raw).astype(int)
        for i in np.argsort(-(raw - np.floor(raw)))[:total-int(q.sum())]: q[i] += 1
    return q

def cosine_similarity_to_centers(x, centers):
    """Cosine similarity from original vectors; x is never modified in place."""
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1)
    return (x @ centers.T) / np.maximum(norms[:, None], 1e-12)


def fit_cosine_kmeans(x, n_clusters, seed, batch_size, max_iter):
    """Lloyd-style spherical k-means using explicit cosine distance."""
    rng = np.random.default_rng(seed)
    n, dim = x.shape
    initial = rng.choice(n, n_clusters, replace=False)
    centers = np.asarray(x[initial], dtype=np.float32).copy()
    centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
    previous = None
    for iteration in range(max_iter):
        sums = np.zeros((n_clusters, dim), dtype=np.float64)
        counts = np.zeros(n_clusters, dtype=np.int64)
        changed = 0
        labels_all = np.empty(n, dtype=np.int32)
        for lo in range(0, n, batch_size):
            hi = min(n, lo + batch_size)
            raw = np.asarray(x[lo:hi], dtype=np.float32)
            labels = np.argmax(cosine_similarity_to_centers(raw, centers), axis=1).astype(np.int32)
            labels_all[lo:hi] = labels
            unit = raw / np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
            np.add.at(sums, labels, unit)
            np.add.at(counts, labels, 1)
        if previous is not None:
            changed = int(np.count_nonzero(labels_all != previous))
        empty = np.flatnonzero(counts == 0)
        if len(empty):
            replacement = rng.choice(n, len(empty), replace=False)
            sums[empty] = np.asarray(x[replacement], dtype=np.float32)
            counts[empty] = 1
        centers = sums.astype(np.float32)
        centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
        log(f"  cosine k-means iteration {iteration + 1}/{max_iter}: changed={changed:,}")
        if previous is not None and changed == 0:
            break
        previous = labels_all
    return centers

def hash_strings(values):
    import pandas as pd
    return pd.util.hash_array(np.asarray(values, dtype=object), categorize=False).astype(np.uint64, copy=False)

def aggregate_celltypes(metadata, cell_ids, cell_to_node, n_nodes, chunk_size):
    import pandas as pd
    n = len(cell_ids)
    log(f"building cell-ID hash index for {n:,} embedding rows")
    hashes = np.empty(n, dtype=np.uint64)
    for a in range(0, n, chunk_size):
        b=min(n,a+chunk_size); hashes[a:b]=hash_strings(cell_ids[a:b])
    order=np.argsort(hashes); sorted_hash=hashes[order]
    counts=defaultdict(int); matched=0; total_rows=0
    for df in pd.read_csv(metadata, usecols=["cell","celltype"], chunksize=chunk_size):
        total_rows += len(df); h=hash_strings(df["cell"].astype(str).to_numpy())
        pos=np.searchsorted(sorted_hash,h); ok=pos<n; ok &= sorted_hash[np.minimum(pos,n-1)]==h
        rows=order[pos[ok]]; cts=df.loc[ok,"celltype"].fillna("Unknown").astype(str).to_numpy()
        nodes=np.asarray(cell_to_node[rows],dtype=np.int32)
        for node,ct in zip(nodes,cts): counts[(int(node),ct)] += 1
        matched += int(ok.sum()); log(f"celltype metadata: matched {matched:,}/{total_rows:,} rows")
    records=[]; summary=[]
    for node in range(n_nodes):
        vals=[(ct,c) for (nd,ct),c in counts.items() if nd==node]; vals.sort(key=lambda z:(-z[1],z[0]))
        tot=sum(c for _,c in vals)
        for rank,(ct,c) in enumerate(vals,1): records.append({"node_id":node,"celltype":ct,"cell_count":c,"fraction":c/tot,"rank":rank})
        dom,dc=(vals[0] if vals else ("Unknown",0)); second=vals[1][0] if len(vals)>1 else ""
        ent=-sum((c/tot)*math.log(c/tot) for _,c in vals) if tot else np.nan
        summary.append({"node_id":node,"dominant_celltype":dom,"celltype_purity":dc/tot if tot else np.nan,
                        "dominant_celltype_count":dc,"second_celltype":second,"celltype_entropy":ent,
                        "celltype_types":len(vals),"celltype_matched_cells":tot})
    return summary, records, {"metadata_rows":total_rows,"matched_rows":matched}

def main():
    a=parse_args(); out=a.output_dir.resolve(); tmp=out.with_name(f".{out.name}.building-{os.getpid()}")
    if out.exists() and not a.overwrite: raise FileExistsError(f"{out} exists; use --overwrite")
    if tmp.exists(): shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        import pandas as pd
        emb=np.load(a.embeddings,mmap_mode="r"); n,d=emb.shape
        meta=json.loads((a.csr_dir/"metadata.json").read_text()); names=list(meta["lineages"])
        lineage=np.load(a.csr_dir/"lineage_ids.npy",mmap_mode="r")
        stage_ids=np.load(a.csr_dir/"stage_ids.npy",mmap_mode="r")
        cell_ids=np.load(a.csr_dir/"cell_ids.npy",mmap_mode="r")
        if not (len(lineage)==len(stage_ids)==len(cell_ids)==n): raise ValueError("CSR labels and embeddings are not row-aligned")
        stage_values=np.asarray([float(x) for x in meta["stages"]],dtype=np.float32)
        counts=np.bincount(lineage,minlength=len(names)); q=quotas(counts,a.target_nodes,a.min_nodes_per_lineage,a.quota_power)
        log(f"input={n:,} x {d}; target nodes={a.target_nodes}")
        rng=np.random.default_rng(a.seed); models=[]; offsets=np.r_[0,np.cumsum(q)]
        for lid,name in enumerate(names):
            idx=np.flatnonzero(lineage==lid); take=min(len(idx),a.fit_sample_per_lineage)
            sample=idx if take==len(idx) else rng.choice(idx,take,replace=False)
            log(f"fit {name}: cells={len(idx):,}, sample={take:,}, nodes={q[lid]}")
            centers = fit_cosine_kmeans(
                emb[sample], int(q[lid]), a.seed + lid,
                a.batch_size, a.cosine_kmeans_iterations,
            )
            models.append(centers)
        assign=np.lib.format.open_memmap(tmp/"cell_to_node.npy",mode="w+",dtype=np.int32,shape=(n,))
        dist=np.lib.format.open_memmap(tmp/"cell_to_node_distance.npy",mode="w+",dtype=np.float32,shape=(n,))
        sums=np.zeros((a.target_nodes,d),np.float64); sizes=np.zeros(a.target_nodes,np.int64)
        dist_sum=np.zeros(a.target_nodes,np.float64); stage_sum=np.zeros(a.target_nodes,np.float64); stage_sq=np.zeros(a.target_nodes,np.float64)
        stage_min=np.full(a.target_nodes,np.inf); stage_max=np.full(a.target_nodes,-np.inf)
        best=np.full(a.target_nodes,np.inf); representative=np.full(a.target_nodes,-1,np.int64)
        for lo in range(0,n,a.chunk_size):
            hi=min(n,lo+a.chunk_size); lids=np.asarray(lineage[lo:hi]); raw=np.asarray(emb[lo:hi],np.float32)
            local=np.empty(hi-lo,np.int32); dd=np.empty(hi-lo,np.float32)
            for lid,centers in enumerate(models):
                mask=lids==lid
                if not mask.any(): continue
                similarities = cosine_similarity_to_centers(raw[mask], centers)
                labels=np.argmax(similarities,axis=1).astype(np.int32); node=labels+offsets[lid]; local[mask]=node
                dd[mask]=1.0-similarities[np.arange(len(labels)),labels]
            assign[lo:hi]=local; dist[lo:hi]=dd
            st=stage_values[np.asarray(stage_ids[lo:hi],dtype=int)]
            np.add.at(sizes,local,1); np.add.at(sums,local,raw); np.add.at(dist_sum,local,dd); np.add.at(stage_sum,local,st); np.add.at(stage_sq,local,st*st)
            np.minimum.at(stage_min,local,st); np.maximum.at(stage_max,local,st)
            for j in range(hi-lo):
                nd=int(local[j])
                if dd[j]<best[nd]: best[nd]=dd[j]; representative[nd]=lo+j
            if hi%1000000<a.chunk_size or hi==n: log(f"assigned {hi:,}/{n:,} cells")
        node_emb=(sums/sizes[:,None]).astype(np.float32); np.save(tmp/"node_embeddings.npy",node_emb)
        ctsum,composition,ctqc=aggregate_celltypes(a.metadata,cell_ids,assign,a.target_nodes,a.metadata_chunk_size)
        ctmap={x["node_id"]:x for x in ctsum}; rows=[]
        for lid,name in enumerate(names):
            for node in range(offsets[lid],offsets[lid+1]):
                mean=stage_sum[node]/sizes[node]; var=max(0.,stage_sq[node]/sizes[node]-mean*mean)
                row={"node_id":node,"lineage":name,"lineage_id":lid,"n_cells":int(sizes[node]),
                     "mean_stage":mean,"stage_std":math.sqrt(var),"min_stage":stage_min[node],"max_stage":stage_max[node],
                     "representative_idx":int(representative[node]),"representative_cell_id":str(cell_ids[representative[node]]),
                     "mean_assignment_distance":float(dist_sum[node]/sizes[node])}
                row.update(ctmap[node]); rows.append(row)
        pd.DataFrame(rows).to_parquet(tmp/"nodes.parquet",index=False)
        pd.DataFrame(composition).to_parquet(tmp/"node_celltype_composition.parquet",index=False)
        np.savez_compressed(tmp/"node_members.npz",cell_to_node=np.asarray(assign))
        config=vars(a).copy(); config={k:str(v) if isinstance(v,Path) else v for k,v in config.items()}; config.update(ctqc)
        (tmp/"run_config.json").write_text(json.dumps(config,indent=2,ensure_ascii=False))
        del assign,dist
        if out.exists(): shutil.rmtree(out)
        tmp.rename(out); log(f"done: {out}")
    except Exception:
        log(f"failed; partial output kept at {tmp}"); r