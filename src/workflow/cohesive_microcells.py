"""Embedding-only microcells with local distance limits and no forced assignment."""
import heapq

import numpy as np


def cohesive_partition(vectors, target_size, min_size, max_size, k=30,
                       radius_factor=2.0, mutual=False):
    """Grow seed-bounded graph regions; retain small groups when merging is unsafe.

    target_size is the growth target, max_size is a hard merge limit, and
    min_size triggers optional compactness-checked merging. Every cell is
    retained, including isolated cells as singleton groups. No labels are used.
    """
    import faiss

    if not 1 <= min_size <= target_size <= max_size:
        raise ValueError('require 1 <= min_size <= target_size <= max_size')
    if k < 1 or not np.isfinite(radius_factor) or radius_factor <= 0:
        raise ValueError('k and radius_factor must be positive and finite')
    z = np.array(vectors, dtype=np.float32, copy=True)
    if z.ndim != 2 or not np.isfinite(z).all():
        raise ValueError('embeddings must be a finite 2D matrix')
    norms = np.linalg.norm(z, axis=1)
    if np.any(norms <= 0):
        raise ValueError('cosine distance requires non-zero embeddings')
    z /= norms[:, None]
    n = len(z)
    if n <= 1:
        return np.arange(n, dtype=np.int32)

    index = faiss.IndexFlatIP(z.shape[1])
    index.add(z)
    similarities, indices = index.search(z, min(k + 1, n))
    neighbors, distances = [], []
    for i, (ids, sims) in enumerate(zip(indices, similarities)):
        # Explicitly remove self: duplicate vectors need not return self first.
        keep = (ids >= 0) & (ids != i)
        neighbors.append(ids[keep][:k])
        distances.append(np.maximum(0., 1. - sims[keep][:k]))
    # Near-neighbor scales avoid using the far edge of a large kNN neighborhood.
    scales = np.array([max(float(np.median(d[:min(5, len(d))])), 1e-6)
                       for d in distances])
    neighbor_sets = [set(ids.tolist()) for ids in neighbors] if mutual else None
    adj = [set() for _ in range(n)]
    for i, (ids, ds) in enumerate(zip(neighbors, distances)):
        for j, distance in zip(ids, ds):
            j = int(j)
            if mutual and i not in neighbor_sets[j]:
                continue
            if distance <= radius_factor * min(scales[i], scales[j]):
                adj[i].add(j)
                adj[j].add(i)

    labels = np.full(n, -1, dtype=np.int32)
    groups = []
    for seed in np.argsort(scales, kind='stable'):
        seed = int(seed)
        if labels[seed] >= 0:
            continue
        group_id = len(groups)
        members = [seed]
        labels[seed] = group_id
        center_sum = z[seed].astype(np.float64)
        heap, queued = [], {seed}

        def enqueue(node):
            for other in sorted(adj[node]):
                if labels[other] < 0 and other not in queued:
                    queued.add(other)
                    distance = max(0., 1. - float(z[seed] @ z[other]))
                    heapq.heappush(heap, (distance, other))

        enqueue(seed)
        while heap and len(members) < target_size:
            seed_distance, node = heapq.heappop(heap)
            if labels[node] >= 0:
                continue
            limit = radius_factor * min(scales[seed], scales[node])
            if seed_distance > limit:
                continue
            center = center_sum / max(np.linalg.norm(center_sum), 1e-12)
            if max(0., 1. - float(z[node] @ center)) > limit:
                continue
            labels[node] = group_id
            members.append(node)
            center_sum += z[node]
            enqueue(node)
        groups.append(members)

    # Only graph-adjacent small groups can merge. All members must satisfy the
    # resulting center-distance limit; a single bridging edge is insufficient.
    for g in sorted(range(len(groups)), key=lambda i: (len(groups[i]), i)):
        members = groups[g]
        if not members or len(members) >= min_size:
            continue
        candidates = {int(labels[j]) for i in members for j in adj[i]} - {g}
        center = z[members].mean(axis=0)
        center /= max(np.linalg.norm(center), 1e-12)
        ranked = []
        for h in candidates:
            if not groups[h] or len(members) + len(groups[h]) > max_size:
                continue
            other = z[groups[h]].mean(axis=0)
            other /= max(np.linalg.norm(other), 1e-12)
            ranked.append((max(0., 1. - float(center @ other)), h))
        for _, h in sorted(ranked):
            merged = members + groups[h]
            centroid = z[merged].mean(axis=0)
            centroid /= max(np.linalg.norm(centroid), 1e-12)
            ds = np.maximum(0., 1. - z[merged] @ centroid)
            limits = radius_factor * np.minimum(scales[merged], np.median(scales[merged]))
            if np.all(ds <= limits):
                labels[members] = h
                groups[h] = merged
                groups[g] = []
                break
    return np.unique(labels, return_inverse=True)[1].astype(np.int32)
