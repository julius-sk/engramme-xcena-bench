"""CPU-only check of tiered_store: a fake im_store backed by numpy replaces the card.

    python tests/test_tiered.py
Checks that gather() returns exactly feat[ids] (HBM hits and misses, duplicates) and that every
sampled edge exists in the graph, seeds come first, n_id is unique, fanouts are respected.
"""
import os
import sys
import tempfile
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
fake = types.ModuleType("im_store")
fake.PAGE = 512*1024
fake.issued = []
fake.load_npy = lambda path, **k: np.ascontiguousarray(np.load(path))
def prefetch_rows(arr, rows):
    pages = np.unique(np.asarray(rows, dtype=np.int64) * arr.strides[0] // fake.PAGE)
    fake.issued.append(pages.size); return int(pages.size)
fake.prefetch_rows = prefetch_rows
sys.modules["im_store"] = fake
sys.path.insert(0, os.path.dirname(HERE))
from tiered_store import TieredFeatures, TieredGraph, Stats, hot_by_degree

rng = np.random.default_rng(1); N, D = 5000, 64
deg = rng.integers(0, 40, N); deg[:5] = 0          # some isolated nodes
rowptr = np.zeros(N+1, np.int64); np.cumsum(deg, out=rowptr[1:])
colind = rng.integers(0, N, rowptr[-1]).astype(np.int64)
feat = rng.standard_normal((N, D)).astype(np.float32)
d = tempfile.mkdtemp()
for k, v in dict(rowptr=rowptr, colind=colind, feat=feat).items(): np.save(f"{d}/{k}.npy", v)

dev = torch.device("cpu"); st = Stats()
g = TieredGraph(f"{d}/rowptr.npy", f"{d}/colind.npy", dev, stats=st)
hot = hot_by_degree(g, 300 * D * 4, D * 4); assert hot.numel() == 300
f = TieredFeatures(f"{d}/feat.npy", dev, hot_ids=hot, stats=st)

# gather correctness, with duplicates, hits and misses
ids = torch.from_numpy(rng.integers(0, N, 4000)).long()
assert torch.equal(f.gather(ids), torch.from_numpy(feat)[ids]), "gather mismatch"
hits = np.isin(ids.numpy(), hot.numpy()).sum()
assert st.feat_hbm_hits == hits, (st.feat_hbm_hits, hits)

# sampling correctness
nbrs = [set(colind[rowptr[v]:rowptr[v+1]]) for v in range(N)]
fan = [5, 4, 3]
for trial in range(20):
    seeds = torch.from_numpy(rng.choice(N, 64, replace=False)).long()
    n_id, ei = g.sample(seeds, fan)
    assert torch.equal(n_id[:64], seeds)
    assert n_id.unique().numel() == n_id.numel(), "n_id not unique"
    assert (ei >= 0).all() and (ei < n_id.numel()).all()
    src, dst = n_id[ei[0]].numpy(), n_id[ei[1]].numpy()
    for s_, t_ in zip(src, dst): assert s_ in nbrs[t_], "sampled edge not in graph"
    # per-target count <= fanout of its hop; low-degree targets take all neighbors
    from collections import Counter
    c = Counter(dst.tolist())
    assert max(c.values()) <= max(fan)
    for t_ in seeds.numpy():
        exp = min(len(colind[rowptr[t_]:rowptr[t_+1]]), fan[0])
        assert c.get(int(t_), 0) == exp, (t_, c.get(int(t_)), exp)
print("OK  gather + sampling correct;", st.summary())
