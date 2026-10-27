"""Check page_sim: stack distances equal a direct LRU simulation; sampling and orderings valid.

    python tests/test_page_sim.py
"""
import os
import sys
from collections import Counter, OrderedDict  # noqa: F401

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from page_sim import StackDistance, misses_for, sample_batch, make_order

# 1. stack distances vs exact LRU on random batches with locality
rng = np.random.default_rng(3)
P, B = 3000, 120
batches = []
for b in range(B):
    centre = rng.integers(0, P)
    pg = np.unique(np.concatenate((rng.integers(0, P, 40), (centre + rng.integers(-150, 150, 120)) % P)))
    batches.append(pg)
sd = StackDistance(P)
d = np.concatenate([sd.access(pg) for pg in batches])
for C in (50, 200, 500, 1000, 2000, 3000):
    lru = OrderedDict(); miss = 0; last = {}; t = 0
    for pg in batches:
        # same within-batch order as StackDistance: least recently used first
        for p in sorted(pg, key=lambda q: last.get(q, -1)):
            last[p] = t; t += 1
            if p in lru: lru.move_to_end(p)
            else:
                miss += 1; lru[p] = 1
                if len(lru) > C: lru.popitem(last=False)
    est = misses_for(d, [C])[0]
    maxb = max(len(x) for x in batches)
    print(f"C={C:5d} exact {miss:6d} sim {est:6d} diff {est - miss:+5d} ({abs(est-miss)/miss:.1%})")
    assert est == miss, "stack distance differs from exact LRU"

full = [np.arange(200)] * 10
sd2 = StackDistance(200); d2 = np.concatenate([sd2.access(x) for x in full])
assert misses_for(d2, [20])[0] == 2000 and misses_for(d2, [200])[0] == 200, "full-scan case wrong"

# 2. sampling: every col_idx read points inside its node's list
N = 2000; deg = rng.integers(0, 30, N); rowptr = np.zeros(N+1, np.int64); np.cumsum(deg, out=rowptr[1:])
colind = rng.integers(0, N, rowptr[-1]).astype(np.int64)
mark = np.zeros(N, np.uint8)
n_id, tgt, off = sample_batch(rowptr, colind, rng.choice(N, 50, replace=False), [5, 5], rng, mark)
assert (off >= 0).all() and (off < deg[tgt]).all() and mark.sum() == 0 and np.unique(n_id).size == n_id.size
for name in ("original", "random", "degree", "rcm"):
    o = make_order(name, rowptr, colind, rng)
    assert np.array_equal(np.sort(o), np.arange(N)), name
print("OK  stack distance == exact LRU; sampling and orderings valid")
