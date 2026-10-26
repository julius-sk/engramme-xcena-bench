# -*- coding: utf-8 -*-
"""
softmax_cost.py — measure what the softmax actually costs next to the scan.

Times the four stages of one attractor iteration separately, at the same scale the
retrieval runs at, so the softmax share is a measured number rather than an argument.

    python3 softmax_cost.py                 # M = 1e6
    python3 softmax_cost.py --M 4000000     # scale it up: the share does not move
"""
import argparse, os, time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "96")
import numpy as np


def med(f, reps=5):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2]


ap = argparse.ArgumentParser()
ap.add_argument("--M", type=int, default=1_000_000)
ap.add_argument("--n", type=int, default=1024)
ap.add_argument("--k", type=int, default=64)
a = ap.parse_args()
n, M, k, beta = a.n, a.M, a.k, 8.0

print(f"M = {M:,}   n = {n}   k = {k}")
K = np.asfortranarray(np.random.default_rng(0).standard_normal((n, M), dtype=np.float32))
xi = np.random.default_rng(1).standard_normal(n).astype(np.float32)
xi /= np.linalg.norm(xi)

s = K.T @ xi                                    # warm up / produce scores
idx = np.argpartition(s, -k)[-k:]

t_gemv = med(lambda: K.T @ xi)
t_topk = med(lambda: np.argpartition(s, -k)[-k:])
t_soft_k = med(lambda: (lambda e: e / e.sum())(np.exp(beta * (s[idx] - s[idx].max()))), reps=999)
t_soft_M = med(lambda: (lambda e: e / e.sum())(np.exp(beta * (s - s.max()))))

flops_gemv = 2 * n * M
bytes_gemv = 4 * n * M                          # fp32 in memory

print(f"\n{'stage':<34}{'time':>12}{'share of scan':>16}")
print(f"{'-'*62}")
print(f"{'GEMV  s = K^T xi':<34}{t_gemv*1e3:>10.2f} ms{'100.000 %':>16}")
print(f"{'top-k select (over all M)':<34}{t_topk*1e3:>10.2f} ms{t_topk/t_gemv*100:>14.3f} %")
print(f"{'softmax over top-k (k=%d)' % k:<34}{t_soft_k*1e6:>10.2f} us{t_soft_k/t_gemv*100:>14.5f} %")
print(f"{'softmax over ALL M (worst case)':<34}{t_soft_M*1e3:>10.2f} ms{t_soft_M/t_gemv*100:>14.3f} %")

print(f"\nper-column work: GEMV does {2*n} flop on every one of {M:,} columns;")
print(f"softmax touches {k} of them.  ratio = {k}/{M} = {k/M:.2e}")
print(f"\nanalytic check (M cancels, so the share is independent of memory count):")
print(f"  full-softmax share  = c/(2n) with c~23  ->  {23/(2*n)*100:.2f} %")
print(f"  even charging exp() 100 flop ->            {100/(2*n)*100:.2f} %")
print(f"  measured scan       = {flops_gemv/t_gemv/1e9:.1f} GFLOP/s, "
      f"{bytes_gemv/t_gemv/1e9:.1f} GB/s")
