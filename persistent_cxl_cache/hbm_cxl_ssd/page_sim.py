"""page_sim.py - how big must the CXL cache be? Offline page-level simulation, no card needed.

A CXL memory device that caches an SSD (e.g. XCENA InfiniteMemory) moves whole pages (512 KB on
MX1P; 64 KB announced). BaM/GIDS read the SSD in 4 KB cache lines, so the 4K page size models
that granularity for comparison. For GNN mini-batch training what matters is how many DISTINCT pages a
batch touches, and how many of those are still cached from earlier batches. Both depend on how
node ids are laid out in memory (the ordering) and on how batches are formed. This script
replays real sampled batches and reports, for each ordering and page size:

  * feature pages and col_idx pages touched per batch, and the read amplification
    (bytes moved / bytes used);
  * the miss-ratio curve: SSD bytes per batch as a function of cache size, from one pass of
    exact LRU stack distances (Mattson et al.); within a batch pages are touched least recently
    used first (the order inside a batch is not known, this is one valid order). Checked against
    a direct LRU simulation in tests/test_page_sim.py;
  * the smallest cache that keeps SSD reads under --target-mb per batch.

Layout model: the hottest feature rows (--hbm-gb, by degree) are in GPU HBM. The feature table
is on the storage tier, stored in the given node ORDER (row of node v at position pos[v]). The
graph structure is in host memory, as in GIDS (--colind-on host, default); with --colind-on card
col_idx (int64) is on the device too, each adjacency list at its position in the reordered CSR,
and feature pages and col_idx pages share one LRU cache.

    python page_sim.py --data DIR --orders original,degree,random,rcm --batching random
    python page_sim.py --data DIR --orders original --colind-on card       # col_idx on the device too
    python page_sim.py --data DIR --orders original --batching contiguous   # partition-like batches
    python page_sim.py --data DIR --orders file --perm my_order.npy        # e.g. a METIS order

DIR holds rowptr.npy, colind.npy, feat.npy (only its shape/dtype is read), optional train_idx.npy
(prepare_ogbn.py writes this layout). Sampling: uniform, with replacement when degree > fanout,
same as tiered_store.TieredGraph.sample.
"""
import argparse
import os
import time

import numpy as np

PAGE_SIZES = {"512K": 512 * 1024, "64K": 64 * 1024, "4K": 4 * 1024}


# ---------------------------------------------------------------------------------------------
# sampling (original node ids)
def sample_batch(rowptr, colind, seeds, fanouts, rng, mark):
    """Returns (n_id, edge_targets, edge_offsets): all nodes whose features are needed, and for
    every col_idx entry read, the node whose list it belongs to and its offset in that list."""
    frontier = seeds
    visited = [seeds]
    mark[seeds] = 1
    tg_all, off_all = [], []
    for k in fanouts:
        if frontier.size == 0:
            break
        start = rowptr[frontier]
        deg = rowptr[frontier + 1] - start
        cnt = np.minimum(deg, k)
        rep = np.repeat(np.arange(frontier.size), cnt)
        if rep.size == 0:
            break
        first = np.repeat(np.cumsum(cnt) - cnt, cnt)
        within = np.arange(rep.size) - first
        rnd = (rng.random(rep.size) * deg[rep]).astype(np.int64)
        off = np.where(deg[rep] <= k, within, rnd)
        tgt = frontier[rep]
        nbr = np.asarray(colind[start[rep] + off])
        tg_all.append(tgt)
        off_all.append(off)
        nxt = np.unique(nbr)
        nxt = nxt[mark[nxt] == 0]
        mark[nxt] = 1
        visited.append(nxt)
        frontier = nxt
    n_id = np.concatenate(visited)
    mark[n_id] = 0
    cat = (lambda a: np.concatenate(a) if a else np.empty(0, np.int64))
    return n_id, cat(tg_all), cat(off_all)


# ---------------------------------------------------------------------------------------------
# orderings: order[i] = node placed at position i
def make_order(name, rowptr, colind, rng, perm_path=None):
    n = rowptr.size - 1
    if name == "original":
        return np.arange(n, dtype=np.int64)
    if name == "random":
        return rng.permutation(n).astype(np.int64)
    if name == "degree":
        return np.argsort(-(rowptr[1:] - rowptr[:-1]), kind="stable").astype(np.int64)
    if name == "rcm":
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import reverse_cuthill_mckee
        a = csr_matrix((np.ones(colind.size, dtype=np.int8), np.asarray(colind), rowptr), shape=(n, n))
        return reverse_cuthill_mckee(a, symmetric_mode=True).astype(np.int64)
    if name == "file":
        order = np.load(perm_path).astype(np.int64)
        assert order.size == n and np.array_equal(np.sort(order), np.arange(n)), "--perm is not a permutation"
        return order
    raise ValueError(name)


# ---------------------------------------------------------------------------------------------
class StackDistance:
    """Exact LRU stack distances over a fixed page universe.

    Each batch's pages are accessed least-recently-used first (first touches first). With that
    order the distance of the i-th access is (#pages last used after this page's last use) + i,
    so a whole batch is computed with one sort instead of a per-access loop.
    """

    def __init__(self, num_pages):
        self.last = np.full(num_pages, -1, dtype=np.int64)  # global access time of last use
        self.t = 0

    def access(self, pages):
        """pages: unique page ids of one batch. Returns distances (-1 = first touch)."""
        tp = self.last[pages]
        o = np.argsort(tp, kind="stable")
        pages, tp = pages[o], tp[o]
        live = np.sort(self.last[self.last >= 0])
        d = live.size - np.searchsorted(live, tp, side="right") + np.arange(pages.size)
        d[tp < 0] = -1
        self.last[pages] = self.t + np.arange(pages.size)
        self.t += pages.size
        return d


def misses_for(dists, cache_pages):
    """Number of misses at each cache size (in pages) for the recorded distances."""
    d = np.sort(dists[dists >= 0])
    cold = int((dists < 0).sum())
    return np.array([cold + d.size - np.searchsorted(d, c, side="left") for c in cache_pages])


# ---------------------------------------------------------------------------------------------
def run(args):
    t0 = time.time()
    rowptr = np.load(os.path.join(args.data, "rowptr.npy"))
    colind = np.load(os.path.join(args.data, "colind.npy"), mmap_mode="r")
    fmeta = np.load(os.path.join(args.data, "feat.npy"), mmap_mode="r")
    n, dim = fmeta.shape
    row_bytes = args.feat_bytes or dim * fmeta.dtype.itemsize
    deg = rowptr[1:] - rowptr[:-1]
    tpath = os.path.join(args.data, "train_idx.npy")
    train = np.load(tpath).astype(np.int64) if os.path.exists(tpath) else np.arange(n, dtype=np.int64)
    fanouts = [int(x) for x in args.fanouts.split(",")]
    hot_k = min(int(args.hbm_gb * (1 << 30) // row_bytes), n)
    in_hbm = np.zeros(n, dtype=bool)
    if hot_k > 0:
        in_hbm[np.argpartition(-deg, hot_k - 1)[:hot_k]] = True
    feat_bytes_total = n * row_bytes
    col_bytes_total = colind.size * 8
    print(f"N={n} E={colind.size} row={row_bytes} B  features {feat_bytes_total / 2**30:.2f} GiB, "
          f"col_idx {col_bytes_total / 2**30:.2f} GiB, HBM rows {hot_k} ({hot_k / n:.1%})  "
          f"[{time.time() - t0:.0f} s]", flush=True)

    cache_gb = [float(x) for x in args.cache_gb.split(",")]
    rows = []
    mark = np.zeros(n, dtype=np.uint8)
    for oname in args.orders.split(","):
        rng = np.random.default_rng(args.seed)
        t1 = time.time()
        order = make_order(oname, rowptr, colind, rng, args.perm)
        pos = np.empty(n, dtype=np.int64)
        pos[order] = np.arange(n)
        new_start = np.empty(n, dtype=np.int64)                    # start of v's list, reordered CSR
        new_start[order] = np.concatenate(([0], np.cumsum(deg[order])[:-1]))
        print(f"\n[{oname}] ordering ready [{time.time() - t1:.0f} s]", flush=True)

        # seeds per batch
        if args.batching == "random":
            seeds_all = rng.permutation(train)
        else:  # contiguous: seeds sorted by position, chunks visited in random order
            s = train[np.argsort(pos[train], kind="stable")]
            nchunk = s.size // args.batch
            chunks = rng.permutation(nchunk)
            seeds_all = np.concatenate([s[c * args.batch:(c + 1) * args.batch] for c in chunks])
        total = args.warmup + args.batches
        nfull = max(seeds_all.size // args.batch, 1)

        batches = []  # (feature byte positions, col_idx byte positions, useful bytes)
        for b in range(total):
            lo = (b % nfull) * args.batch
            seeds = np.unique(seeds_all[lo:lo + args.batch])
            n_id, tgt, off = sample_batch(rowptr, colind, seeds, fanouts, rng, mark)
            fn = n_id[~in_hbm[n_id]]
            fpos = pos[fn] * row_bytes
            cpos = (new_start[tgt] + off) * 8 if args.colind_on == "card" else np.empty(0, np.int64)
            batches.append((fpos, cpos, fn.size * row_bytes + cpos.size * 8))
        print(f"[{oname}] sampled {total} batches [{time.time() - t1:.0f} s]", flush=True)

        for pname in args.pages.split(","):
            page = PAGE_SIZES[pname]
            pf = -(-feat_bytes_total // page)
            pc = -(-col_bytes_total // page) if args.colind_on == "card" else 0
            sd = StackDistance(pf + pc)
            dists, fpg, cpg, useful = [], [], [], []
            for b, (fpos, cpos, use) in enumerate(batches):
                fp = np.unique(np.concatenate((fpos // page, (fpos + row_bytes - 1) // page)))
                cp = np.unique(cpos // page) + pf
                d = sd.access(np.concatenate((fp, cp)))
                if b >= args.warmup:
                    dists.append(d)
                    fpg.append(fp.size)
                    cpg.append(cp.size)
                    useful.append(use)
            dists = np.concatenate(dists)
            nb = args.batches
            touched = (np.mean(fpg) + np.mean(cpg)) * page
            cache_pages = [int(g * (1 << 30) // page) for g in cache_gb]
            ssd_mb = misses_for(dists, cache_pages) * page / nb / 1e6
            # smallest cache meeting the target, by bisection over a fine grid
            grid = np.unique(np.geomspace(1, pf + pc, 400).astype(np.int64))
            miss_grid = misses_for(dists, grid) * page / nb / 1e6
            ok = np.nonzero(miss_grid <= args.target_mb)[0]
            need = f"{grid[ok[0]] * page / 2**30:.2f} GiB" if ok.size else "> whole dataset"
            rows.append((oname, pname, np.mean(fpg), np.mean(cpg), touched / 1e6, np.mean(useful) / 1e6,
                         touched / max(np.mean(useful), 1), ssd_mb, need))

    # report
    print(f"\nbatch {args.batch}, fanouts {fanouts}, batching {args.batching}, {args.batches} batches "
          f"after {args.warmup} warm-up; SSD MB/batch at cache sizes (GiB) {cache_gb}; "
          f"target {args.target_mb} MB/batch")
    hdr = f"{'order':<9}{'page':<6}{'feat pg':>9}{'col pg':>8}{'touch MB':>10}{'use MB':>8}{'ampl':>7}  " \
          + "".join(f"{g:>8g}" for g in cache_gb) + "   cache for target"
    print(hdr)
    for o, p, fp, cp, tmb, umb, amp, ssd, need in rows:
        print(f"{o:<9}{p:<6}{fp:>9.0f}{cp:>8.0f}{tmb:>10.0f}{umb:>8.1f}{amp:>7.0f}  "
              + "".join(f"{x:>8.0f}" for x in ssd) + f"   {need}")
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["order", "page", "feat_pages", "col_pages", "touched_MB", "useful_MB", "amplification"]
                       + [f"ssd_MB_at_{g}GiB" for g in cache_gb] + ["cache_for_target"])
            for o, p, fp, cp, tmb, umb, amp, ssd, need in rows:
                w.writerow([o, p, f"{fp:.0f}", f"{cp:.0f}", f"{tmb:.1f}", f"{umb:.2f}", f"{amp:.1f}"]
                           + [f"{x:.1f}" for x in ssd] + [need])
        print(f"wrote {args.csv}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", required=True)
    p.add_argument("--orders", default="original,degree,random")
    p.add_argument("--perm", help="order .npy for --orders file (order[i] = node at position i)")
    p.add_argument("--batching", choices=["random", "contiguous"], default="random")
    p.add_argument("--pages", default="4K,64K,512K", help="4K = BaM/GIDS cache-line granularity")
    p.add_argument("--colind-on", choices=["host", "card"], default="host",
                   help="host = graph structure in host memory, as GIDS (only features hit the device)")
    p.add_argument("--fanouts", default="10,10,10")
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--batches", type=int, default=200)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--hbm-gb", type=float, default=0.0)
    p.add_argument("--feat-bytes", type=int, default=0, help="override bytes per feature row (e.g. 256 for fp16)")
    p.add_argument("--cache-gb", default="1,2,4,8,16,32,64")
    p.add_argument("--target-mb", type=float, default=50.0, help="acceptable SSD MB read per batch")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--csv")
    run(p.parse_args())


if __name__ == "__main__":
    main()
