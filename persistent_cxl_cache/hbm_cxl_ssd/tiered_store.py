"""tiered_store.py - GNN graph structure and features on GPU HBM -> CXL DRAM cache -> SSD.

Placement (the method; GIDS is the baseline):

    graph structure  CSC (indptr, indices) on the XCENA device by default (graph_on="card"):
                     the device DRAM caches hot blocks, the SSD holds the full arrays.
                     graph_on="split": indptr in HBM, indices on the device.
                     graph_on="host":  both pinned in host DRAM, as GIDS does.
    feature table    on the XCENA device; a static HBM cache holds the hottest rows.
                     GIDS instead reads the SSD from GPU threads via BaM (4 KB lines), with a
                     GPU software cache and a constant CPU buffer.

The device DRAM and its SSD look like one flat memory to the host; the device moves 512 KB
pages between them by itself. Software controls how requests reach it, for indptr, indices
and features alike:

    GPU: reduce        drop HBM hits, dedupe, sort by address
    host: prefetch     prefetchMemory once per distinct device page (async, overlaps SSD reads)
    host: gather       index_select from the device into pinned DRAM
    GPU: H2D + scatter copy to the GPU and put the values back in request order

Not here yet: a look-ahead cache policy (GIDS's window buffering is the counterpart), GPU
zero-copy reads of the device, overlap of batch i+1's gather with batch i's compute.
"""
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "im_loader"))
import im_store as im  # noqa: E402



def _pin(t):
    """Page-locked host memory makes H2D copies async and faster (no-op without CUDA)."""
    return t.pin_memory() if torch.cuda.is_available() else t


def _sync(device):
    if device.type == "cuda":
        torch.cuda.current_stream(device).synchronize()


# ---------------------------------------------------------------------------------------------
# SSD counters: the only reliable proof that a step touched the SSD.
class SsdCounter:
    def __init__(self, device=0):
        self.cmd = ["xcena_cli", "im", "get-smart", str(device)]

    def read(self):
        """(nvme_read_bytes, nvme_write_bytes) or (None, None) if xcena_cli is unavailable."""
        try:
            out = subprocess.run(self.cmd, capture_output=True, text=True, timeout=10).stdout
        except (OSError, subprocess.TimeoutExpired):
            return None, None
        rd = wr = None
        in_bytes = False
        for line in out.splitlines():
            if "Bytes:" in line:
                in_bytes = True
                continue
            if in_bytes and ":" in line:
                key, val = line.split(":", 1)
                if key.strip() == "nvme_read":
                    rd = int(val.split()[0])
                elif key.strip() == "nvme_write":
                    wr = int(val.split()[0])
        return rd, wr


class Stats:
    def __init__(self):
        self.reset()

    def reset(self):
        self.feat_rows = 0        # feature rows requested
        self.feat_hbm_hits = 0    # served from the HBM cache
        self.feat_pages = 0       # distinct 512 KB pages requested from the card for features
        self.ptr_entries = 0      # indptr entries read (graph_on card / host)
        self.ptr_pages = 0        # distinct device pages requested for indptr
        self.col_entries = 0      # indices entries read
        self.col_pages = 0        # distinct device pages requested for indices
        self.t_sample = 0.0
        self.t_feat_card = 0.0    # prefetch + gather from the card
        self.t_feat_h2d = 0.0

    def summary(self):
        hit = self.feat_hbm_hits / max(self.feat_rows, 1)
        return (f"HBM hit {hit:.1%} | feature pages {self.feat_pages} | indptr entries {self.ptr_entries} "
                f"pages {self.ptr_pages} | indices entries {self.col_entries} pages {self.col_pages} | "
                f"sample {self.t_sample * 1e3:.0f} ms, "
                f"card gather {self.t_feat_card * 1e3:.0f} ms, H2D {self.t_feat_h2d * 1e3:.0f} ms")


# ---------------------------------------------------------------------------------------------
class TieredFeatures:
    """Feature table: hot rows in HBM, everything on the card (CXL cache + SSD)."""

    def __init__(self, feat_path, device, hot_ids=None, prefetch=True, stats=None):
        self.device = device
        self.prefetch = prefetch
        self.stats = stats or Stats()

        self.feat = im.load_npy(feat_path)                 # numpy view of card memory, (N, D)
        self.feat_t = torch.from_numpy(self.feat)          # zero-copy CPU tensor over the card
        self.num_nodes, self.dim = self.feat.shape

        # HBM tier: slot[v] = row in self.hbm, or -1. int32 keeps it at 4 B per node.
        self.slot = torch.full((self.num_nodes,), -1, dtype=torch.int32, device=device)
        if hot_ids is not None and hot_ids.numel() > 0:
            hot_ids = torch.sort(hot_ids.to(torch.int64).cpu()).values
            self.hbm = self.feat_t.index_select(0, hot_ids).to(device)
            self.slot[hot_ids.to(device)] = torch.arange(hot_ids.numel(), dtype=torch.int32, device=device)
        else:
            self.hbm = torch.empty((0, self.dim), dtype=self.feat_t.dtype, device=device)

        self._pinned = _pin(torch.empty((0, self.dim), dtype=self.feat_t.dtype))

    def _pinned_rows(self, n):
        if self._pinned.shape[0] < n:
            self._pinned = _pin(torch.empty((max(n, 2 * self._pinned.shape[0]), self.dim),
                                            dtype=self.feat_t.dtype))
        return self._pinned[:n]

    def gather(self, ids):
        """ids: int64 CUDA tensor of global node ids. Returns (len(ids), D) on the GPU."""
        st = self.stats
        out = torch.empty((ids.numel(), self.dim), dtype=self.feat_t.dtype, device=self.device)

        # 1. HBM tier
        slot = self.slot[ids]
        hit = slot >= 0
        if self.hbm.shape[0] > 0:
            out[hit] = self.hbm[slot[hit].long()]
        miss_pos = torch.nonzero(~hit, as_tuple=True)[0]
        st.feat_rows += ids.numel()
        st.feat_hbm_hits += ids.numel() - miss_pos.numel()
        if miss_pos.numel() == 0:
            return out

        # 2. request reduction on the GPU: dedupe + sort (sorted ids = ascending addresses)
        uniq, inv = torch.unique(ids[miss_pos], sorted=True, return_inverse=True)
        uniq_cpu = uniq.cpu()

        # 3. card tier: one async prefetch per distinct page, then gather into pinned DRAM
        t0 = time.perf_counter()
        if self.prefetch:
            st.feat_pages += im.prefetch_rows(self.feat, uniq_cpu.numpy())
        buf = self._pinned_rows(uniq_cpu.numel())
        torch.index_select(self.feat_t, 0, uniq_cpu, out=buf)
        st.t_feat_card += time.perf_counter() - t0

        # 4. H2D and scatter back to batch order
        t0 = time.perf_counter()
        rows = buf.to(self.device, non_blocking=True)
        out[miss_pos] = rows[inv]
        _sync(self.device)
        st.t_feat_h2d += time.perf_counter() - t0
        return out


# ---------------------------------------------------------------------------------------------
class TieredGraph:
    """Graph structure as CSC: indptr (N+1) and indices (E). The list of node v,
    indices[indptr[v]:indptr[v+1]], holds v's in-neighbors, the nodes that send messages to v
    (prepare_ogbn.py --format csc). A symmetric graph gives the same arrays as CSR.

    graph_on="card"  (default) indptr and indices both on the device (CXL cache + SSD).
                     Each hop reads indptr[v], indptr[v+1] for the frontier, then only the
                     sampled indices entries; both with page prefetch.
    graph_on="split" indptr in GPU HBM, indices on the device.
    graph_on="host"  both in pinned host DRAM, the GIDS placement (baseline).
    """

    PLACEMENTS = ("card", "split", "host")

    def __init__(self, indptr_path, indices_path, device, prefetch=True, stats=None, graph_on="card"):
        if graph_on not in self.PLACEMENTS:
            raise ValueError(graph_on)
        self.device = device
        self.prefetch = prefetch
        self.stats = stats or Stats()
        self.graph_on = graph_on
        self.indptr_path = indptr_path
        self.indptr = self.indptr_t = self.indptr_gpu = None
        if graph_on == "card":
            self.indptr = im.load_npy(indptr_path)
            self.indptr_t = torch.from_numpy(self.indptr)
        elif graph_on == "split":
            self.indptr_gpu = torch.from_numpy(np.load(indptr_path)).to(device=device, dtype=torch.int64)
        else:
            self.indptr_t = _pin(torch.from_numpy(np.load(indptr_path)))
        if graph_on == "host":
            self.indices = None
            self.indices_t = _pin(torch.from_numpy(np.load(indices_path)))
        else:
            self.indices = im.load_npy(indices_path)
            self.indices_t = torch.from_numpy(self.indices)
        self.num_nodes = int(np.load(indptr_path, mmap_mode="r").shape[0]) - 1
        self._g2l = torch.full((self.num_nodes,), -1, dtype=torch.int64, device=device)

    def degree(self):
        """In-degree of every node, on the CPU (read from the indptr file)."""
        p = torch.from_numpy(np.load(self.indptr_path).astype(np.int64, copy=False))
        return p[1:] - p[:-1]

    def _read(self, dev_arr, t, idx, kind):
        """t[idx] on the GPU. dev_arr is the device-resident numpy array (None = host memory)."""
        st = self.stats
        uniq, inv = torch.unique(idx, sorted=True, return_inverse=True)
        uniq_cpu = uniq.cpu()
        if dev_arr is not None and self.prefetch:
            pages = im.prefetch_rows(dev_arr, uniq_cpu.numpy())
            if kind == "indptr":
                st.ptr_pages += pages
            else:
                st.col_pages += pages
        vals = _pin(t.index_select(0, uniq_cpu)).to(self.device, non_blocking=True)
        if kind == "indptr":
            st.ptr_entries += uniq_cpu.numel()
        else:
            st.col_entries += uniq_cpu.numel()
        return vals.to(torch.int64)[inv]

    def _ranges(self, frontier):
        """(start, degree) of each frontier node's list."""
        if self.graph_on == "split":
            start = self.indptr_gpu[frontier]
            return start, self.indptr_gpu[frontier + 1] - start
        both = self._read(self.indptr if self.graph_on == "card" else None, self.indptr_t,
                          torch.cat((frontier, frontier + 1)), "indptr")
        start, end = both[: frontier.numel()], both[frontier.numel():]
        return start, end - start

    def sample(self, seeds, fanouts, generator=None):
        """Layer-wise uniform in-neighbor sampling (with replacement when degree > fanout).

        Returns (n_id, edge_index): n_id are global ids with the seeds first; edge_index is
        (2, E) in local ids, row 0 = in-neighbor (source of the message), row 1 = target, which is
        the PyG convention for message passing toward the seeds.
        """
        t0 = time.perf_counter()
        dev = self.device
        seeds = seeds.to(dev, torch.int64)
        frontier = seeds
        visited = [seeds]
        src_all, dst_all = [], []
        for fanout in fanouts:
            if frontier.numel() == 0:
                break
            start, deg = self._ranges(frontier)
            j = torch.arange(fanout, device=dev).expand(frontier.numel(), fanout)
            take_all = (deg <= fanout).unsqueeze(1)
            rnd = torch.rand((frontier.numel(), fanout), device=dev, generator=generator)
            offs = torch.where(take_all, j, (rnd * deg.unsqueeze(1)).long())
            valid = j < torch.clamp(deg, max=fanout).unsqueeze(1)
            eidx = (start.unsqueeze(1) + offs)[valid]                       # entries to read
            tgt = frontier.unsqueeze(1).expand(-1, fanout)[valid]
            nbr = self._read(self.indices, self.indices_t, eidx, "indices") if eidx.numel() else eidx
            src_all.append(nbr)
            dst_all.append(tgt)
            seen = torch.cat(visited)
            nxt = torch.unique(nbr)
            nxt = nxt[~torch.isin(nxt, seen)]
            visited.append(nxt)
            frontier = nxt

        n_id = torch.cat(visited)                                           # seeds first, unique
        g2l = self._g2l
        g2l[n_id] = torch.arange(n_id.numel(), device=dev)
        src = torch.cat(src_all) if src_all else torch.empty(0, dtype=torch.int64, device=dev)
        dst = torch.cat(dst_all) if dst_all else torch.empty(0, dtype=torch.int64, device=dev)
        edge_index = torch.stack((g2l[src], g2l[dst]))
        g2l[n_id] = -1
        _sync(self.device)
        self.stats.t_sample += time.perf_counter() - t0
        return n_id, edge_index


def graph_files(data_dir):
    """(indptr path, indices path) in data_dir: indptr.npy/indices.npy, or the older
    rowptr.npy/colind.npy names."""
    for a, b in (("indptr.npy", "indices.npy"), ("rowptr.npy", "colind.npy")):
        if os.path.exists(os.path.join(data_dir, a)):
            return os.path.join(data_dir, a), os.path.join(data_dir, b)
    raise FileNotFoundError(f"no indptr.npy / rowptr.npy in {data_dir}")


def hot_by_degree(graph, hbm_bytes, row_bytes):
    """Pick the highest-degree nodes that fit in hbm_bytes (a common static cache policy)."""
    k = min(int(hbm_bytes // row_bytes), graph.num_nodes)
    if k <= 0:
        return torch.empty(0, dtype=torch.int64)
    return torch.topk(graph.degree(), k).indices
