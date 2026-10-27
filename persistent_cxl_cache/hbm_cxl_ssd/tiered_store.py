"""tiered_store.py - GNN feature table on GPU HBM -> CXL DRAM cache -> SSD.

Built to be compared with BaM / GIDS, so it uses the same split as GIDS:

    graph structure  pinned host memory (GIDS does the same); sampled on the GPU side.
                     Optional --colind-on card puts col_idx on the device instead.
    feature table    the storage tier. GIDS: GPU-initiated reads from the SSD via BaM, with a
                     GPU software cache and a constant CPU buffer for hot nodes.
                     Here: the XCENA device - its DRAM caches its SSD in hardware (512 KB pages),
                     plus a static HBM cache of hot rows.

So the only thing that differs from GIDS is the feature path:

    GPU: reduce        drop HBM hits, dedupe, sort by address
    host: prefetch     prefetchMemory once per distinct device page (async, overlaps SSD reads)
    host: gather       index_select from the device into a pinned DRAM buffer
    GPU: H2D + scatter copy the rows to the GPU and put them in batch order

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
        self.col_entries = 0      # col_idx entries read from the card
        self.col_pages = 0        # distinct pages requested for col_idx
        self.t_sample = 0.0
        self.t_feat_card = 0.0    # prefetch + gather from the card
        self.t_feat_h2d = 0.0

    def summary(self):
        hit = self.feat_hbm_hits / max(self.feat_rows, 1)
        return (f"HBM hit {hit:.1%} | feature pages {self.feat_pages} | col_idx entries "
                f"{self.col_entries} pages {self.col_pages} | sample {self.t_sample * 1e3:.0f} ms, "
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
    """Graph structure as CSR (rowptr, colind). With a symmetric graph CSR equals CSC, the
    in-neighbor format DGL/GIDS sample from.

    colind_on="host" (default, same placement as GIDS): colind in pinned host DRAM.
    colind_on="card": colind on the device; only the sampled entries are read, with page prefetch.
    rowptr is kept on the GPU for degree lookups in both cases.
    """

    def __init__(self, rowptr_path, colind_path, device, prefetch=True, stats=None, colind_on="host"):
        if colind_on not in ("host", "card"):
            raise ValueError(colind_on)
        self.device = device
        self.prefetch = prefetch
        self.stats = stats or Stats()
        self.colind_on = colind_on
        self.rowptr = torch.from_numpy(np.load(rowptr_path)).to(device=device, dtype=torch.int64)
        self.num_nodes = self.rowptr.numel() - 1
        if colind_on == "card":
            self.colind = im.load_npy(colind_path)
            self.colind_t = torch.from_numpy(self.colind)
        else:
            self.colind = None
            self.colind_t = _pin(torch.from_numpy(np.load(colind_path)))
        self._g2l = torch.full((self.num_nodes,), -1, dtype=torch.int64, device=device)

    def degree(self):
        return self.rowptr[1:] - self.rowptr[:-1]

    def _read_colind(self, eidx):
        """eidx: int64 CUDA tensor of edge offsets. Returns colind[eidx] on the GPU."""
        st = self.stats
        uniq, inv = torch.unique(eidx, sorted=True, return_inverse=True)
        uniq_cpu = uniq.cpu()
        if self.colind_on == "card" and self.prefetch:
            st.col_pages += im.prefetch_rows(self.colind, uniq_cpu.numpy())
        vals = _pin(self.colind_t.index_select(0, uniq_cpu)).to(self.device, non_blocking=True)
        st.col_entries += uniq_cpu.numel()
        return vals.to(torch.int64)[inv]

    def sample(self, seeds, fanouts, generator=None):
        """Layer-wise uniform neighbor sampling (with replacement when degree > fanout).

        Returns (n_id, edge_index): n_id are global ids with the seeds first; edge_index is
        (2, E) in local ids, row 0 = neighbor (source of the message), row 1 = target, which is
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
            start = self.rowptr[frontier]
            deg = self.rowptr[frontier + 1] - start
            j = torch.arange(fanout, device=dev).expand(frontier.numel(), fanout)
            take_all = (deg <= fanout).unsqueeze(1)
            rnd = torch.rand((frontier.numel(), fanout), device=dev, generator=generator)
            offs = torch.where(take_all, j, (rnd * deg.unsqueeze(1)).long())
            valid = j < torch.clamp(deg, max=fanout).unsqueeze(1)
            eidx = (start.unsqueeze(1) + offs)[valid]                       # edges to read
            tgt = frontier.unsqueeze(1).expand(-1, fanout)[valid]
            nbr = self._read_colind(eidx)                                   # col_idx from the card
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


def hot_by_degree(graph, hbm_bytes, row_bytes):
    """Pick the highest-degree nodes that fit in hbm_bytes (a common static cache policy)."""
    k = min(int(hbm_bytes // row_bytes), graph.num_nodes)
    if k <= 0:
        return torch.empty(0, dtype=torch.int64)
    return torch.topk(graph.degree(), k).indices.cpu()
