# -*- coding: utf-8 -*-
"""
bench_lane_b.py — Lane B of retrieval-arch-compare.pptx, WITHOUT GPUDirect Storage.

Design under test: K and V live on SSD permanently. Every retrieval re-fetches K,
ships it across PCIe, scans it in HBM for 3 attractor iterations, then gathers the
k winning V columns. Nothing stays resident between retrievals.

Without GPUDirect the key matrix takes four hops:

    SSD -> page cache -> pinned host buffer -> PCIe -> HBM
           (2) read                            (3) H2D

Both are timed separately, because the whole point of the GPUDirect variant is to
collapse them into one.

Buffers (pinned host, device) are allocated once outside the measurement loop: that
is allocator setup, not per-retrieval work, and pinning 2 GB takes long enough to
swamp the numbers if it is repeated.

`--direct` removes the page-cache hop alone, leaving the H2D in place. It therefore
separates the two savings GDS bundles together: the cache copy (measurable today,
no driver change) and the bounce through host memory (needs nvidia-fs).

Usage:
  python3 bench_lane_b.py --data-dir /mnt/data/bench_data                  # B1 buffered
  python3 bench_lane_b.py --data-dir /mnt/data/bench_data --direct         # B2 O_DIRECT
  python3 bench_lane_b.py --data-dir /mnt/data/bench_data --io-threads 1   # single-stream read

Requires the K.bin / V.bin written by `bench_retrieval_ab.py --setup`.
"""
import argparse, os, sys, time

import numpy as np
from concurrent.futures import ThreadPoolExecutor


def evict(paths):
    """Drop the files from the OS page cache so the read is genuinely cold."""
    if not hasattr(os, "posix_fadvise"):
        return False
    for p in paths:
        try:
            fd = os.open(p, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        except OSError:
            pass
    return True


ALIGN = 4096          # logical block size; O_DIRECT requires offset/length/address multiples
BLK = 16 << 20        # per-syscall request size


def read_into(path, view, pool, nio, direct):
    """Cold read of the whole file straight into `view`, striped across `nio` readers.

    Reading into the pinned buffer directly avoids a second copy before the H2D.

    With direct=True the file is opened O_DIRECT, so the bytes are DMA'd from the
    drive into the pinned buffer without passing through the kernel's page cache.
    That is the same I/O path `fio --direct=1` measures, and the same one GDS takes
    minus the final peer-to-peer hop into HBM — which makes it the honest baseline
    to compare GDS against. O_DIRECT demands that the file offset, the request
    length and the destination address all be multiples of ALIGN; the pinned
    allocation is page-aligned, the stripe boundaries are rounded up to ALIGN, and
    BLK is a multiple of it, so every request qualifies."""
    sz = os.path.getsize(path)
    chunk = ((sz + nio - 1) // nio + ALIGN - 1) // ALIGN * ALIGN
    flags = os.O_RDONLY
    if direct:
        flags |= getattr(os, "O_DIRECT", 0)

    def worker(i):
        lo, hi = i * chunk, min((i + 1) * chunk, sz)
        if lo >= hi:
            return
        fd = os.open(path, flags)
        try:
            off = lo
            while off < hi:
                got = os.preadv(fd, [view[off:min(off + BLK, hi)]], off)
                if not got:
                    break
                off += got
        finally:
            os.close(fd)

    t0 = time.perf_counter()
    if nio == 1:
        worker(0)
    else:
        list(pool.map(worker, range(nio)))
    return time.perf_counter() - t0


def gather_v(vp, idx, w, o, dst_view):
    """Read the k winning V columns from SSD into a pinned staging buffer.
    Columns are visited in ascending file offset; each keeps its own weight.

    Stays on the buffered path even when the K read is O_DIRECT: a V column is
    2*o = 6144 bytes at an offset of idx*6144, which is not block-aligned, and 64
    scattered small reads are exactly the case the page cache serves well."""
    colbytes = 2 * o
    order = np.argsort(idx)
    t0 = time.perf_counter()
    with open(vp, "rb", buffering=0) as f:
        for slot, j in enumerate(order):
            f.seek(int(idx[j]) * colbytes)
            dst_view[slot * colbytes:(slot + 1) * colbytes] = f.read(colbytes)
    return time.perf_counter() - t0, order


def report(label, secs, gbytes=None):
    m = sorted(secs)[len(secs) // 2]
    bw = f"   ({gbytes / m:6.2f} GB/s)" if gbytes else ""
    print(f"  {label:<42} median {m * 1e3:9.2f} ms{bw}")
    return m


def main(a):
    try:
        import torch
    except ImportError:
        print("torch not installed"); return 1
    if not torch.cuda.is_available():
        print("no CUDA device visible"); return 1

    kp, vp = os.path.join(a.data_dir, "K.bin"), os.path.join(a.data_dir, "V.bin")
    for p in (kp, vp):
        if not os.path.exists(p):
            print(f"missing {p} — run: python3 bench_retrieval_ab.py --setup"); return 1

    dev = torch.device("cuda")
    kgb = 2 * a.n * a.M / 1e9
    mode = "O_DIRECT" if a.direct else "buffered"
    hops = ("SSD -> pinned -> PCIe -> HBM" if a.direct
            else "SSD -> page cache -> pinned -> PCIe -> HBM")
    print(f"\n[LANE B - GPU, no GPUDirect]  {torch.cuda.get_device_name(0)}")
    print(f"  M={a.M} n={a.n} o={a.o} k={a.k} iters={a.iters} reps={a.reps} "
          f"io_threads={a.io_threads} read={mode}")
    print(f"  K = {kgb:.3f} GB (fp16)   path: {hops}")

    # ---- one-time buffers (not part of a retrieval) ----
    t0 = time.perf_counter()
    K_host = torch.empty(a.M * a.n, dtype=torch.float16, pin_memory=True)
    K_view = memoryview(K_host.numpy().view(np.uint8))
    K_dev = torch.empty((a.M, a.n), dtype=torch.float16, device=dev)   # row c == column c of K
    V_host = torch.empty(a.k * a.o, dtype=torch.float16, pin_memory=True)
    V_view = memoryview(V_host.numpy().view(np.uint8))
    xi_host = torch.empty(a.n, dtype=torch.float16, pin_memory=True)
    torch.cuda.synchronize()
    print(f"  one-time: pinned {kgb:.2f} GB + HBM {kgb:.2f} GB allocated in "
          f"{(time.perf_counter()-t0)*1e3:.0f} ms")

    pool = ThreadPoolExecutor(max(a.io_threads, 1))
    if not evict([kp, vp]):
        print("  (no fadvise on this OS - reads may be warm)")

    # ---- the probe: a noised copy of one stored key, so a run can be checked ----
    probe = a.M // 3
    colbytes = 2 * a.n
    with open(kp, "rb", buffering=0) as f:
        f.seek(probe * colbytes)
        probe_col = np.frombuffer(f.read(colbytes), dtype=np.float16).astype(np.float32)
    rng = np.random.default_rng(1)
    xi0 = probe_col + 0.05 * rng.uniform(-1, 1, a.n).astype(np.float32)
    xi0 /= np.linalg.norm(xi0)

    t_read, t_h2d, t_iter, t_gather = [], [], [], []
    winner = -1
    for rep in range(a.reps):
        evict([kp, vp])

        # (2) SSD -> pinned host
        t_read.append(read_into(kp, K_view, pool, a.io_threads, a.direct))

        torch.cuda.synchronize()                                       # (3) H2D over PCIe
        t0 = time.perf_counter()
        K_dev.view(-1).copy_(K_host, non_blocking=True)
        torch.cuda.synchronize()
        t_h2d.append(time.perf_counter() - t0)

        xi_host.copy_(torch.from_numpy(xi0))
        xi = xi_host.to(dev, non_blocking=True)
        torch.cuda.synchronize()

        t0 = time.perf_counter()                                       # (4) GEMV x3 in HBM
        for _ in range(a.iters):
            s = (K_dev @ xi).float()                                   # (M,n)@(n,) -> (M,)
            wv, idx = torch.topk(s, a.k)
            wv = torch.softmax(8.0 * wv, dim=0)
            xi = (K_dev[idx].T @ wv.half())
            xi = xi / (xi.norm() + 1e-6)
        torch.cuda.synchronize()
        t_iter.append(time.perf_counter() - t0)

        idx_np = idx.cpu().numpy()
        w_np = wv.float().cpu().numpy()
        winner = int(idx_np[int(np.argmax(w_np))])

        t0 = time.perf_counter()                                       # (6) V gather -> GPU
        _, order = gather_v(vp, idx_np, w_np, a.o, V_view)
        V_dev = V_host.to(dev, non_blocking=True).view(a.k, a.o)
        wsorted = torch.from_numpy(w_np[order].copy()).to(dev).half()
        out = wsorted @ V_dev
        torch.cuda.synchronize()
        t_gather.append(time.perf_counter() - t0)

        print(f"    rep {rep+1}/{a.reps}", file=sys.stderr)

    pool.shutdown()
    m1 = report(f"(2) SSD->pinned host, read K ({kgb:.2f} GB, {mode})", t_read, kgb)
    m2 = report(f"(3) H2D over PCIe ({kgb:.2f} GB)", t_h2d, kgb)
    m3 = report(f"(4) GEMV + top-k x{a.iters} in HBM", t_iter, kgb * a.iters)
    m4 = report(f"(6) gather {a.k} V cols SSD->host->GPU", t_gather)
    print(f"  ---- retrieval total: {(m1 + m2 + m3 + m4) * 1e3:.1f} ms")
    print(f"\n  probe column #{probe} -> winner #{winner}  "
          f"({'HIT' if winner == probe else 'MISS'})")
    print(f"  PCIe traffic per retrieval: {kgb:.3f} GB in "
          f"+ {a.k * a.o * 2 / 1e6:.3f} MB of V")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                       "bench_data"))
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--o", type=int, default=3072)
    ap.add_argument("--M", type=int, default=1_000_000)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--io-threads", type=int, default=16,
                    help="concurrent readers for stage (2); 1 = single sequential stream")
    ap.add_argument("--direct", action="store_true",
                    help="open K.bin with O_DIRECT for stage (2), bypassing the page cache")
    sys.exit(main(ap.parse_args()))
