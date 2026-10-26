# -*- coding: utf-8 -*-
"""
bench_retrieval_ab.py — measure the two lanes of retrieval-arch-compare.pptx.

Design under test (both lanes): K and V live on SSD permanently. Every retrieval
re-fetches K; the compute tier holds K only during the 3 iterations.

  Lane A (CPU):  (2) SSD->DRAM read K  (3)(4) CPU GEMV x3 + top-k  (6) gather 64 V cols from SSD
  Lane B (GPU):  (2) SSD->DRAM read K  (3) H2D over PCIe  (4) HBM GEMV x3 + top-k
                 (6) gather 64 V cols from SSD -> GPU

One knob per dimension, so a run differs from another in exactly one respect:
  -j / --threads    threads for the GEMV, applied to BLAS (-j 1 = single-threaded scan)
  --io-threads      concurrent readers for stage (2); 1 = single sequential stream
  --direct          O_DIRECT for stage (2), so the read skips the page-cache copy

Usage:
  python bench_retrieval_ab.py --setup      # once: writes K.bin (2.05 GB) + V.bin (6.1 GB)
  python bench_retrieval_ab.py --lane A -j 96 --io-threads 16
  python bench_retrieval_ab.py --lane A -j 96 --io-threads 16 --direct
  python bench_retrieval_ab.py --lane B     # requires torch + CUDA

K/V are evicted from the OS page cache before every repetition (Linux fadvise), so
the read is genuinely cold. Numbers print per diagram step, median of --reps.
"""
import argparse, os, sys, time

# BLAS fixes its thread count when the library loads, so -j must be applied
# before numpy is imported.
_j = "96"
for _i, _a in enumerate(sys.argv):
    if _a in ("-j", "--threads") and _i + 1 < len(sys.argv):
        _j = sys.argv[_i + 1]
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = _j

import numpy as np
from concurrent.futures import ThreadPoolExecutor


def evict(paths):
    if not hasattr(os, "posix_fadvise"):
        return False  # non-Linux: only the first run after boot is cold
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


def read_file(path, pool, nio, direct=False):
    """Cold read of the whole file, striped across `nio` concurrent readers.
    nio = 1 is a single sequential stream. Returns (seconds, buffer view).

    With direct=True the file is opened O_DIRECT, so the drive DMAs straight into
    the destination buffer instead of being copied through the kernel's page cache.
    O_DIRECT requires the file offset, the request length and the destination
    address to all be multiples of ALIGN. A bytearray gives no alignment guarantee,
    so the buffer is over-allocated by one block and sliced forward to the next
    aligned address; stripe boundaries are rounded up to ALIGN and BLK is a multiple
    of it, which makes every request qualify.

    The zero-fill is not waste: it faults the pages in before the clock starts, the
    same way the previous `bytearray(sz)` did, so the timed region measures I/O and
    not first-touch page faults."""
    sz = os.path.getsize(path)
    backing = np.empty(sz + ALIGN, dtype=np.uint8)
    backing.fill(0)
    pad = (-backing.ctypes.data) % ALIGN
    view = memoryview(backing[pad:pad + sz])
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
    return time.perf_counter() - t0, view


def gather_v(vp, idx, w, o):
    """Read the k winning V columns from SSD and mix them.

    idx and w are parallel arrays. Columns are visited in ascending file offset,
    and each column keeps the weight belonging to it — indexing both arrays with
    the same permutation, not with the loop counter."""
    colbytes = 2 * o
    order = np.argsort(idx)
    out = np.zeros(o, dtype=np.float32)
    t0 = time.perf_counter()
    with open(vp, "rb", buffering=0) as f:
        for j in order:
            f.seek(int(idx[j]) * colbytes)
            col = np.frombuffer(f.read(colbytes), dtype=np.float16)
            out += float(w[j]) * col.astype(np.float32)
    return time.perf_counter() - t0, out


def report(label, secs, gbytes=None):
    m = sorted(secs)[len(secs) // 2]
    bw = f"   ({gbytes / m:6.2f} GB/s)" if gbytes else ""
    print(f"  {label:<40} median {m * 1e3:9.2f} ms{bw}")
    return m


def setup(a):
    os.makedirs(a.data_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    with open(os.path.join(a.data_dir, "K.bin"), "wb") as f:
        for i0 in range(0, a.M, 1 << 16):
            cols = min(1 << 16, a.M - i0)
            blk = rng.standard_normal((a.n, cols), dtype=np.float32)
            blk /= np.linalg.norm(blk, axis=0, keepdims=True) + 1e-9
            f.write(blk.astype(np.float16).tobytes(order="F"))
    with open(os.path.join(a.data_dir, "V.bin"), "wb") as f:
        for i0 in range(0, a.M, 1 << 15):
            cols = min(1 << 15, a.M - i0)
            f.write(rng.standard_normal((a.o, cols), dtype=np.float32)
                    .astype(np.float16).tobytes(order="F"))
    print(f"setup done: K {2*a.n*a.M/1e9:.2f} GB + V {2*a.o*a.M/1e9:.2f} GB in {a.data_dir}")


def lane_a(a):
    kp, vp = os.path.join(a.data_dir, "K.bin"), os.path.join(a.data_dir, "V.bin")
    kgb = 2 * a.n * a.M / 1e9
    mode = "O_DIRECT" if a.direct else "buffered"
    print(f"\n[LANE A - CPU]  n={a.n} M={a.M} k={a.k} iters={a.iters} reps={a.reps} "
          f"scan_threads={a.threads} io_threads={a.io_threads} read={mode}")
    if not evict([kp, vp]):
        print("  (no fadvise on this OS - reads may be warm)")
    t_read, t_iter, t_gather = [], [], []
    rng = np.random.default_rng(1)
    pool = ThreadPoolExecutor(a.io_threads)
    for r in range(a.reps):
        evict([kp, vp])
        sec, raw = read_file(kp, pool, a.io_threads, a.direct)      # (2)
        t_read.append(sec)
        K = np.frombuffer(raw, dtype=np.float16).reshape(a.n, a.M, order="F").astype(np.float32)
        del raw
        xi = rng.standard_normal(a.n).astype(np.float32)
        xi /= np.linalg.norm(xi)
        t0 = time.perf_counter()                                    # (3)(4) x iters
        for _ in range(a.iters):
            s = K.T @ xi
            idx = np.argpartition(s, -a.k)[-a.k:]
            w = np.exp(8.0 * (s[idx] - s[idx].max())); w /= w.sum()
            xi = K[:, idx] @ w
            xi /= np.linalg.norm(xi) + 1e-9
        t_iter.append(time.perf_counter() - t0)
        sec, _ = gather_v(vp, idx, w, a.o)                          # (6)
        t_gather.append(sec)
        del K
        print(f"    rep {r+1}/{a.reps}", file=sys.stderr)
    pool.shutdown()
    m1 = report(f"(2) SSD->DRAM read K ({kgb:.2f} GB, {mode})", t_read, kgb)
    m2 = report(f"(3)(4) CPU GEMV + top-k x{a.iters}", t_iter, 2 * kgb * a.iters)
    m3 = report(f"(6) gather {a.k} V cols from SSD", t_gather)
    print(f"  ---- retrieval total: {(m1 + m2 + m3) * 1e3:.1f} ms")


def lane_b(a):
    try:
        import torch
        assert torch.cuda.is_available()
    except Exception:
        print("\n[LANE B] torch + CUDA not available on this machine.")
        return
    import torch
    kp, vp = os.path.join(a.data_dir, "K.bin"), os.path.join(a.data_dir, "V.bin")
    kgb = 2 * a.n * a.M / 1e9
    dev = torch.device("cuda")
    print(f"\n[LANE B - GPU]  {torch.cuda.get_device_name(0)}  "
          f"n={a.n} M={a.M} k={a.k} iters={a.iters} reps={a.reps} io_threads={a.io_threads}")
    t_read, t_h2d, t_iter, t_gather = [], [], [], []
    pool = ThreadPoolExecutor(a.io_threads)
    for r in range(a.reps):
        evict([kp, vp])
        sec, raw = read_file(kp, pool, a.io_threads, a.direct)      # (2)
        t_read.append(sec)
        K_host = torch.from_numpy(
            np.frombuffer(raw, dtype=np.float16).reshape(a.n, a.M, order="F").copy()
        ).pin_memory()
        del raw
        torch.cuda.synchronize()
        t0 = time.perf_counter()                                    # (3)
        K = K_host.to(dev, non_blocking=True)
        torch.cuda.synchronize()
        t_h2d.append(time.perf_counter() - t0)
        xi = torch.randn(a.n, device=dev, dtype=torch.float16)
        xi /= xi.norm()
        torch.cuda.synchronize()
        t0 = time.perf_counter()                                    # (4) x iters
        for _ in range(a.iters):
            s = (K.T @ xi).float()
            w, idx = torch.topk(s, a.k)
            w = torch.softmax(8.0 * w, dim=0)
            xi = K[:, idx] @ w.half()
            xi /= xi.norm() + 1e-6
        torch.cuda.synchronize()
        t_iter.append(time.perf_counter() - t0)
        t0 = time.perf_counter()                                    # (6)
        _, out = gather_v(vp, idx.cpu().numpy(), w.cpu().numpy(), a.o)
        torch.from_numpy(out).to(dev)
        torch.cuda.synchronize()
        t_gather.append(time.perf_counter() - t0)
        del K, K_host
        torch.cuda.empty_cache()
        print(f"    rep {r+1}/{a.reps}", file=sys.stderr)
    pool.shutdown()
    m1 = report(f"(2) SSD->DRAM read K ({kgb:.2f} GB)", t_read, kgb)
    m2 = report(f"(3) H2D over PCIe ({kgb:.2f} GB)", t_h2d, kgb)
    m3 = report(f"(4) HBM GEMV + top-k x{a.iters}", t_iter, kgb * a.iters)
    m4 = report(f"(6) gather {a.k} V cols SSD->GPU", t_gather)
    print(f"  ---- retrieval total: {(m1 + m2 + m3 + m4) * 1e3:.1f} ms")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", action="store_true")
    ap.add_argument("--lane", choices=["A", "B"])
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_data"))
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--o", type=int, default=3072)
    ap.add_argument("--M", type=int, default=1_000_000)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("-j", "--threads", type=int, default=96, help="GEMV threads (applied to BLAS)")
    ap.add_argument("--io-threads", type=int, default=16, help="concurrent readers for stage (2)")
    ap.add_argument("--direct", action="store_true",
                    help="open K.bin with O_DIRECT for stage (2), bypassing the page cache")
    a = ap.parse_args()
    if a.setup:
        setup(a)
    elif a.lane == "A":
        lane_a(a)
    elif a.lane == "B":
        lane_b(a)
    else:
        ap.print_help()
