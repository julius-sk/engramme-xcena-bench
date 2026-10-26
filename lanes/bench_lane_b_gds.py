# -*- coding: utf-8 -*-
"""
bench_lane_b_gds.py — Lane B of retrieval-arch-compare.pptx, WITH GPUDirect Storage.

Same workload, same files, same probe check and same stage boundaries as
`bench_lane_b.py`, so the two can be read side by side. The one difference is the
path the key matrix takes:

    bench_lane_b.py --direct :  SSD -> pinned host -> PCIe -> HBM     (2) then (3)
    this script              :  SSD -----------------------> HBM      (2) only

GDS has the drive DMA straight into GPU memory, so host memory is never a way
station and stage (3) disappears rather than shrinking. Stage (2) is bounded by the
same drive either way, so the honest expectation is that this run reproduces the
~420 ms read and simply drops the ~82 ms transfer — but P2P DMA does not always
match the host path, which is exactly why this is measured instead of subtracted.

The V gather deliberately stays on the buffered host path used by `bench_lane_b.py`.
A V column is 6144 bytes at an offset of idx*6144, which is not block-aligned, so
routing it through cuFile would land in per-I/O compat mode and measure nothing
useful. Keeping it identical isolates the K path, which is the only thing GDS
changes here.

Usage:
  python3 bench_lane_b_gds.py --data-dir /mnt/data/bench_data
  python3 bench_lane_b_gds.py --data-dir /mnt/data/bench_data --io-threads 32

Requires nvidia_fs loaded and kvikio-cu12 installed.
"""
import argparse, os, sys, time

import numpy as np


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


def gather_v(vp, idx, w, o, dst_view):
    """Read the k winning V columns from SSD into a pinned staging buffer.
    Identical to bench_lane_b.py: buffered, ascending file offset, weights carried
    by the same permutation as the indices."""
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


def describe_kvikio(kvikio):
    """Report the settings that decide whether this run means anything.

    kvikio 26.x configures through defaults.get(); older releases used named
    functions, so both shapes are tried. Note that `compat_mode` is a tri-state
    enum (OFF / ON / AUTO), not a boolean — AUTO means kvikio decides per
    operation, so the value that actually matters is is_compat_mode_preferred(),
    which reports the decision rather than the policy."""
    d = kvikio.defaults
    raw, nthreads, preferred = None, None, None
    for get in (lambda: d.get("compat_mode"), lambda: d.compat_mode()):
        try:
            raw = get()
            break
        except Exception:
            continue
    for get in (lambda: d.get("num_threads"), lambda: d.get_num_threads()):
        try:
            nthreads = get()
            break
        except Exception:
            continue
    try:
        preferred = bool(d.is_compat_mode_preferred())
    except Exception:
        pass
    try:
        mode = kvikio.CompatMode(raw).name
    except Exception:
        mode = {0: "OFF", 1: "ON", 2: "AUTO"}.get(raw, str(raw))
    return mode, nthreads, preferred


def main(a):
    # kvikio reads its configuration at import time, so the knobs go in first.
    os.environ.setdefault("KVIKIO_NTHREADS", str(a.io_threads))
    if not a.allow_compat:
        # Refuse to silently fall back to a POSIX read. A compat-mode run would
        # produce plausible numbers that are not GDS numbers.
        os.environ.setdefault("KVIKIO_COMPAT_MODE", "OFF")

    try:
        import torch
    except ImportError:
        print("torch not installed"); return 1
    if not torch.cuda.is_available():
        print("no CUDA device visible"); return 1
    try:
        import kvikio, kvikio.defaults
    except ImportError:
        print("kvikio not installed — pip install kvikio-cu12"); return 1

    kp, vp = os.path.join(a.data_dir, "K.bin"), os.path.join(a.data_dir, "V.bin")
    for p in (kp, vp):
        if not os.path.exists(p):
            print(f"missing {p} — run: python3 bench_retrieval_ab.py --setup"); return 1

    dev = torch.device("cuda")
    kbytes = 2 * a.n * a.M
    kgb = kbytes / 1e9
    mode, nthreads, preferred = describe_kvikio(kvikio)
    print(f"\n[LANE B - GPU, GPUDirect Storage]  {torch.cuda.get_device_name(0)}")
    print(f"  M={a.M} n={a.n} o={a.o} k={a.k} iters={a.iters} reps={a.reps}")
    print(f"  kvikio {kvikio.__version__}   compat_mode={mode}   "
          f"compat_preferred={preferred}   nthreads={nthreads}")
    print(f"  K = {kgb:.3f} GB (fp16)   path: SSD -> HBM (no host bounce)")
    if preferred and not a.allow_compat:
        print("  ABORT: kvikio would fall back to a POSIX read, so this would not be"
              " a GDS number.\n  Check `lsmod | grep nvidia_fs`, /proc/driver/nvidia-fs/stats"
              " and the filesystem type.")
        return 2

    # ---- one-time buffers (not part of a retrieval) ----
    t0 = time.perf_counter()
    K_dev = torch.empty((a.M, a.n), dtype=torch.float16, device=dev)  # row c == column c of K
    K_bytes = K_dev.view(-1).view(torch.uint8)          # cuFile wants a byte-addressed buffer
    V_host = torch.empty(a.k * a.o, dtype=torch.float16, pin_memory=True)
    V_view = memoryview(V_host.numpy().view(np.uint8))
    xi_host = torch.empty(a.n, dtype=torch.float16, pin_memory=True)
    torch.cuda.synchronize()
    print(f"  one-time: HBM {kgb:.2f} GB allocated in "
          f"{(time.perf_counter()-t0)*1e3:.0f} ms  (no pinned staging buffer needed)")

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

    t_read, t_iter, t_gather = [], [], []
    winner = -1
    for rep in range(a.reps):
        evict([kp, vp])

        torch.cuda.synchronize()                       # (2) SSD -> HBM, one hop
        t0 = time.perf_counter()
        with kvikio.CuFile(kp, "r") as f:
            got = f.read(K_bytes, kbytes, 0)
        torch.cuda.synchronize()
        t_read.append(time.perf_counter() - t0)
        if got != kbytes:
            print(f"  short read: {got} of {kbytes} bytes"); return 1

        xi_host.copy_(torch.from_numpy(xi0))
        xi = xi_host.to(dev, non_blocking=True)
        torch.cuda.synchronize()

        t0 = time.perf_counter()                       # (4) GEMV x3 in HBM
        for _ in range(a.iters):
            s = (K_dev @ xi).float()                   # (M,n)@(n,) -> (M,)
            wv, idx = torch.topk(s, a.k)
            wv = torch.softmax(8.0 * wv, dim=0)
            xi = (K_dev[idx].T @ wv.half())
            xi = xi / (xi.norm() + 1e-6)
        torch.cuda.synchronize()
        t_iter.append(time.perf_counter() - t0)

        idx_np = idx.cpu().numpy()
        w_np = wv.float().cpu().numpy()
        winner = int(idx_np[int(np.argmax(w_np))])

        t0 = time.perf_counter()                       # (6) V gather -> GPU
        _, order = gather_v(vp, idx_np, w_np, a.o, V_view)
        V_dev = V_host.to(dev, non_blocking=True).view(a.k, a.o)
        wsorted = torch.from_numpy(w_np[order].copy()).to(dev).half()
        out = wsorted @ V_dev
        torch.cuda.synchronize()
        t_gather.append(time.perf_counter() - t0)

        print(f"    rep {rep+1}/{a.reps}", file=sys.stderr)

    m1 = report(f"(2) SSD->HBM via GDS, read K ({kgb:.2f} GB)", t_read, kgb)
    print(f"  {'(3) H2D over PCIe':<42} {'eliminated':>16}")
    m3 = report(f"(4) GEMV + top-k x{a.iters} in HBM", t_iter, kgb * a.iters)
    m4 = report(f"(6) gather {a.k} V cols SSD->host->GPU", t_gather)
    print(f"  ---- retrieval total: {(m1 + m3 + m4) * 1e3:.1f} ms")
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
                    help="cuFile worker threads (KVIKIO_NTHREADS)")
    ap.add_argument("--allow-compat", action="store_true",
                    help="permit kvikio's POSIX fallback; the result is then not a GDS number")
    sys.exit(main(ap.parse_args()))
