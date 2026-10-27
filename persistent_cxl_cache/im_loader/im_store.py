"""im_store.py - put numpy / torch arrays in XCENA InfiniteMemory from Python.

    import im_store as im
    im.init(0)
    feat = im.load_npy("/mnt/data/papers100M/node_feat.npy")   # numpy array living on the card
    feat_t = torch.from_numpy(feat)                              # zero-copy torch view
    rows = feat_t[idx]                                           # ordinary gather, served by the card
    im.free(feat)

The arrays are ordinary host-addressable memory on the card. The card caches 512 KB pages in
its DRAM (the Cache Region) and fetches misses from its SSD by itself.

Things to know:
  * The data lives only as long as this process (or until free()). Exiting releases the memory,
    and release DISCARDS the data. Load in the same process that trains.
  * Never call unmap to evict data: it zeroes it.
  * Writing to never-cached pages first reads them from the SSD (write-allocate), so loading
    runs at about 1.4-2.4 GB/s (measured). A dataset larger than the Cache Region also forces
    write-back to the SSD while loading.
  * One 512 B feature row read from the SSD costs a 512 KB page read. Random gathers are only
    fast while the table fits in the Cache Region (see `xcena_cli im get-smart`, dram_capacity).
"""
import ctypes
import os

import numpy as np

_GB = 1 << 30
_lib = None
_live = {}  # data pointer -> (allocated bytes, ctypes buffer)

_STATUS = {0: "Success", 1: "ResourceExhaust", 2: "InvalidRequest", 3: "NotSupported"}


def _load_lib():
    global _lib
    if _lib is not None:
        return _lib
    path = os.environ.get("IM_STORE_LIB",
                          os.path.join(os.path.dirname(os.path.abspath(__file__)), "build", "libim_store.so"))
    lib = ctypes.CDLL(path)
    lib.im_init.argtypes = [ctypes.c_int]
    lib.im_init.restype = ctypes.c_int
    lib.im_alloc.argtypes = [ctypes.c_size_t]
    lib.im_alloc.restype = ctypes.c_void_p
    lib.im_free.argtypes = [ctypes.c_void_p]
    lib.im_free.restype = None
    lib.im_flush.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.im_flush.restype = None
    lib.im_prefetch_offsets.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_uint64]
    lib.im_prefetch_offsets.restype = ctypes.c_int64
    for name in ("im_commit", "im_prefetch", "im_pin", "im_unpin", "im_preload", "im_unload"):
        fn = getattr(lib, name)
        fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        fn.restype = ctypes.c_int
    _lib = lib
    return lib


def init(device=0):
    if _load_lib().im_init(device) != 0:
        raise RuntimeError(f"pxl createContext({device}) failed")


def _ptr(arr):
    return arr.__array_interface__["data"][0]


def empty(shape, dtype, round_to_gb=True):
    """Allocate an uninitialised array on the card. Rounded up to whole GB so it can be preloaded."""
    dtype = np.dtype(dtype)
    nbytes = int(np.prod(shape)) * dtype.itemsize
    alloc = -(-nbytes // _GB) * _GB if round_to_gb else nbytes
    p = _load_lib().im_alloc(alloc)
    if not p:
        raise MemoryError(f"im_alloc({alloc / _GB:.1f} GB) failed")
    buf = (ctypes.c_char * alloc).from_address(p)
    arr = np.frombuffer(buf, dtype=dtype, count=int(np.prod(shape))).reshape(shape)
    _live[p] = (alloc, buf)
    return arr


def free(arr):
    p = _ptr(arr)
    if p in _live:
        _lib.im_free(ctypes.c_void_p(p))
        del _live[p]


def _span(arr):
    p = _ptr(arr)
    return ctypes.c_void_p(p), (_live[p][0] if p in _live else arr.nbytes)


def flush(arr):
    p, n = _span(arr)
    _lib.im_flush(p, n)


def _call(name, arr):
    p, n = _span(arr)
    s = getattr(_lib, name)(p, n)
    return _STATUS.get(s, str(s))


def commit(arr):   return _call("im_commit", arr)    # dirty pages -> SSD, wait
def pin(arr):      return _call("im_pin", arr)       # keep in DRAM cache
def unpin(arr):    return _call("im_unpin", arr)
def preload(arr):  return _call("im_preload", arr)   # into the Preload Region, for MU kernels
def unload(arr):   return _call("im_unload", arr)    # always writes back


PAGE = 512 * 1024  # InfiniteMemory page size on MX1P (64 KB announced)


def prefetch_rows(arr, rows):
    """Asynchronously fetch the pages holding the given rows (call before a scattered gather).

    rows: int64 numpy array, ideally sorted (unsorted works, but repeats pages).
    Returns the number of distinct pages issued.
    """
    offs = np.ascontiguousarray(np.asarray(rows, dtype=np.int64) * int(arr.strides[0]))
    return int(_lib.im_prefetch_offsets(ctypes.c_void_p(_ptr(arr)), offs.ctypes.data_as(ctypes.c_void_p),
                                        offs.size, PAGE))


def load_npy(path, chunk_bytes=_GB, verbose=True):
    """Copy a .npy file into InfiniteMemory chunk by chunk; returns the on-card numpy array."""
    import time
    src = np.load(path, mmap_mode="r")
    dst = empty(src.shape, src.dtype)
    flat_src, flat_dst = src.reshape(-1), dst.reshape(-1)
    step = max(1, chunk_bytes // src.dtype.itemsize)
    t0 = time.time()
    for i in range(0, flat_src.size, step):
        flat_dst[i:i + step] = flat_src[i:i + step]
        if verbose and (i // step) % 8 == 0:
            done = min(i + step, flat_src.size) * src.dtype.itemsize
            print(f"  {path}: {done / _GB:.1f} / {src.nbytes / _GB:.1f} GB, "
                  f"{done / _GB / max(time.time() - t0, 1e-9):.2f} GB/s", flush=True)
    flush(dst)
    if verbose:
        print(f"  loaded {src.nbytes / _GB:.1f} GB in {time.time() - t0:.0f} s", flush=True)
    return dst


if __name__ == "__main__":
    # Smoke test: 1 GB round trip, no dataset needed.
    import sys
    init(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    a = empty((_GB // 4,), np.float32)
    a[:] = np.arange(a.size, dtype=np.float32)
    flush(a)
    print("commit:", commit(a))
    flush(a)
    ok = np.array_equal(a[::4096], np.arange(0, a.size, 4096, dtype=np.float32))
    print("read back correct:", ok)
    free(a)
