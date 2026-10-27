# persistent_cxl_cache: GNN features on GPU HBM + a CXL DRAM cache + SSD

Example code for out-of-memory GNN training, where the node feature table sits on an
SSD-backed CXL memory device (XCENA MX1P with InfiniteMemory). The method is meant to be
compared with **BaM** and **GIDS**, which read the feature table straight from the SSD with
GPU-initiated I/O.

**Status:** the code is checked on the CPU with a fake device (`tests/`). It has not yet been
run on a card, so it comes with no performance numbers.

## Same setup as GIDS, different feature path

GIDS keeps the graph structure pinned in host memory and puts only the feature table on
storage. This code does the same by default, so the feature path is the only difference.

| | GIDS (BaM) | This method |
| --- | --- | --- |
| Graph structure | pinned host memory, GPU sampling | pinned host memory (`--colind-on host`, default); optional `--colind-on card` |
| Feature table | NVMe SSD, read by GPU threads through BaM | the CXL device: device DRAM caches its SSD in hardware |
| Read granularity | 4 KB cache lines | 512 KB device pages on MX1P (64 KB announced) |
| GPU-side cache | BaM software cache with window buffering (look-ahead eviction) | static HBM cache of the highest-degree rows |
| Hot-node buffer | constant CPU buffer (hot nodes pinned in host DRAM) | the device DRAM cache (hardware-managed) |
| Who issues I/O | GPU threads, many concurrent SSD requests | host CPU: one async prefetch per distinct page, then a gather |

The feature path for each batch:

1. **Reduce requests on the GPU.** Drop HBM hits, deduplicate, and sort by address.
2. **Prefetch.** Issue one asynchronous `prefetchMemory` per distinct device page, so the SSD
   reads overlap.
3. **Gather.** `index_select` from the device into pinned host memory.
4. **H2D and scatter.** Copy to the GPU and put the rows back in batch order.

The graph is stored as CSR (`rowptr`, `colind`). `prepare_ogbn.py` makes the graph symmetric,
so CSR equals CSC, the in-neighbor format that DGL and GIDS sample from.

## Page granularity is the main question

The device moves whole pages. A 512 B feature row (128 × fp32) read through a 512 KB page
costs 1024× the bytes it uses. BaM's 4 KB lines cost 8×. Unless node ordering and batch
construction put the rows a batch needs on the same pages, the device tier reads far more
from the SSD than BaM does.

`page_sim.py` measures this offline, without a card or SSD. It replays real sampled batches
under each node ordering (`original`, `degree`, `random`, `rcm`, or your own permutation with
`--orders file --perm`) and each page size (`4K` = BaM, `64K`, `512K`). For each combination
it reports:

- pages touched per batch, and the read amplification (bytes moved / bytes used);
- SSD MB per batch at each cache size, from exact LRU stack distances (one pass gives the
  whole miss-ratio curve);
- the smallest cache that keeps SSD reads under `--target-mb` per batch.

The device's replacement policy is not documented. The simulation assumes LRU, which is
pessimistic for scan-like access: when a batch touches more pages than the cache holds, every
access misses.

## Layout

| Path | Contents |
| --- | --- |
| `im_loader/im_store.cpp` | Small C API over libpxl: init, alloc/free, flush, commit, prefetch (batched by page), pin, preload/unload |
| `im_loader/im_store.py` | ctypes wrapper. `load_npy` copies a `.npy` file into device memory and returns a zero-copy numpy array |
| `hbm_cxl_ssd/tiered_store.py` | `TieredGraph` (sampling; col_idx on host or device), `TieredFeatures` (HBM → device gather), `SsdCounter` (device SSD byte counters) |
| `hbm_cxl_ssd/bench_tiered.py` | Mini-batch loop. Reports time per stage, HBM hit rate, pages touched and SSD bytes. Optional 3-layer GraphSAGE (`--train`) |
| `hbm_cxl_ssd/page_sim.py` | Offline page-granularity, ordering and cache-size simulation (4K / 64K / 512K) |
| `hbm_cxl_ssd/prepare_ogbn.py` | OGB node dataset → `rowptr.npy`, `colind.npy`, `feat.npy`, `labels.npy`, `train_idx.npy` (`--fp16` halves features) |
| `hbm_cxl_ssd/tests/` | CPU tests: fake device for `tiered_store`; stack distance vs direct LRU for `page_sim` |

See also `../dran_mu/im_examples.cpp` for basic InfiniteMemory usage from C++.

## Quick start

```bash
pip install ogb numpy torch scipy                     # torch_geometric only for --train
cd persistent_cxl_cache/hbm_cxl_ssd
python prepare_ogbn.py --name ogbn-products --root ~/ogb --out ~/csr_products

# no card needed:
python tests/test_tiered.py
python tests/test_page_sim.py
python page_sim.py --data ~/csr_products --orders original,degree,random,rcm --cache-gb 0.5,1,2,4

# on the card host (GPU + XCENA SDK):
(cd ../im_loader && cmake -B build && cmake --build build)
python ../im_loader/im_store.py 0                     # 1 GB round trip: "read back correct: True"
python bench_tiered.py --data ~/csr_products --hbm-gb 0.2
python bench_tiered.py --data ~/csr_products --hbm-gb 0 --no-prefetch   # ablation
```

For the BaM/GIDS side, use the same dataset files, batch size, fanouts and GPU. Run GIDS's own
dataloader with the feature table on a dedicated NVMe SSD: BaM takes the SSD over from the
kernel NVMe driver, so it cannot be a disk the system is using.

## Things to know about InfiniteMemory (measured on MX1P)

- **Data lives only as long as the process.** Releasing memory, including at process exit,
  unmaps it and the data is gone. Load the dataset in the process that trains.
- **Never use `unmapMemory` to evict data.** It zeroes the range. To make data cold, commit it
  and let other data push it out of the cache, or shrink the cache with
  `xcena_cli im set-dram-capacity` (test use, minimum 4 GB, `max` restores).
- **`set-dram-capacity` applies to the whole device.** On a shared card it changes the cache
  for everyone.
- **Writes to never-cached pages first read them from the SSD** (write-allocate). Loading
  measured 1.4–2.4 GB/s.
- **Small datasets never reach the SSD.** ogbn-products (about 1 GB of features) stays in the
  device DRAM even at the 4 GB minimum cache. Use ogbn-papers100M (`--fp16`, about 28 GB of
  features) or larger to exercise the SSD tier.
- **This code reads the device from the host CPU, not from the GPU.** A GPU load that misses to
  the SSD would wait about 100 µs or more, and that is untested.

## Not done yet

- A look-ahead cache policy for the HBM tier (the counterpart of GIDS window buffering).
- Overlapping batch *i+1*'s gather with batch *i*'s compute.
- Sampling without replacement (the current sampler takes neighbors with replacement when
  degree > fanout; DGL samples without replacement by default).
- Multi-GPU.
