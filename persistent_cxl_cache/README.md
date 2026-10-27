# persistent_cxl_cache: GNN graph and features on GPU HBM + a CXL DRAM cache + SSD

Example code for out-of-memory GNN training. Both the graph structure (CSC `indptr` and
`indices`) and the node feature table sit on an SSD-backed CXL memory device (XCENA MX1P
with InfiniteMemory). The device DRAM caches hot blocks, and its SSD holds the full arrays.
The method is meant to be compared with **BaM** and **GIDS**, which keep the graph in host
memory and read features from the SSD with GPU-initiated I/O.

**Status:** the code is checked on the CPU with a fake device (`tests/`). It has not yet been
run on a card, so it comes with no performance numbers.

## Placement

| Data | This method (default) | Variants | GIDS (BaM) baseline |
| --- | --- | --- | --- |
| `indptr` (CSC, N+1) | device | `--graph-on split`: GPU HBM | pinned host memory |
| `indices` (CSC, E) | device | `--graph-on host`: pinned host memory | pinned host memory |
| Feature table | device; hottest rows also in HBM (`--hbm-gb`) | — | NVMe SSD via BaM, GPU software cache, constant CPU buffer |
| Read granularity | 512 KB device pages (64 KB announced) | — | 4 KB cache lines |
| Who issues I/O | host CPU: one async prefetch per distinct page, then a gather | — | GPU threads |

`--graph-on host` reproduces the GIDS graph placement. With it, only the feature path differs
from GIDS.

**CSC:** the list of node `v`, `indices[indptr[v]:indptr[v+1]]`, holds its in-neighbors, the
nodes that send messages to `v`. This is the format DGL and GIDS sample from. cuGraph-PyG reaches the
same in-neighbor lists another way: its GraphStore reverses the edge index before building the
graph. For a symmetric graph, CSC and
CSR are the same arrays.

Every hop does the same steps on each device-resident array:

1. **Reduce requests on the GPU.** Deduplicate and sort by address. For features, drop HBM hits
   first.
2. **Prefetch.** Issue one asynchronous `prefetchMemory` per distinct device page, so the SSD
   reads overlap.
3. **Gather.** `index_select` from the device into pinned host memory.
4. **H2D.** Copy to the GPU and restore the request order.

The hop itself reads `indptr[v]` and `indptr[v+1]` for the frontier, picks k offsets per node
on the GPU, then reads only those `indices` entries.

## Page granularity is the main question

The device moves whole pages. A 512 B feature row read through a 512 KB page costs 1024× the
bytes it uses, and an 8 B `indptr` entry costs far more. BaM's 4 KB lines cost 8× for that
row. Unless node ordering and batch construction put what a batch needs on the same pages, the
device tier reads far more from the SSD than BaM does.

`page_sim.py` measures this offline, without a card or SSD. It replays real sampled batches
under each node ordering (`original`, `degree`, `random`, `rcm`, or your own permutation with
`--orders file --perm`), each placement (`--graph-on card|split|host`) and each page size (`4K`
= BaM, `64K`, `512K`). For each combination it reports:

- pages touched per batch (features, and graph pages for `indptr` + `indices`);
- the read amplification (bytes moved / bytes used);
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
| `hbm_cxl_ssd/tiered_store.py` | `TieredGraph` (CSC sampling; `graph_on` = card / split / host), `TieredFeatures` (HBM → device gather), `SsdCounter`, `graph_files` |
| `hbm_cxl_ssd/bench_tiered.py` | Mini-batch loop. Reports time per stage, HBM hit rate, pages touched and SSD bytes. Optional 3-layer GraphSAGE (`--train`) |
| `hbm_cxl_ssd/page_sim.py` | Offline page-granularity, ordering, placement and cache-size simulation |
| `hbm_cxl_ssd/prepare_ogbn.py` | OGB node dataset → `indptr.npy`, `indices.npy` (CSC), `feat.npy`, `labels.npy`, `train_idx.npy`. Graph: `--format sym` (default, add reverse edges) or `csc` (keep direction). Features: `--fp16` |
| `hbm_cxl_ssd/tests/` | CPU tests: fake device for all three placements, CSC conversion; stack distance vs direct LRU |

Older data directories with `rowptr.npy` / `colind.npy` are still accepted. See also
`../dran_mu/im_examples.cpp` for basic InfiniteMemory usage from C++.

## Quick start

```bash
pip install ogb numpy torch scipy                     # torch_geometric only for --train
cd persistent_cxl_cache/hbm_cxl_ssd
python prepare_ogbn.py --name ogbn-products --root ~/ogb --out ~/csc_products

# no card needed:
python tests/test_tiered.py
python tests/test_page_sim.py
python page_sim.py --data ~/csc_products --orders original,degree,rcm --graph-on card --cache-gb 0.5,1,2,4

# on the card host (GPU + XCENA SDK):
(cd ../im_loader && cmake -B build && cmake --build build)
python ../im_loader/im_store.py 0                     # 1 GB round trip: "read back correct: True"
python bench_tiered.py --data ~/csc_products --hbm-gb 0.2                    # graph + features on the device
python bench_tiered.py --data ~/csc_products --hbm-gb 0.2 --graph-on host    # GIDS graph placement
python bench_tiered.py --data ~/csc_products --hbm-gb 0 --no-prefetch        # ablation
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
- **Small datasets never reach the SSD.** ogbn-products (about 1 GB of features, 1 GB of
  `indices`) stays in the device DRAM even at the 4 GB minimum cache. Use ogbn-papers100M
  (`--fp16`, about 28 GB of features and 26 GB of `indices`) or larger to exercise the SSD tier.
- **This code reads the device from the host CPU, not from the GPU.** A GPU load that misses to
  the SSD would wait about 100 µs or more, and that is untested.

## Not done yet

- A look-ahead cache policy over a window of future batches (the counterpart of GIDS window
  buffering).
- Overlapping batch *i+1*'s sampling and gather with batch *i*'s compute.
- Sampling without replacement (the current sampler takes neighbors with replacement when
  degree > fanout; DGL samples without replacement by default).
- Multi-GPU.
