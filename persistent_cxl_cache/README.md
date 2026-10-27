# persistent_cxl_cache: GNN data on GPU HBM + a CXL DRAM cache + SSD

Example code for GNN mini-batch training where the graph and feature table live in an
SSD-backed CXL memory device (XCENA MX1P with InfiniteMemory). The device's DRAM acts as a
hardware-managed cache in front of its SSD, and the GPU keeps the hottest data in HBM.

**Status:** the code is checked on the CPU with a fake device (`tests/`). It has not yet been
run on a card, so it comes with no performance numbers.

## The three tiers

| Tier | What it holds | Managed by |
| --- | --- | --- |
| GPU HBM | `row_ptr`; feature rows of the highest-degree nodes (static cache) | software |
| CXL cache | the device DRAM (InfiniteMemory "Cache Region") | device hardware, 512 KB pages |
| SSD | the full `col_idx` and feature table, behind the device DRAM | device hardware |

To the host, the CXL cache and the SSD form one flat address space. Software decides how
requests reach that space:

1. **Sample on the GPU.** Degrees and edge ranges come from `row_ptr` in HBM. Only the
   selected `col_idx` entries are read from the device.
2. **Reduce requests.** Drop HBM hits, deduplicate, and sort by address.
3. **Prefetch.** Issue one asynchronous `prefetchMemory` per distinct page, so the SSD reads
   overlap.
4. **Gather.** `index_select` from the device into pinned host memory, then copy to the GPU
   and scatter back into batch order.

## How big must the CXL cache be?

That depends on how many distinct pages a batch touches, and that depends on the node
ordering and on how batches are formed. For example, with 512 B feature rows and a 512 KB
page, one page holds 1024 rows. A random-order batch that samples tens of thousands of nodes
touches most pages of the table. The cache then has to hold nearly the whole table, or every
batch reads gigabytes from the SSD.

`page_sim.py` answers the question offline, without a card. It replays real sampled batches
under each ordering (`original`, `degree`, `random`, `rcm`, or your own permutation with
`--orders file --perm`) and each page size (512 KB, and 64 KB announced). For each
combination it reports:

- feature and `col_idx` pages touched per batch, and the read amplification (bytes moved /
  bytes used);
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
| `hbm_cxl_ssd/tiered_store.py` | `TieredGraph` (sampling), `TieredFeatures` (gather), `SsdCounter` (device SSD byte counters) |
| `hbm_cxl_ssd/bench_tiered.py` | Mini-batch loop. Reports time per stage, HBM hit rate, pages touched and SSD bytes. Optional 3-layer GraphSAGE (`--train`) |
| `hbm_cxl_ssd/page_sim.py` | Offline cache-size and ordering simulation |
| `hbm_cxl_ssd/prepare_ogbn.py` | OGB node dataset → `rowptr.npy`, `colind.npy`, `feat.npy`, `labels.npy`, `train_idx.npy` |
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
- **Small datasets never reach the SSD.** ogbn-products (about 2 GB) stays in the device DRAM
  even at the 4 GB minimum cache. Use a larger dataset (for example ogbn-papers100M with fp16
  features) to exercise the SSD tier.
- **This code reads the device from the host CPU, not from the GPU.** A GPU load that misses to
  the SSD would wait about 100 µs or more, and that is untested.

## Not done yet

- A look-ahead cache policy over a window of future batches.
- Overlapping batch *i+1*'s gather with batch *i*'s compute.
- Sampling without replacement (the current sampler takes neighbors with replacement when
  degree > fanout).
- Multi-GPU.
