# persistent_cxl_cache: Usage Guide

Last updated 2026-10-27

## Overview

`persistent_cxl_cache` trains GNNs with both the graph structure (CSC `indptr` and `indices`) and the node feature table stored on an SSD-backed CXL memory device (XCENA MX1P, InfiniteMemory mode). It is built to be compared head to head with BaM and GIDS. The code lives in the `persistent_cxl_cache/` folder of [github.com/julius-sk/engramme-xcena-bench](https://github.com/julius-sk/engramme-xcena-bench).

**Status:** all code is tested on the CPU against a fake device. It has not run on a card yet, so there are no performance numbers.

| Tier | Holds (default `--graph-on card`) | Managed by |
| --- | --- | --- |
| GPU HBM | feature rows of the highest-degree nodes (static cache) | our code |
| CXL device DRAM | cached pages of `indptr`, `indices` and the feature table (InfiniteMemory "Cache Region") | device hardware, 512 KB pages |
| Device SSD | the full `indptr`, `indices` and feature table | device hardware |

Two other graph placements exist for comparison: `--graph-on split` keeps `indptr` in HBM, and `--graph-on host` pins both graph arrays in host DRAM, as GIDS does.

The device DRAM and its SSD look like one flat memory to the host. Our code controls only how requests reach it. For `indptr`, `indices` and features alike it deduplicates, sorts by address, prefetches each distinct page once and then gathers; for features it drops HBM hits first.

## Setup

You need Python 3.10+, numpy, torch and scipy for everything. A GPU and the XCENA SDK (libpxl, `xcena_cli`) are needed only for the on-device steps. Use `ogb` to prepare data, and `torch_geometric` only for `--train`.

1. Get the code:

   ```bash
   git clone https://github.com/julius-sk/engramme-xcena-bench.git
   cd engramme-xcena-bench/persistent_cxl_cache
   ```

   Already cloned: `git pull`.
2. Install Python packages: `pip install numpy torch scipy ogb` (add `torch_geometric` for training).
3. Run the CPU tests. No device is needed, and each should print `OK`:

   ```bash
   python hbm_cxl_ssd/tests/test_tiered.py
   python hbm_cxl_ssd/tests/test_page_sim.py
   ```
4. On the device host only, build the C library:

   ```bash
   cd im_loader && cmake -B build && cmake --build build && cd ..
   ```

   If CMake cannot find `pxl`, add `-DCMAKE_PREFIX_PATH=/usr/local`. The library lands in `im_loader/build/libim_store.so`. To load it from elsewhere, set `IM_STORE_LIB=/path/to/libim_store.so`.
5. On the device host, run the smoke test. It writes 1 GB to the device, commits it and reads it back:

   ```bash
   python im_loader/im_store.py 0
   ```

   Expected output: `commit: Success` and `read back correct: True`. The argument is the device id.

## Preparing data

All tools read one directory of `.npy` files. `prepare_ogbn.py` writes it from any OGB node dataset:

```bash
python hbm_cxl_ssd/prepare_ogbn.py --name ogbn-products --root ~/ogb --out ~/csc_products
python hbm_cxl_ssd/prepare_ogbn.py --name ogbn-papers100M --root ~/ogb --out ~/csc_papers --fp16
python hbm_cxl_ssd/prepare_ogbn.py --name ogbn-papers100M --root ~/ogb --out ~/csc_papers_dir --format csc
```

| File | dtype, shape | Contents |
| --- | --- | --- |
| `indptr.npy` | int64, N+1 | CSC pointers: the list of node v is `indices[indptr[v]:indptr[v+1]]` |
| `indices.npy` | int64, E | in-neighbor ids (nodes that send messages to v), grouped by v |
| `feat.npy` | float32 or float16 (`--fp16`), N × D | node features |
| `labels.npy` | int64, N | labels (NaN → 0) |
| `train_idx.npy` | int64 | training node ids |

The graph is stored as CSC, the in-neighbor format DGL and GIDS sample from. `--format sym` (default) adds the reverse of every edge first. That is the usual graph for an undirected dataset such as ogbn-products (123.7M edges), and CSC and CSR are then the same arrays. `--format csc` keeps OGB's edge direction. Your own data works too: write the same files. Only `indptr`, `indices` and `feat` are required, and older directories with `rowptr.npy` / `colind.npy` are still accepted.

| Dataset | Features | `indices` | Reaches the device SSD? |
| --- | --- | --- | --- |
| ogbn-products | about 1 GB (fp32) | about 1 GB | no: fits in device DRAM even at the 4 GB minimum cache |
| ogbn-papers100M | about 28 GB (fp16) / 57 GB (fp32) | about 26 GB (`sym`) | yes, if the cache is set below the data size |

## Python API: im_store (device memory)

`im_loader/im_store.py` puts numpy arrays in device memory. Each array is an ordinary host-addressable numpy array, so `torch.from_numpy` gives a zero-copy CPU tensor over the device.

```python
import sys; sys.path.insert(0, "persistent_cxl_cache/im_loader")
import im_store as im
import numpy as np, torch

im.init(0)                                   # device id, once per process
feat = im.load_npy("/data/csc_papers/feat.npy")   # copy file -> device, prints progress
feat_t = torch.from_numpy(feat)              # zero-copy view, shape (N, D)

rows = np.sort(np.array([5, 9000, 123456]))
im.prefetch_rows(feat, rows)                 # async: one request per distinct 512 KB page
x = feat_t[torch.from_numpy(rows)]           # gather; misses are served from the device SSD

im.free(feat)                                # data is discarded here (or at process exit)
```

| Call | What it does | Returns |
| --- | --- | --- |
| `init(device=0)` | create the device context | raises on failure |
| `load_npy(path, chunk_bytes=1 GiB)` | allocate on the device, copy the file in 1 GiB chunks, flush | numpy array on the device |
| `empty(shape, dtype)` | allocate uninitialized, rounded up to whole GiB | numpy array on the device |
| `free(arr)` | release; **the data is gone** | — |
| `flush(arr)` | push the CPU's cached copy out to the device (after host writes) | — |
| `commit(arr)` | write dirty pages to the SSD and wait; data stays readable | status string |
| `prefetch_rows(arr, rows)` | async fetch of the pages holding `rows` (pass them sorted) | distinct pages issued |
| `pin(arr)` / `unpin(arr)` | keep a range in device DRAM | status string |
| `preload(arr)` / `unload(arr)` | move into or out of the Preload Region, only for MU kernels (not needed for GNN training) | status string |

Status strings are `Success`, `ResourceExhaust`, `InvalidRequest` or `NotSupported`. On a device without an SSD, `preload` returns `NotSupported`, which is harmless.

## Python API: tiered_store (sampling and feature gather)

`hbm_cxl_ssd/tiered_store.py` gives you a sampler and a feature store that plug into any PyTorch training loop. This is the whole loop with a PyG model:

```python
import sys; sys.path.insert(0, "persistent_cxl_cache/hbm_cxl_ssd")
import torch
from tiered_store import TieredGraph, TieredFeatures, graph_files, hot_by_degree, Stats, SsdCounter, im

D = "/data/csc_papers"; dev = torch.device("cuda")
im.init(0)
stats, ssd = Stats(), SsdCounter(0)

indptr_path, indices_path = graph_files(D)                                     # indptr.npy, indices.npy
graph = TieredGraph(indptr_path, indices_path, dev, stats=stats, graph_on="card")   # both on the device
hot = hot_by_degree(graph, hbm_bytes=4 << 30, row_bytes=128 * 2)               # 4 GiB HBM cache, fp16 rows
feats = TieredFeatures(f"{D}/feat.npy", dev, hot_ids=hot, stats=stats)       # loads features onto the device

gen = torch.Generator(device=dev); gen.manual_seed(0)
rd0, wr0 = ssd.read()
for seeds in train_batches:                          # int64 CUDA tensors of seed node ids
    n_id, edge_index = graph.sample(seeds, [10, 10, 10], generator=gen)
    x = feats.gather(n_id)                           # (len(n_id), D) on the GPU
    out = model(x.float(), edge_index)[: seeds.numel()]   # seeds come first in n_id
    loss = loss_fn(out, labels[seeds]); opt.zero_grad(); loss.backward(); opt.step()
rd1, wr1 = ssd.read()
print(stats.summary(), f"SSD read {(rd1 - rd0) / 1e6:.0f} MB")
```

| Object | Call | Notes |
| --- | --- | --- |
| `TieredGraph(indptr, indices, device, prefetch=True, stats=None, graph_on="card")` | `sample(seeds, fanouts, generator=None)` → `(n_id, edge_index)` | samples in-neighbors (CSC). `n_id`: global ids, seeds first, unique. `edge_index`: (2, E) local ids, row 0 = in-neighbor (message source), row 1 = target (PyG convention) |
| | `graph_on` | `"card"`: `indptr` + `indices` on the device. `"split"`: `indptr` in HBM, `indices` on the device. `"host"`: both pinned in host DRAM (GIDS) |
| | `degree()` | in-degree of every node, on the CPU |
| `graph_files(data_dir)` | → `(indptr path, indices path)` | finds `indptr.npy`/`indices.npy`, or the older `rowptr.npy`/`colind.npy` |
| `TieredFeatures(feat, device, hot_ids=None, prefetch=True, stats=None)` | `gather(ids)` → tensor on the GPU | HBM hits first; misses are deduplicated, sorted, prefetched by page, gathered into pinned memory, then copied |
| `hot_by_degree(graph, hbm_bytes, row_bytes)` | → CPU int64 ids | the highest in-degree nodes that fit in `hbm_bytes` |
| `Stats()` | `reset()`, `summary()` | HBM hit rate; entries and device pages read for features, `indptr` and `indices`; time split (sample / device gather / H2D) |
| `SsdCounter(device)` | `read()` → `(nvme_read_bytes, nvme_write_bytes)` | device-wide counters from `xcena_cli im get-smart`. Take a delta around a run; `(None, None)` if `xcena_cli` cannot run |

The sampler is uniform. It samples **with replacement** when degree > fanout, which differs from DGL's default; match the two before any official comparison.

## Command-line tools

Two tools take the same data directory: `page_sim.py` runs anywhere with no device; `bench_tiered.py` needs the GPU and the device.

### page_sim.py: how big must the cache be? (no device needed)

```bash
cd persistent_cxl_cache/hbm_cxl_ssd
python page_sim.py --data ~/csc_products --orders original,degree,random,rcm
python page_sim.py --data ~/csc_papers --orders original,rcm --batching contiguous --cache-gb 4,8,16,32 --csv out.csv
python page_sim.py --data ~/csc_papers --orders original --graph-on host       # GIDS placement: features only
python page_sim.py --data ~/csc_papers --orders file --perm metis_order.npy    # your own ordering
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--orders` | `original,degree,random` | node layouts to compare: `original`, `degree`, `random`, `rcm`, `file` |
| `--perm` | — | `.npy` for `--orders file`: `order[i]` = node stored at position i |
| `--batching` | `random` | `random` seeds, or `contiguous` (seeds that sit next to each other in the layout, partition-like) |
| `--pages` | `4K,64K,512K` | page sizes; 4K = BaM/GIDS cache line, 512K = MX1P today, 64K announced |
| `--graph-on` | `card` | `card` = `indptr` + `indices` pages hit the device; `split` = `indices` only; `host` = features only (GIDS) |
| `--fanouts`, `--batch` | `10,10,10`, `1024` | sampling, same as training |
| `--batches`, `--warmup` | `200`, `50` | measured batches after warm-up batches |
| `--hbm-gb` | `0` | static HBM cache (highest-degree rows), excluded from device traffic |
| `--feat-bytes` | from `feat.npy` | override bytes per row, e.g. `256` to model fp16 from an fp32 file |
| `--cache-gb` | `1,2,4,8,16,32,64` | cache sizes to report SSD traffic at |
| `--target-mb` | `50` | acceptable SSD MB per batch; the tool prints the smallest cache that meets it |
| `--seed`, `--csv` | `0`, — | random seed; write the table to a CSV |

The `rcm` ordering needs scipy and can take a long time on 100M nodes.

### bench_tiered.py: the real loop on the device

```bash
python bench_tiered.py --data ~/csc_papers --hbm-gb 4                      # graph + features on the device
python bench_tiered.py --data ~/csc_papers --hbm-gb 4 --graph-on split     # indptr in HBM
python bench_tiered.py --data ~/csc_papers --hbm-gb 4 --graph-on host      # GIDS graph placement
python bench_tiered.py --data ~/csc_papers --hbm-gb 0 --no-prefetch        # ablation
python bench_tiered.py --data ~/csc_papers --hbm-gb 4 --train              # plus a 3-layer GraphSAGE step
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--device-id` | `0` | XCENA device id |
| `--hbm-gb` | `0.5` | HBM feature cache size; `0` turns it off |
| `--graph-on` | `card` | where the CSC graph lives: `card`, `split` (indptr in HBM) or `host` (GIDS) |
| `--fanouts`, `--batch` | `10,10,10`, `1024` | sampling |
| `--batches`, `--warmup` | `50`, `3` | measured batches after warm-up |
| `--no-prefetch` | off | skip the per-page async prefetch |
| `--train` | off | run forward, backward and step of a 3-layer GraphSAGE (needs `torch_geometric`) |

## Reading the output

The number that decides everything is **read amplification**, the bytes the device moves per byte the batch uses. It depends on page size and node ordering.

### page_sim.py

In this example run (a synthetic 2M-node graph, not real data, `--graph-on host` so only feature pages count), 4K pages amplify 6–7× and 512K pages 29–53×. With `--graph-on card` the graph pages are added on top: on the same graph, 512K pages amplify 59×, and no cache size meets a 20 MB-per-batch target because first-touch misses alone exceed it. With `split` the cache needed grows from 0.95 to 1.09 GiB.

```
order    page    feat pg  col pg  touch MB  use MB   ampl       0.1    0.25     0.5       1   cache for target
original 4K        26468       0       108    19.4      6       108      89      56      11   0.82 GiB
original 512K       1953       0      1024    19.4     53      1024    1024    1024       0   0.95 GiB
random   4K        60432       0       248    35.0      7       248     240     152       2   0.90 GiB
```

| Column | Meaning |
| --- | --- |
| `feat pg` | distinct device pages of the feature table touched per batch |
| `col pg` | distinct device pages of the graph touched per batch: `indptr` + `indices` with `card`, `indices` with `split`, 0 with `host` |
| `touch MB` | pages × page size: what the device must have in DRAM for this batch |
| `use MB` | bytes the batch actually uses (feature rows not in HBM, plus graph entries read) |
| `ampl` | `touch MB / use MB` |
| numbers under the cache sizes | SSD MB read per batch with that much device cache, from exact LRU |
| `cache for target` | smallest cache that keeps SSD reads ≤ `--target-mb` per batch; `never` if first-touch misses alone exceed it |

Good: `ampl` in single digits and a `cache for target` well below the table size. Bad: the cache needed is close to the whole table, so the SSD tier adds nothing. That is the default outcome with 512 KB pages and random node ids. Orderings (`rcm`, METIS via `--perm`) and `--batching contiguous` exist to move this number.

The simulator assumes LRU. The device's real replacement policy is not documented, so treat the curve as a planning estimate and confirm it with `bench_tiered.py`.

### bench_tiered.py

```
[result] 50 batches, batch 1024, fanouts [10, 10, 10], prefetch True
  per batch: median … ms, mean … ms
  HBM hit …% | feature pages … | indptr entries … pages … | indices entries … pages … | sample … ms, card gather … ms, H2D … ms
  card SSD: read … MB, write … MB over the run
```

- `card SSD: read` is the proof of where the data came from. Near 0 means everything was served from device DRAM, and you measured a DRAM tier, not the SSD.
- `card gather` is the device-side time; compare it with `H2D` and with compute (`--train`) to see what bounds a batch.
- `write` should stay near 0. Training only reads features.

## Rules for using the device

The device is shared and some calls destroy data silently, so follow these rules on every run.

1. **Check that the card is idle before a run.** Only `pxl_resourced` may hold the device, and the counters should not move:

   ```bash
   sudo fuser -v /dev/mx_dma/* /dev/dax*
   xcena_cli im get-smart 0 | grep -E "used_dram|dirty_dram"
   nvidia-smi
   ```

   Tell the team before any run that writes tens of GB.
2. **Run long jobs in `tmux`**, so a dropped SSH session does not kill them.
3. **Data lives only as long as your process.** `free()` or process exit releases the memory, and release discards the data. Load the dataset in the same process that trains.
4. **Never call `unmapMemory` to evict data.** It zeroes the range; nothing in this code calls it.
5. **Making data cold:** commit it, then shrink the cache with `xcena_cli im set-dram-capacity` (minimum 4 GB, `max` restores). This setting applies to the whole card and to everyone on it. Use it only when you have the card to yourself, and restore it when done.
6. **Loading is slow by design.** A write to a never-cached page first reads that page from the SSD (write-allocate). Expect 1.4–2.4 GB/s, about 15–20 s for 28 GB.
7. **Small datasets never reach the SSD.** ogbn-products fits in device DRAM even at the 4 GB minimum, so its numbers describe a DRAM tier. Use ogbn-papers100M or larger for SSD results.
8. **Do not point the GPU at device memory directly** (zero-copy). A GPU load that misses to the SSD waits about 100 µs or more, and that has not been tested. This code reads the device from the CPU on purpose.

## Comparing with BaM and GIDS

The method moves both the CSC graph and the features to the CXL device; GIDS keeps the graph in host memory. Run `--graph-on host` as well: with it only the feature path differs from GIDS, which separates the effect of moving the graph from the effect of the feature tier.

| | GIDS (BaM) | persistent_cxl_cache |
| --- | --- | --- |
| Graph structure (CSC) | pinned host memory, GPU sampling | CXL device (`--graph-on card`); `split` = `indptr` in HBM; `host` = GIDS placement |
| Feature table | NVMe SSD, read by GPU threads through BaM | CXL device; device DRAM caches its SSD in hardware |
| Read granularity | 4 KB cache lines | 512 KB pages (64 KB announced) |
| GPU-side cache | BaM software cache with window buffering | static HBM cache of the highest-degree rows |
| Hot-node buffer | constant CPU buffer | device DRAM cache |
| Who issues I/O | GPU threads | host CPU: one async prefetch per page, then a gather |

**Keep these identical across runs:** dataset files, feature dtype, batch size, fanouts, number of layers, GPU model, and the HBM cache budget. GIDS's GPU cache and our `--hbm-gb` should get the same bytes.

**BaM needs its own NVMe SSD.** It takes the SSD over from the kernel NVMe driver, so the drive cannot hold a mounted filesystem the system uses.

**Report for each configuration:**

- time per batch, split into sampling, feature gather and compute;
- SSD bytes read per batch (ours: `card SSD: read` from `bench_tiered.py`; BaM: its I/O counters);
- read amplification and cache hit rate;
- the cache size needed for a target SSD traffic, using `page_sim.py` at 4K (BaM) and 512K / 64K (device) on the same batches.
