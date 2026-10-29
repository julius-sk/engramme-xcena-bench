# DRAN recall benchmark: CPU vs GPU vs XCENA MX1P

This package measures one DRAN recall over a memory of M = 10^6 entries along four
data paths ("lanes"). Lane C runs the scan inside an XCENA MX1P CXL memory card.
Lanes A, B2 and B3 read the key matrix from an NVMe SSD on every recall.

| Lane | Where the scan runs | Data path per recall | reference result |
| --- | --- | --- | --- |
| A | Host CPU, 96 threads | SSD → DRAM (O_DIRECT) | 648.0 ms |
| B2 | GPU | SSD → pinned DRAM → PCIe → HBM | 380.8 ms |
| B3 | GPU | SSD → HBM with GPUDirect Storage | 300.4 ms |
| C | XCENA MX1P MU cores | K/V resident in card DRAM; 2 KB in, 12.8 KB out | 161.0 ms |

All four lanes must report `(HIT)`: the planted probe column wins the recall.

## Workload

K is n × M = 1024 × 10^6 and V is o × M = 3072 × 10^6, both fp16 and column-major,
with k = 64 and β = 8. One recall runs 3 attractor iterations:

```
s  = Kᵀ ξ
w  = softmax(β · s[top-k])
ξ ← normalize(K[:, top-k] · w)      (repeat 3 times)
out = V[:, top-k] · w
```

K is 2.05 GB and V is 6.14 GB. They are synthetic Gaussian data with fixed seeds, so
every machine generates identical bytes. The benchmark measures latency and data
movement, not recall quality.

## Layout

| Path | Contents |
| --- | --- |
| `run_lanes.sh` | Runs one lane (or all) and logs to `results/` |
| `lanes/bench_retrieval_ab.py` | Data setup (`--setup`) and lane A |
| `lanes/bench_lane_b.py` | Lane B2 |
| `lanes/bench_lane_b_gds.py` | Lane B3 (kvikio, GPUDirect Storage) |
| `lanes/softmax_cost.py` | Optional: times each stage of one iteration on the CPU |
| `dran_mu/dran_retrieve_lanec.cpp` | Lane C host program |
| `dran_mu/mu_kernel/mu_dran.cpp` | Lane C MU kernel (scan, local top-k, merge, mix) |
| `dran_mu/dran_retrieve.cpp` | Experimental: `--place dram\|hybrid\|ssd` placement test; needs a card with an SSD tier in Computing mode |
| `dran_mu/im_probe.cpp` | InfiniteMemory probe: one API call at a time, with status, time, device counter deltas and page-level data checks (`--part ABCD`) |
| `dran_mu/im_examples.cpp` | Basic InfiniteMemory usage: plain memory, commit, prefetch before sparse reads, pin, preload → MU kernel → unload (`--ex 12345`); `--ex 6` reuses a preloaded slot by overwriting it from the host, without unload write-back |
| `dran_mu/im_tenant_bench.cpp` | Many users' K/V in InfiniteMemory, a few users' K computable at once; Zipf query stream, LRU swap by host copy into preloaded slots (`--swap copy`) or unload/preload (`--swap preload`); reports hot/cold latency and SSD bytes per swap |
| `persistent_cxl_cache/` | GNN data on GPU HBM + CXL DRAM cache + SSD: tiered store, mini-batch benchmark, offline cache-size simulator (see its README) |
| `setup_new_server.sh` | Staged setup of a new host (see below) |

## Quick start on the reference server

The server is already configured; only the Python environment and the lane C build
are per account.

```bash
cd ~/engramme-xcena-bench
bash setup_new_server.sh py              # once: venv at ~/venv-torch (~2 GB download)
(cd dran_mu && ./build.sh)               # once: builds dran_retrieve_lanec

source ~/venv-torch/bin/activate         # every new shell
export DATA=/mnt/data/bench_data T=3072  # data dir on the test SSD, MU core count

bash run_lanes.sh setup                  # only if $DATA/K.bin does not exist
bash run_lanes.sh A                      # or B2, B3, C, all
```

Each run writes `results/<lane>-<timestamp>.log`. Run one lane at a time on an idle
server.

Lane C directly, for changing parameters (`-t` = MU cores, `-b 3` = task batch size):

```bash
cd dran_mu
./dran_retrieve_lanec -n 100000  -t 3072 -b 3 --device 0   # quick check, a few seconds
./dran_retrieve_lanec -n 1000000 -t 3072 -b 3 --device 0   # full size
```

## Checking a run

| Lane | What to look for |
| --- | --- |
| A | `retrieval total` |
| B2 | `retrieval total`, `(HIT)` |
| B3 | `compat_preferred=False` (real GPUDirect Storage), `retrieval total`, `(HIT)` |
| C | `total retrieval`, `(HIT)`; `preload: NotSupported -> ... skipped` is expected on a card without an SSD tier |

- **B3 shows `compat_preferred=True`:** kvikio fell back to POSIX reads. Check `lsmod | grep nvidia_fs`.
- **C fails at startup:** `xcena_cli device-info 0` must show `Computable : Yes`, and `daxctl list` must show `"mode":"devdax"`.

## Setting up a new server

`setup_new_server.sh` runs numbered stages; run `bash setup_new_server.sh` with no
argument for the list.

```bash
bash setup_new_server.sh check                         # read-only survey
bash setup_new_server.sh apt                           # system packages
bash setup_new_server.sh sdk xcena_sdk_vX.Y.Z_ubuntu_24.04.tar.xz
bash setup_new_server.sh compute                       # InfiniteMemory card: enable Computing mode, then reboot
bash setup_new_server.sh validate                      # Computable, MU cores
bash setup_new_server.sh gpu                           # open driver + GDS check
bash setup_new_server.sh py                            # Python venv (+ kvikio if nvidia_fs is loaded)
```

Requirements, in short:

1. **SDK matches the card firmware** (`sudo xcena_cli fw-info 0`): firmware 1.0.12 → SDK v1.4.11, 1.0.11 → v1.4.10, 1.0.10 → v1.4.9.
2. **Kernel headers** for the running kernel, at `/lib/modules/$(uname -r)/build`; the mx_dma driver builds with DKMS.
3. **devdax host setup:** `sudo bash scripts/setup_host.sh` from the SDK (v1.4.10 and later), then reboot.
4. **GPU:** `nvidia-driver-595-server-open` plus `nvidia-fs-dkms=2.29.4-1` from NVIDIA's CUDA apt repository. GPUDirect Storage needs the open kernel module.
5. **Data SSD:** ext4 or xfs directly on a partition, not LVM. Check the negotiated link with `lspci -vv` (`LnkSta`); the read bandwidth sets the A/B results.

Do not flash card firmware without guidance from XCENA, and never flash a version
older than the one the card runs. The `fw` stage refuses a downgrade.

## Known limitations

- At M = 10^6, K fits in A100 HBM; a GPU with K resident would scan in about 4.4 ms. Lane C's advantage applies once the store outgrows HBM.
- One query and one device; concurrency and multi-card scaling are not measured.
- Lane A on the reference server is provisional: its CPU scan (354 ms) is slower than on a sister server with the same SSD (123 ms), cause not yet known.
- The test SSD runs at PCIe x2 (7.1 GB/s); a full x4 Gen5 link would shorten lanes A/B.
