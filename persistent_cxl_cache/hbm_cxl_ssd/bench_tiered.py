"""bench_tiered.py - mini-batch GNN loop on HBM + CXL cache (XCENA DRAM) + SSD.

    python bench_tiered.py --data /mnt/data/ogbn_products_csr --hbm-gb 0.2 --batches 50
    python bench_tiered.py --data ... --hbm-gb 0 --no-prefetch          # ablation
    python bench_tiered.py --data ... --train                            # 3-layer SAGE (needs PyG)
    python bench_tiered.py --data ... --graph-on split                   # indptr in HBM, indices on the device
    python bench_tiered.py --data ... --graph-on host                    # GIDS placement: graph in host DRAM

--data must hold indptr.npy (int64, N+1), indices.npy (int64, E) - the CSC graph - feat.npy (N, D),
and optionally
labels.npy (N,) and train_idx.npy. prepare_ogbn.py writes this layout.

Reported per run: HBM hit rate, distinct 512 KB pages requested from the card, time split
(sample / card gather / H2D / compute), and the card's SSD bytes read and written over the run
(from `xcena_cli im get-smart`; the SSD is only touched when the data is not in the card's DRAM).
"""
import argparse
import os
import time

import numpy as np
import torch

from tiered_store import SsdCounter, Stats, TieredFeatures, TieredGraph, graph_files, hot_by_degree, im


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--device-id", type=int, default=0, help="XCENA device id")
    p.add_argument("--hbm-gb", type=float, default=0.5, help="HBM feature cache size; 0 = off")
    p.add_argument("--fanouts", type=str, default="10,10,10")
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--batches", type=int, default=50)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--no-prefetch", action="store_true")
    p.add_argument("--graph-on", choices=["card", "split", "host"], default="card",
                   help="card = indptr+indices on the device; split = indptr in HBM; host = GIDS placement")
    p.add_argument("--train", action="store_true", help="also run a 3-layer GraphSAGE step")
    args = p.parse_args()

    dev = torch.device("cuda")
    fanouts = [int(x) for x in args.fanouts.split(",")]
    im.init(args.device_id)
    ssd = SsdCounter(args.device_id)
    stats = Stats()
    prefetch = not args.no_prefetch

    print(f"[load] features -> device (CXL cache + SSD); graph (CSC) placement: {args.graph_on}", flush=True)
    indptr_path, indices_path = graph_files(args.data)
    graph = TieredGraph(indptr_path, indices_path, dev, prefetch=prefetch, stats=stats, graph_on=args.graph_on)
    feat_path = os.path.join(args.data, "feat.npy")
    fmeta = np.load(feat_path, mmap_mode="r")
    row_bytes = fmeta.shape[1] * fmeta.dtype.itemsize
    hot = hot_by_degree(graph, args.hbm_gb * (1 << 30), row_bytes)
    feats = TieredFeatures(feat_path, dev, hot_ids=hot, prefetch=prefetch, stats=stats)
    print(f"[load] N={graph.num_nodes} E={graph.indices_t.shape[0]} D={feats.dim} "
          f"HBM cache {hot.numel()} rows ({hot.numel() / graph.num_nodes:.1%} of nodes)", flush=True)

    tpath = os.path.join(args.data, "train_idx.npy")
    train_idx = (torch.from_numpy(np.load(tpath)) if os.path.exists(tpath)
                 else torch.arange(graph.num_nodes)).to(dev)
    labels = None
    lpath = os.path.join(args.data, "labels.npy")
    if os.path.exists(lpath):
        labels = torch.from_numpy(np.load(lpath).astype(np.int64).reshape(-1)).to(dev)

    model = opt = None
    if args.train:
        from torch_geometric.nn import SAGEConv  # imported only when training

        class SAGE(torch.nn.Module):
            def __init__(self, i, h, o, layers):
                super().__init__()
                dims = [i] + [h] * (layers - 1) + [o]
                self.convs = torch.nn.ModuleList(SAGEConv(dims[k], dims[k + 1]) for k in range(layers))

            def forward(self, x, ei):
                for k, conv in enumerate(self.convs):
                    x = conv(x, ei)
                    if k < len(self.convs) - 1:
                        x = torch.relu(x)
                return x

        n_cls = int(labels.max().item()) + 1 if labels is not None else 2
        model = SAGE(feats.dim, 256, n_cls, len(fanouts)).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    gen = torch.Generator(device=dev)
    gen.manual_seed(0)
    perm = train_idx[torch.randperm(train_idx.numel(), device=dev, generator=gen)]
    t_compute = 0.0
    times = []
    rd0 = wr0 = None
    for b in range(args.warmup + args.batches):
        if b == args.warmup:  # start measuring after warm-up
            stats.reset()
            t_compute = 0.0
            rd0, wr0 = ssd.read()
        seeds = perm[(b * args.batch) % perm.numel():][:args.batch]
        t0 = time.perf_counter()
        n_id, edge_index = graph.sample(seeds, fanouts, generator=gen)
        x = feats.gather(n_id)
        if model is not None:
            tc = time.perf_counter()
            out = model(x.float(), edge_index)[: seeds.numel()]
            y = labels[seeds] if labels is not None else torch.zeros_like(seeds)
            loss = torch.nn.functional.cross_entropy(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            t_compute += time.perf_counter() - tc
        if b >= args.warmup:
            times.append(time.perf_counter() - t0)

    rd1, wr1 = ssd.read()
    n = len(times)
    print(f"\n[result] {n} batches, batch {args.batch}, fanouts {fanouts}, prefetch {prefetch}")
    print(f"  per batch: median {np.median(times) * 1e3:.1f} ms, mean {np.mean(times) * 1e3:.1f} ms")
    print(f"  {stats.summary()} (totals over {n} batches)")
    if model is not None:
        print(f"  compute {t_compute * 1e3:.0f} ms total")
    if rd0 is not None and rd1 is not None:
        print(f"  card SSD: read {(rd1 - rd0) / 1e6:.1f} MB, write {(wr1 - wr0) / 1e6:.1f} MB over the run")
    else:
        print("  card SSD counters: n/a (xcena_cli not runnable from this user?)")


if __name__ == "__main__":
    main()
