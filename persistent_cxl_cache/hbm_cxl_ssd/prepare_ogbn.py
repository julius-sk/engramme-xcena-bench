"""prepare_ogbn.py - write an OGB node-property dataset as the CSR .npy files bench_tiered.py reads.

    python prepare_ogbn.py --name ogbn-products --root /mnt/data/ogb --out /mnt/data/ogbn_products_csr
    python prepare_ogbn.py --name ogbn-papers100M ... --fp16      # halve the feature table

Output: rowptr.npy (int64, N+1), colind.npy (int64, E), feat.npy (N, D), labels.npy, train_idx.npy.
The graph is made undirected (both directions) and sorted by source, so the neighbors of node v
are colind[rowptr[v]:rowptr[v+1]] and sit next to each other on the card's 512 KB pages.
Needs the `ogb` package (pip install ogb); no PyG or DGL required.
"""
import argparse
import os

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="ogbn-products")
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fp16", action="store_true")
    args = p.parse_args()

    from ogb.nodeproppred import NodePropPredDataset

    ds = NodePropPredDataset(name=args.name, root=args.root)
    g, y = ds[0]
    n = int(g["num_nodes"])
    ei = g["edge_index"].astype(np.int64)
    src = np.concatenate([ei[0], ei[1]])
    dst = np.concatenate([ei[1], ei[0]])
    order = np.argsort(src, kind="stable")
    src, dst = src[order], dst[order]
    rowptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(src, minlength=n), out=rowptr[1:])

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "rowptr.npy"), rowptr)
    np.save(os.path.join(args.out, "colind.npy"), dst)
    feat = g["node_feat"].astype(np.float16 if args.fp16 else np.float32)
    np.save(os.path.join(args.out, "feat.npy"), feat)
    lab = np.nan_to_num(y.reshape(-1), nan=0).astype(np.int64)
    np.save(os.path.join(args.out, "labels.npy"), lab)
    np.save(os.path.join(args.out, "train_idx.npy"), ds.get_idx_split()["train"].astype(np.int64))
    print(f"N={n} E={dst.size} D={feat.shape[1]} feat {feat.nbytes / 2**30:.2f} GB "
          f"colind {dst.nbytes / 2**30:.2f} GB -> {args.out}")


if __name__ == "__main__":
    main()
