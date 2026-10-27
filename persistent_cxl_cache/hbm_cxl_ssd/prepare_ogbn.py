"""prepare_ogbn.py - write an OGB node-property dataset as the CSC .npy files the tools read.

    python prepare_ogbn.py --name ogbn-products --root ~/ogb --out ~/csc_products
    python prepare_ogbn.py --name ogbn-papers100M --root ~/ogb --out ~/csc_papers --fp16
    python prepare_ogbn.py --name ogbn-papers100M ... --format csc     # keep edge direction

Output: indptr.npy (int64, N+1), indices.npy (int64, E), feat.npy (N, D), labels.npy, train_idx.npy.

The graph is CSC: the in-neighbors of node v (the nodes with an edge u -> v, which send messages
to v) are indices[indptr[v]:indptr[v+1]], stored next to each other on the device's pages.
  --format sym (default): add the reverse of every edge first, then build CSC. For an undirected
                          dataset such as ogbn-products this is the usual graph (123.7M edges);
                          CSC and CSR are then the same arrays.
  --format csc:           keep OGB's edge direction (u -> v as given) and store in-neighbors only.
Needs the `ogb` package (pip install ogb); no PyG or DGL required.
"""
import argparse
import os

import numpy as np


def to_csc(src, dst, n):
    """indptr over destinations, indices = sources (in-neighbors), sorted by destination."""
    order = np.argsort(dst, kind="stable")
    indices = src[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(dst, minlength=n), out=indptr[1:])
    return indptr, indices


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="ogbn-products")
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--format", choices=["sym", "csc"], default="sym")
    p.add_argument("--fp16", action="store_true")
    args = p.parse_args()

    from ogb.nodeproppred import NodePropPredDataset

    ds = NodePropPredDataset(name=args.name, root=args.root)
    g, y = ds[0]
    n = int(g["num_nodes"])
    ei = g["edge_index"].astype(np.int64)
    src, dst = ei[0], ei[1]
    if args.format == "sym":
        src, dst = np.concatenate([src, dst]), np.concatenate([dst, src])
    indptr, indices = to_csc(src, dst, n)

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "indptr.npy"), indptr)
    np.save(os.path.join(args.out, "indices.npy"), indices)
    feat = g["node_feat"].astype(np.float16 if args.fp16 else np.float32)
    np.save(os.path.join(args.out, "feat.npy"), feat)
    lab = np.nan_to_num(y.reshape(-1), nan=0).astype(np.int64)
    np.save(os.path.join(args.out, "labels.npy"), lab)
    np.save(os.path.join(args.out, "train_idx.npy"), ds.get_idx_split()["train"].astype(np.int64))
    print(f"N={n} E={indices.size} D={feat.shape[1]} ({args.format}) feat {feat.nbytes / 2**30:.2f} GiB "
          f"indices {indices.nbytes / 2**30:.2f} GiB indptr {indptr.nbytes / 2**30:.2f} GiB -> {args.out}")


if __name__ == "__main__":
    main()
