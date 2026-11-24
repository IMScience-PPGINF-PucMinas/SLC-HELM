"""Checks the graphs before training, in the environment that will train:
- every graph file exists;
- for a sample of graphs of every augmentation (and of val/test), the image rebuilt by dataset.py is the
  one its superpixel map was built from (MD5 recorded by build_dataset.py);
- the maps are consistent: labels 0..N-1 all used, edges inside the graph and without self-loops;
- the number of superpixels is the DISF target (reported only: DISF may end with fewer in flat images).

    python check_dataset.py [--per-aug 20] [--nodes 300]
    python check_dataset.py --nodes 300 --protocol tr20   # only the graphs of splits/tr20.npz (e.g. a copy of them)
"""
import argparse
import hashlib
import os
import random

import numpy as np

from augmentations import target_nodes
from config import N_AUGMENTATIONS, N_NODES, SPLIT_SIZES, processed_dir
from dataset import RESISCGraphs
from split_protocol import source_graphs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-aug", type=int, default=20, help="train images checked per augmentation (and val/test graphs)")
    p.add_argument("--nodes", type=int, default=N_NODES, help="base number of superpixels (build_dataset.py --nodes)")
    p.add_argument("--protocol", help="check only the graphs of splits/<protocol>.npz (split_protocol.py)")
    args = p.parse_args()
    rng = random.Random(0)
    directory = processed_dir(args.nodes)
    existing = set(os.listdir(directory))
    wanted = source_graphs(args.protocol) if args.protocol else None
    ok = True
    for split in SPLIT_SIZES:
        ds = RESISCGraphs(split, n_nodes=args.nodes)
        graph_ids = wanted[split] if wanted is not None else range(len(ds))
        missing = sum(f"{split}_{idx}.npz" not in existing for idx in graph_ids)
        if split == "train" and wanted is None:
            groups = {k: [rng.randrange(SPLIT_SIZES[split]) * N_AUGMENTATIONS + k for _ in range(args.per_aug)]
                      for k in range(N_AUGMENTATIONS)}
        elif split == "train":
            by_aug = {k: graph_ids[graph_ids % N_AUGMENTATIONS == k].tolist() for k in range(N_AUGMENTATIONS)}
            groups = {k: rng.sample(g, min(args.per_aug, len(g))) for k, g in by_aug.items() if g}
        else:
            groups = {0: rng.sample(list(graph_ids), min(args.per_aug, len(graph_ids)))}
        bad_md5, bad_map, other_count = {}, {}, {}
        for k, ids in groups.items():
            for idx in ids:
                if f"{split}_{idx}.npz" not in existing:
                    continue
                z = np.load(os.path.join(directory, f"{split}_{idx}.npz"))
                image = z["img"] if "img" in z.files else ds.image(idx)
                if hashlib.md5(image.tobytes()).digest() != z["img_md5"].tobytes():
                    bad_md5[k] = bad_md5.get(k, 0) + 1
                pix2node, edges = z["pix2node"].astype(np.int64), z["edge_index"].astype(np.int64)
                n = int(pix2node.max()) + 1
                if (pix2node.shape != image.shape[:2] or len(np.unique(pix2node)) != n or edges.min() < 0
                        or edges.max() >= n or (edges[0] == edges[1]).any()):
                    bad_map[k] = bad_map.get(k, 0) + 1
                if n != (target_nodes(k, args.nodes) if split == "train" else args.nodes):
                    other_count[k] = other_count.get(k, 0) + 1
        checked = sum(len(v) for v in groups.values())
        print(f"{split}: {len(graph_ids)} graphs, {missing} missing | {checked} checked: image MD5 differs for "
              f"{bad_md5 or 'none'} | inconsistent maps: {bad_map or 'none'} | "
              f"number of superpixels other than the target: {other_count or 'none'}")
        ok &= missing == 0 and not bad_md5 and not bad_map
    print("OK" if ok else "PROBLEMS FOUND (see above)")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
