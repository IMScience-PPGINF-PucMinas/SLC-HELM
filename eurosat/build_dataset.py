"""Builds the graph dataset in data/processed/ (run in the environment with DISF).

    python build_dataset.py --split val
    python build_dataset.py --split test
    python build_dataset.py --split train             # all 51 augmentations
    python build_dataset.py --split train --aug 0 1 2 # only some of them
    python build_dataset.py --split train --nodes 100 # base of 100 superpixels, in data/processed_n100

Train graph k of image i is saved as train_{i * N_AUGMENTATIONS + k}.pt. With --nodes other than N_NODES, val,
test and most augmentations get that number of superpixels; the multiscale augmentations (those with their own
target in augmentations.py) keep their targets. Existing files are skipped, so an interrupted build resumes where
it stopped.
"""
import argparse
import multiprocessing as mp
import os
import random

import numpy as np
import torch
from torchvision.datasets import EuroSAT
from tqdm import tqdm

from augmentations import AUGMENTATIONS
from config import IMG_SIZE, INDICES_DIR, N_AUGMENTATIONS, N_NODES, RAW_DIR, SPLIT_SIZES, graph_path, processed_dir
from superpixels import build_graph

_eurosat = None
_indices = {}


def _init_worker():
    global _eurosat
    torch.set_num_threads(1)
    _eurosat = EuroSAT(RAW_DIR, download=False)
    for split in SPLIT_SIZES:
        _indices[split] = np.load(os.path.join(INDICES_DIR, f"{split}_indices.npy"))


def make_graph(image, label, n_nodes):
    assert image.shape == (IMG_SIZE, IMG_SIZE, 3), image.shape
    pix2node, edge_index = build_graph(image, n_nodes)
    # Plain tensors instead of a PyG Data object, so a graph built in one environment (DISF needs
    # NumPy 1.x) loads in another PyG version; dataset.py turns it into a SuperpixelData.
    return dict(
        img=torch.from_numpy(np.array(image)).permute(2, 0, 1).contiguous().unsqueeze(0),
        pix2node=torch.from_numpy(pix2node.astype(np.int16)),
        edge_index=torch.from_numpy(edge_index),
        y=torch.tensor([label]),
        num_nodes=int(pix2node.max()) + 1,
    )


def process(task):
    """task = (split, image index i in the split, augmentation k, base number of superpixels)."""
    split, i, k, nodes = task
    idx = i * N_AUGMENTATIONS + k if split == "train" else i
    path = graph_path(split, idx, nodes)
    if os.path.exists(path):
        return
    pil_image, label = _eurosat[int(_indices[split][i])]
    image = np.asarray(pil_image)
    n_nodes = nodes
    if split == "train":
        # Deterministic randomness per (image, augmentation), independent of worker scheduling.
        random.seed(idx)
        np.random.seed(idx)
        torch.manual_seed(idx)
        transform, aug_nodes = AUGMENTATIONS[k]
        image = transform(image)
        n_nodes = aug_nodes or nodes
    tmp = path[:-3] + ".tmp.pt"
    torch.save(make_graph(image, label, n_nodes), tmp)
    os.replace(tmp, path)


def run(tasks, workers):
    with mp.Pool(workers, initializer=_init_worker) as pool:
        for _ in tqdm(pool.imap_unordered(process, tasks, chunksize=64), total=len(tasks), mininterval=10):
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=list(SPLIT_SIZES))
    parser.add_argument("--aug", type=int, nargs="*", default=list(range(N_AUGMENTATIONS)),
                        help="augmentations to generate (train only)")
    parser.add_argument("--nodes", type=int, default=N_NODES,
                        help="base number of superpixels (the multiscale augmentations keep their targets)")
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    args = parser.parse_args()

    os.makedirs(processed_dir(args.nodes), exist_ok=True)
    augs = args.aug if args.split == "train" else [0]
    run([(args.split, i, k, args.nodes) for k in augs for i in range(SPLIT_SIZES[args.split])], args.workers)


if __name__ == "__main__":
    main()
