"""Builds the superpixel maps of the hyperparameter search (run in the environment with DISF). The images and
their augmentations are the same as in the full dataset, only the number of superpixels changes, and each map is
written where build_dataset.py --nodes n writes it (data/processed_n{n}), so the full build reuses it. Existing
files are skipped, so an interrupted build resumes where it stopped.

    python search_build.py --fraction 0.1                    # val and subset of train, all counts of the search
    python search_build.py --fraction 0.3 --nodes 450 950    # larger subset of train, some counts
    python search_build.py --protocol tr50                   # labelled images of splits/tr50.npz (all its folds),
                                                             # counts of search_config.PROTOCOL_NODE_VALUES
"""
import argparse
import multiprocessing as mp
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from augmentations import AUGMENTATIONS, augment
from build_dataset import write_graph
from config import IMAGES_DIR, N_AUGMENTATIONS, N_NODES, processed_dir
from search_config import (NODE_VALUES, PROTOCOL_NODE_VALUES, SEARCH_AUGMENTATIONS, graph_file,
                           protocol_labelled_graphs, train_graphs, val_graphs)

_images, _labels = {}, {}


def _init_worker():
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    for split in ("train", "val"):
        _images[split] = np.load(os.path.join(IMAGES_DIR, f"{split}.npy"), mmap_mode="r")
        _labels[split] = np.load(os.path.join(IMAGES_DIR, f"{split}_labels.npy"))


def process(task):
    split, idx, n_nodes = task
    path = graph_file(split, idx, n_nodes)
    if os.path.exists(path):
        return
    if split == "train":
        i, k = divmod(idx, N_AUGMENTATIONS)
        image = augment(np.array(_images[split][i]), idx, k)[0]
    else:
        i, image = idx, np.array(_images[split][idx])
    write_graph(path, image, n_nodes, _labels[split][i])


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fraction", type=float, help="fraction of the training images (and the validation set)")
    source.add_argument("--protocol", help="protocol of split_protocol.py (splits/<name>.npz): its labelled images")
    parser.add_argument("--nodes", type=int, nargs="*",
                        help="default: NODE_VALUES, or PROTOCOL_NODE_VALUES with --protocol (search_config.py)")
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    parser.add_argument("--graphs-per-worker", type=int, default=100,
                        help="graphs built by a worker process before it is replaced (bounds DISF's memory leak)")
    args = parser.parse_args()
    assert all(AUGMENTATIONS[k][1] == N_NODES for k in SEARCH_AUGMENTATIONS)
    if args.nodes is None:
        args.nodes = list(PROTOCOL_NODE_VALUES if args.protocol else NODE_VALUES)

    for n in args.nodes:
        os.makedirs(processed_dir(n), exist_ok=True)
    if args.protocol:
        sets = (("train", protocol_labelled_graphs(args.protocol)),)
    else:
        sets = (("val", val_graphs()), ("train", train_graphs(args.fraction)))
    tasks = [(split, idx, n) for n in args.nodes for split, graphs in sets for idx in graphs]
    chunk = 16
    with mp.Pool(args.workers, initializer=_init_worker, maxtasksperchild=max(1, args.graphs_per_worker // chunk)) as pool:
        for _ in tqdm(pool.imap_unordered(process, tasks, chunksize=chunk), total=len(tasks), mininterval=30):
            pass


if __name__ == "__main__":
    main()
