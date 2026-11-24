"""Builds the superpixel maps of the hyperparameter search (run in the environment with DISF). The images and
their augmentations are the same as in data/processed, only the number of superpixels changes; the maps with
the usual count are already there. Existing files are skipped, so an interrupted build resumes where it stopped.

    python search_build.py --fraction 0.1                    # val and subset of train, all counts of the search
    python search_build.py --fraction 0.3 --nodes 150 200    # larger subset of train, some counts
    python search_build.py --protocol tr20                   # labelled images of splits/tr20.npz (all its folds)

Maps of a count whose full dataset exists (data/processed_n300) are hard links to it (same image and target).
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
from config import IMAGES_DIR, N_AUGMENTATIONS, N_NODES, graph_path
from search_config import (NODE_VALUES, SEARCH_AUGMENTATIONS, SEARCH_DIR, graph_file, protocol_labelled_graphs,
                           train_graphs, val_graphs)

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
    if os.path.exists(graph_path(split, idx, n_nodes)):  # all search augmentations have the base target
        os.link(graph_path(split, idx, n_nodes), path)
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
    parser.add_argument("--nodes", type=int, nargs="*", default=list(NODE_VALUES))
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    parser.add_argument("--graphs-per-worker", type=int, default=400,
                        help="graphs built by a worker process before it is replaced (bounds DISF's memory leak)")
    args = parser.parse_args()
    assert all(AUGMENTATIONS[k][1] == N_NODES for k in SEARCH_AUGMENTATIONS)

    nodes = [n for n in args.nodes if n != N_NODES]
    for n in nodes:
        os.makedirs(os.path.join(SEARCH_DIR, f"n{n}"), exist_ok=True)
    if args.protocol:
        sets = (("train", protocol_labelled_graphs(args.protocol)),)
    else:
        sets = (("val", val_graphs()), ("train", train_graphs(args.fraction)))
    tasks = [(split, idx, n) for n in nodes for split, graphs in sets for idx in graphs]
    chunk = 16
    with mp.Pool(args.workers, initializer=_init_worker, maxtasksperchild=max(1, args.graphs_per_worker // chunk)) as pool:
        for _ in tqdm(pool.imap_unordered(process, tasks, chunksize=chunk), total=len(tasks), mininterval=30):
            pass


if __name__ == "__main__":
    main()
