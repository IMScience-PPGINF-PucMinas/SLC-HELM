"""Builds the superpixel maps of the RESISC45 graphs in data/processed/ (run prepare_images.py first).

    python build_dataset.py --split val
    python build_dataset.py --split test
    python build_dataset.py --split train             # all 51 augmentations
    python build_dataset.py --split train --aug 0 1 2 # only some of them
    python build_dataset.py --split train --nodes 300 # base of 300 superpixels, in data/processed_n300

Train graph k of image i is {split}_{i * N_AUGMENTATIONS + k}.npz. Each file stores the DISF map
(pix2node, int16 HxW), the RAG edges (int16), the label and the MD5 of the augmented image; the image
itself is rebuilt from data/images when the graph is loaded (augmentations.augment), except for the
augmentations in STORED_AUGMENTATIONS, whose image is stored too. With --nodes other than N_NODES, val, test
and most augmentations get that number of superpixels; the multiscale augmentations keep their targets
(augmentations.target_nodes), so their maps are hard links to those of data/processed. Existing files are skipped, so an
interrupted build resumes where it stopped. Each worker process is replaced after a few hundred graphs,
because DISF leaks memory (~0.7 MB per 256x256 image), and runs single-threaded.
"""
import argparse
import hashlib
import multiprocessing as mp
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from augmentations import FIXED_TARGET_AUGMENTATIONS, STORED_AUGMENTATIONS, augment, target_nodes
from config import IMAGES_DIR, N_AUGMENTATIONS, N_NODES, SPLIT_SIZES, graph_path, processed_dir
from superpixels import build_graph

_images = _labels = None


def _init_worker(split):
    global _images, _labels
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    _images = np.load(os.path.join(IMAGES_DIR, f"{split}.npy"), mmap_mode="r")
    _labels = np.load(os.path.join(IMAGES_DIR, f"{split}_labels.npy"))


def write_graph(path, image, n_nodes, label, store_image=False):
    """Builds the superpixel map of `image` and writes its npz atomically."""
    pix2node, edge_index = build_graph(image, n_nodes)
    fields = dict(pix2node=pix2node.astype(np.int16), edge_index=edge_index.astype(np.int16),
                  y=np.int64(label), img_md5=np.frombuffer(hashlib.md5(image.tobytes()).digest(), np.uint8))
    if store_image:
        fields["img"] = image
    tmp = path[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, **fields)
    os.replace(tmp, path)


def process(task):
    split, i, k, nodes = task
    idx = i * N_AUGMENTATIONS + k if split == "train" else i
    path = graph_path(split, idx, nodes)
    if os.path.exists(path):
        return
    if split == "train" and nodes != N_NODES and k in FIXED_TARGET_AUGMENTATIONS and os.path.exists(graph_path(split, idx)):
        os.link(graph_path(split, idx), path)  # same image and target as in data/processed
        return
    if split == "train":
        image, n_nodes = augment(np.array(_images[i]), idx, k)[0], target_nodes(k, nodes)
    else:
        image, n_nodes = np.ascontiguousarray(_images[i]), nodes
    write_graph(path, image, n_nodes, _labels[i], store_image=split == "train" and k in STORED_AUGMENTATIONS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=list(SPLIT_SIZES))
    parser.add_argument("--aug", type=int, nargs="*", default=list(range(N_AUGMENTATIONS)),
                        help="augmentations to generate (train only)")
    parser.add_argument("--nodes", type=int, default=N_NODES,
                        help="base number of superpixels (the multiscale augmentations keep their targets)")
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    parser.add_argument("--graphs-per-worker", type=int, default=400,
                        help="graphs built by a worker process before it is replaced (bounds DISF's memory leak)")
    args = parser.parse_args()

    os.makedirs(processed_dir(args.nodes), exist_ok=True)
    augs = args.aug if args.split == "train" else [0]
    tasks = [(args.split, i, k, args.nodes) for k in augs for i in range(SPLIT_SIZES[args.split])]
    chunk = 16
    with mp.Pool(args.workers, initializer=_init_worker, initargs=(args.split,),
                 maxtasksperchild=max(1, args.graphs_per_worker // chunk)) as pool:
        for _ in tqdm(pool.imap_unordered(process, tasks, chunksize=chunk), total=len(tasks), mininterval=30):
            pass


if __name__ == "__main__":
    main()
