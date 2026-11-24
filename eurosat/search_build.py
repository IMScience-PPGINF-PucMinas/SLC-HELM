"""Builds the graphs of the hyperparameter search (run in the environment with DISF). The images and their
augmentations are the same as in the full dataset, only the number of superpixels changes, and each graph is written
where build_dataset.py --nodes n writes it (data/processed_n{n}; data/processed for 50), so the full build reuses it.
Existing files are skipped, so an interrupted build resumes where it stopped.

    python search_build.py --fraction 0.1                    # val and subset of train, all counts of the search
    python search_build.py --fraction 0.3 --nodes 35 75      # larger subset of train, some counts
"""
import argparse
import multiprocessing as mp
import os

from augmentations import AUGMENTATIONS
from build_dataset import run
from config import N_AUGMENTATIONS, processed_dir
from search_config import NODE_VALUES, SEARCH_AUGMENTATIONS, train_graphs, val_graphs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fraction", type=float, required=True,
                        help="fraction of the training images (and the whole validation set)")
    parser.add_argument("--nodes", type=int, nargs="*", default=list(NODE_VALUES),
                        help="default: NODE_VALUES of search_config.py")
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    args = parser.parse_args()
    assert all(AUGMENTATIONS[k][1] is None for k in SEARCH_AUGMENTATIONS)  # target = base number of superpixels

    for n in args.nodes:
        os.makedirs(processed_dir(n), exist_ok=True)
    tasks = [("val", i, 0, n) for n in args.nodes for i in val_graphs()]
    tasks += [("train", *divmod(idx, N_AUGMENTATIONS), n) for n in args.nodes for idx in train_graphs(args.fraction)]
    run(tasks, args.workers)


if __name__ == "__main__":
    main()
