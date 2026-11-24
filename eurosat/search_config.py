"""What the hyperparameter search (search.py) trains on: a stratified subset of the training images, a few
augmentations and the superpixel counts under test, and where the graphs of each count are.

Imported in the graph-building environment too, so it only needs numpy, torchvision and config.py.
"""
import os

import numpy as np

from config import INDICES_DIR, N_AUGMENTATIONS, RAW_DIR, SPLIT_SIZES, graph_path

# Superpixel counts: around 50, the count of the paper's runs, spaced geometrically (ratio ~1.4-1.5): superpixels of
# ~160 down to ~27 pixels in a 64x64 image. 100 is also the smaller count used by mHELMNet on EuroSAT.
NODE_VALUES = (25, 35, 50, 75, 100, 150)
SEARCH_FRACTION = 0.1   # of the training images, in the search (1620 images x 8 augmentations)
CONFIRM_FRACTION = 0.3  # in the confirmation of the best configurations; contains the search subset
# Entries of AUGMENTATIONS: identity, rotations 270 and 90, Gaussian blur, color jitter, occlusion, affine and
# horizontal flip. None is multiscale, so with base n their target is n and their graphs are the same ones that
# build_dataset.py --nodes n writes.
SEARCH_AUGMENTATIONS = (0, 1, 3, 4, 6, 17, 37, 42)


def train_labels():
    """Labels of the training images, in the order of indices/train_indices.npy."""
    from torchvision.datasets import EuroSAT  # ImageFolder: reads the file list, decodes nothing
    targets = np.asarray(EuroSAT(RAW_DIR, download=False).targets)
    return targets[np.load(os.path.join(INDICES_DIR, "train_indices.npy"))]


def subset_images(fraction, seed=0):
    """Training images of the subset, stratified by class; a smaller fraction is contained in a larger one."""
    labels = train_labels()
    rng = np.random.default_rng(seed)
    chosen = []
    for c in range(labels.max() + 1):
        images = rng.permutation(np.flatnonzero(labels == c))
        chosen.append(images[:round(len(images) * fraction)])
    return np.sort(np.concatenate(chosen))


def train_graphs(fraction):
    """Graph indices of the subset: every image with every augmentation of the search."""
    return [int(i) * N_AUGMENTATIONS + k for i in subset_images(fraction) for k in SEARCH_AUGMENTATIONS]


def val_graphs():
    return list(range(SPLIT_SIZES["val"]))


def graph_file(split, idx, n_nodes):
    """The search graphs are written where the full dataset with that base count goes (config.processed_dir), so
    build_dataset.py --nodes n skips them later."""
    return graph_path(split, idx, n_nodes)
