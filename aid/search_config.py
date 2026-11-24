"""What the hyperparameter search (search.py) trains on: a stratified subset of the training images (or the training
images of a protocol of split_protocol.py, e.g. tr50), a few augmentations and the superpixel counts under test, and
where the graphs of each count are.

Imported in the graph-building environment too, so it only needs numpy and config.py.
"""
import os

import numpy as np

from config import IMAGES_DIR, N_AUGMENTATIONS, SPLIT_SIZES, graph_path
from split_protocol import SOURCES, load_protocol, protocol_fold

# Superpixel counts: 300 to 1400 spaced geometrically (ratio ~1.47, superpixel side 35 to 16 px). On 60 val images
# the explained variation of the pixel colors and of the VGG features grows by about the same amount at each doubling
# of the count, with no saturation in this range (logs/superpixel_eval/). 1400 superpixels of a 600x600 image have
# the size of the 300 of a 256x256 RESISC45 image, the best count of the RESISC45 search.
NODE_VALUES = (300, 450, 650, 950, 1400)
# Counts of the search of a protocol (e.g. tr50): around 300, the count chosen by the search above, which was the lower
# bound of its range (950 tied with the reference, 1400 never did well), with the same ratio ~1.4-1.5
PROTOCOL_NODE_VALUES = (150, 200, 300, 450, 650)
SEARCH_FRACTION = 0.1   # of the training images, in the search
CONFIRM_FRACTION = 0.3  # in the confirmation of the best configurations; contains the search subset
# Entries of AUGMENTATIONS: identity, rotations 270 and 90, Gaussian blur, color jitter, occlusion, affine and
# horizontal flip. None is multiscale, so with base n their target is n and their maps are the same ones that
# build_dataset.py --nodes n writes.
SEARCH_AUGMENTATIONS = (0, 1, 3, 4, 6, 17, 37, 42)


def subset_images(fraction, seed=0):
    """Training images of the subset, stratified by class; a smaller fraction is contained in a larger one."""
    labels = np.load(os.path.join(IMAGES_DIR, "train_labels.npy"))
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


def protocol_images(protocol, fold, split):
    """Images of `split` of a protocol with validation fold `fold`: indices in the torchgeo-bench train split, where
    all the labelled images of the protocols of split_protocol.py come from."""
    source, image = protocol_fold(protocol, fold)[split]
    assert (source == SOURCES.index("train")).all()
    return image


def protocol_train_graphs(protocol, fold):
    """Graph indices (train split) of the training images of the fold, with every augmentation of the search."""
    return [int(i) * N_AUGMENTATIONS + k for i in protocol_images(protocol, fold, "train") for k in SEARCH_AUGMENTATIONS]


def protocol_val_graphs(protocol, fold):
    """Unaugmented graph (augmentation 0, the identity) of the validation images of the fold, in the train split."""
    return [int(i) * N_AUGMENTATIONS for i in protocol_images(protocol, fold, "val")]


def protocol_labelled_graphs(protocol):
    """Graphs of every fold: the labelled images with every augmentation of the search (0, of validation, included)."""
    sets = load_protocol(protocol)
    assert all((sets[s][0] == SOURCES.index("train")).all() for s in ("train", "val"))
    images = np.sort(np.concatenate([sets["train"][1], sets["val"][1]]))
    return [int(i) * N_AUGMENTATIONS + k for i in images for k in SEARCH_AUGMENTATIONS]


def graph_file(split, idx, n_nodes):
    """The search maps are written where the full dataset with that base count goes (config.processed_dir), so
    build_dataset.py --nodes n skips them later."""
    return graph_path(split, idx, n_nodes)
