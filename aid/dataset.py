"""Graph dataset: each sample is the (augmented) image and its superpixel map.

Node features are not stored; the model computes them from `img` and `pix2node` (VGG features
aggregated per superpixel). The superpixel maps come from build_dataset.py; the images from
data/images (prepare_images.py), augmented again here with the same seeds as in the build.
"""
import hashlib
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from augmentations import augment
from config import IMAGES_DIR, N_AUGMENTATIONS, N_NODES, SPLIT_SIZES, graph_path
from split_protocol import SOURCES
from split_protocol import graphs as protocol_graphs


class SuperpixelData(Data):
    """Data with a pixel -> node map.

    Fields: img [1, 3, H, W] uint8, pix2node [H*W] int32 (half the transfer of int64; the model
    converts it), edge_index [2, E], y [1].
    """

    def __inc__(self, key, value, *args, **kwargs):
        if key == "pix2node":
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)


class AIDGraphs(Dataset):
    """verify=True checks the MD5 of every image against the one recorded when its map was built;
    n_nodes is the base number of superpixels of the graphs (build_dataset.py --nodes)."""

    def __init__(self, split, verify=False, n_nodes=N_NODES):
        self.split = split
        self.n_nodes = n_nodes
        self.size = SPLIT_SIZES[split] * (N_AUGMENTATIONS if split == "train" else 1)
        self.verify = verify
        self._images = None  # opened lazily, once per DataLoader worker

    def __len__(self):
        return self.size

    def image(self, idx):
        """The (augmented) image of graph idx, as an HxWx3 uint8 array."""
        if self._images is None:
            self._images = np.load(os.path.join(IMAGES_DIR, f"{self.split}.npy"), mmap_mode="r")
        if self.split != "train":
            return np.array(self._images[idx])
        i, k = divmod(idx, N_AUGMENTATIONS)
        return augment(np.array(self._images[i]), idx, k)[0]

    def path(self, idx):
        return graph_path(self.split, idx, self.n_nodes)

    def __getitem__(self, idx):
        z = np.load(self.path(idx))
        image = z["img"] if "img" in z.files else self.image(idx)
        if self.verify and hashlib.md5(image.tobytes()).digest() != z["img_md5"].tobytes():
            raise RuntimeError(f"{self.split} graph {idx}: the image differs from the one its superpixel map was "
                               "built from (different library versions?); run check_dataset.py")
        pix2node = torch.from_numpy(z["pix2node"].astype(np.int32)).flatten()
        return SuperpixelData(img=torch.from_numpy(image).permute(2, 0, 1).contiguous().unsqueeze(0),
                              pix2node=pix2node, edge_index=torch.from_numpy(z["edge_index"].astype(np.int64)),
                              y=torch.tensor([int(z["y"])]), num_nodes=int(pix2node.max()) + 1)


class ProtocolGraphs(Dataset):
    """Split of a protocol of split_protocol.py (e.g. tr50: 50% of the images for training/validation, the rest for
    test). Its graphs are those of the torchgeo splits (AIDGraphs), so the maps, images and seeds are the same."""

    def __init__(self, protocol, split, verify=False, n_nodes=N_NODES):
        self.source, self.graph = protocol_graphs(protocol, split)
        self.sources = [AIDGraphs(s, verify, n_nodes) for s in SOURCES]

    def __len__(self):
        return len(self.graph)

    def __getitem__(self, idx):
        return self.sources[self.source[idx]][int(self.graph[idx])]


def graphs(split, n_nodes=N_NODES, protocol=None, verify=False):
    """Split of the torchgeo protocol (protocol=None) or of a protocol of split_protocol.py."""
    if protocol is None:
        return AIDGraphs(split, verify, n_nodes)
    return ProtocolGraphs(protocol, split, verify, n_nodes)


def init_worker(_):
    """DataLoader worker_init_fn: one OpenCV thread per worker (PyTorch already uses one)."""
    cv2.setNumThreads(1)
