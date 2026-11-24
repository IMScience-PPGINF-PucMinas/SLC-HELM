"""Graph dataset: each sample keeps the (augmented) image and its superpixel map.

Node features are not stored; the model computes them from `img` and `pix2node`
(VGG features aggregated per superpixel), so they can be learned end-to-end.
"""
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from config import N_AUGMENTATIONS, N_NODES, SPLIT_SIZES, graph_path


class SuperpixelData(Data):
    """Data with a pixel -> node map.

    Fields: img [1, 3, H, W] uint8, pix2node [H*W] long, edge_index [2, E], y [1].
    """

    def __inc__(self, key, value, *args, **kwargs):
        if key == "pix2node":
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)


class EuroSATGraphs(Dataset):
    """Graphs of a split; train graph k of image i is train_{i * N_AUGMENTATIONS + k}. n_nodes is the base number of
    superpixels of the graphs (build_dataset.py --nodes)."""

    def __init__(self, split, n_nodes=N_NODES):
        self.split = split
        self.n_nodes = n_nodes
        self.size = SPLIT_SIZES[split] * (N_AUGMENTATIONS if split == "train" else 1)

    def __len__(self):
        return self.size

    def path(self, idx):
        return graph_path(self.split, idx, self.n_nodes)

    def __getitem__(self, idx):
        data = torch.load(self.path(idx), weights_only=False)
        if isinstance(data, dict):  # build_dataset.py stores plain tensors
            data = SuperpixelData(**data)
        data.pix2node = data.pix2node.long()  # stored as int16 to save disk
        return data


def init_worker(_):
    """DataLoader worker_init_fn: one thread per worker (same interface as the RESISC45 and AID pipelines)."""
    torch.set_num_threads(1)
