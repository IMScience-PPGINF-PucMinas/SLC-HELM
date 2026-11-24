"""DISF superpixels and their Region Adjacency Graph."""
import numpy as np
from disf import DISF_Superpixels

from config import DISF_INIT_SEEDS


def segment(image, n_nodes):
    """DISF superpixels of an HxWx3 uint8 image as an HxW map with labels 0..N-1."""
    labels, _ = DISF_Superpixels(np.int32(image), DISF_INIT_SEEDS, n_nodes)
    # Relabel so labels are contiguous even if DISF skips one.
    _, labels = np.unique(labels, return_inverse=True)
    return labels.reshape(image.shape[:2])


def region_adjacency(labels):
    """Undirected 8-connected RAG of a label map, as a (2, 2E) edge array with both directions."""
    pairs = [
        (labels[:, :-1], labels[:, 1:]),      # right
        (labels[:-1, :], labels[1:, :]),      # down
        (labels[:-1, :-1], labels[1:, 1:]),   # down-right
        (labels[:-1, 1:], labels[1:, :-1]),   # down-left
    ]
    src = np.concatenate([a.ravel() for a, _ in pairs])
    dst = np.concatenate([b.ravel() for _, b in pairs])
    keep = src != dst
    edges = np.stack([np.minimum(src[keep], dst[keep]), np.maximum(src[keep], dst[keep])])
    edges = np.unique(edges, axis=1)
    return np.concatenate([edges, edges[::-1]], axis=1).astype(np.int64)


def build_graph(image, n_nodes):
    """Returns (pix2node [H*W], edge_index [2, E]) for one image."""
    labels = segment(image, n_nodes)
    return labels.ravel(), region_adjacency(labels)
