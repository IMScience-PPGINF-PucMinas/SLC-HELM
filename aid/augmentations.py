"""The 51 training augmentations of the AID graphs (the same list as RESISC45 and EuroSAT).

AUGMENTATIONS[k] = (transform, n_nodes): transform maps an HxWx3 uint8 array to another one, n_nodes is
the DISF target. augment(image, seed, k) applies entry k with fixed seeds, so the same augmented image
can be built offline (for the superpixel map) and again when the graph is loaded (for the VGG features).
The random transforms are seeded per (image, augmentation),
and the random resized crop (k=41) keeps the IMG_SIZE x IMG_SIZE size so all images of a batch have the same shape.
"""
import contextlib
import random

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from config import IMG_SIZE, N_NODES

NODE_VARIATIONS = [100, 200, 300, 400, 500, 600]


def add_random_occlusion(image, min_size=30, max_size=100):
    """Black rectangle at a random position."""
    height, width = image.shape[:2]
    max_size = min(max_size, height, width)
    if min_size > max_size:
        min_size = max_size
    rect_width = random.randint(min_size, max_size)
    rect_height = random.randint(min_size, max_size)
    top_left_x = random.randint(0, width - rect_width)
    top_left_y = random.randint(0, height - rect_height)
    out = image.copy()
    out[top_left_y:top_left_y + rect_height, top_left_x:top_left_x + rect_width] = 0
    return out


def add_salt_and_pepper_noise(image, noise_ratio=0.02):
    """Salt-and-pepper noise on a fraction of the pixels."""
    out = image.copy()
    h, w, c = out.shape
    for _ in range(int(h * w * noise_ratio)):
        row, col = np.random.randint(0, h), np.random.randint(0, w)
        if np.random.rand() < 0.5:
            out[row, col] = [0, 0, 0]
        else:
            out[row, col] = [255, 255, 255]
    return out


def add_gaussian_noise(image, mean=0, std=25):
    """Additive Gaussian noise."""
    noisy = image.astype(float) + np.random.normal(mean, std, image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def pil(transform):
    """Wrap a torchvision PIL transform so it works on numpy arrays."""
    return lambda image: np.array(transform(Image.fromarray(image)))


def rotate(angle):
    return pil(T.RandomRotation((angle, angle)))


def rotate_occlude(angle, min_size, max_size):
    rot = rotate(angle)
    return lambda image: add_random_occlusion(rot(image), min_size, max_size)


def motion_blur(image, kernel_size=15):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[(kernel_size - 1) // 2, :] = 1.0 / kernel_size
    return cv2.filter2D(image, -1, kernel)


def exposure(image):
    gamma = np.random.uniform(0.5, 1.5)
    return np.clip(np.power(image / 255.0, gamma) * 255.0, 0, 255).astype(np.uint8)


def contrast(image):
    factor = np.random.uniform(0.5, 2.0)
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    return np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)


def identity(image):
    return image


AUGMENTATIONS = [
    (identity, N_NODES),                                                                          # 0
    (rotate(270), N_NODES),
    (rotate(180), N_NODES),
    (rotate(90), N_NODES),
    (pil(T.GaussianBlur(kernel_size=(3, 3), sigma=(6, 7))), N_NODES),
    (pil(T.GaussianBlur(kernel_size=(3, 3), sigma=(2, 9))), N_NODES),                            # 5
    (pil(T.ColorJitter(brightness=(0.5, 1.5), contrast=3, saturation=(0.3, 1.5), hue=(-0.1, 0.15))), N_NODES),
    (pil(T.ColorJitter(brightness=(0.5, 1.5), contrast=2, saturation=1.4, hue=(-0.1, 0.5))), N_NODES),
    (lambda im: add_gaussian_noise(im, 0, 0.25), N_NODES),
    (lambda im: add_gaussian_noise(im, 0, 0.45), N_NODES),
    (lambda im: add_gaussian_noise(im, 0, 0.55), N_NODES),                                       # 10
    (lambda im: add_salt_and_pepper_noise(im, 0.01), N_NODES),
    (lambda im: add_salt_and_pepper_noise(im, 0.015), N_NODES),
    (lambda im: add_salt_and_pepper_noise(im, 0.02), N_NODES),
    (lambda im: add_salt_and_pepper_noise(im, 0.025), N_NODES),
    (lambda im: add_random_occlusion(im, 10, 20), N_NODES),                                      # 15
    (lambda im: add_random_occlusion(im, 15, 25), N_NODES),
    (lambda im: add_random_occlusion(im, 20, 30), N_NODES),
    (lambda im: add_random_occlusion(im, 25, 35), N_NODES),
    (rotate_occlude(270, 10, 20), N_NODES),
    (rotate_occlude(180, 10, 20), N_NODES),                                                      # 20
    (rotate_occlude(90, 10, 20), N_NODES),
    (rotate_occlude(270, 15, 20), N_NODES),
    (rotate_occlude(180, 15, 20), N_NODES),
    (rotate_occlude(90, 15, 20), N_NODES),
    (rotate_occlude(270, 15, 15), N_NODES),                                                      # 25
    (rotate_occlude(180, 15, 15), N_NODES),
    (rotate_occlude(90, 15, 15), N_NODES),
    (rotate_occlude(270, 10, 10), N_NODES),
    (rotate_occlude(180, 10, 10), N_NODES),
    (rotate_occlude(90, 10, 10), N_NODES),                                                       # 30
    (identity, NODE_VARIATIONS[0]),
    (identity, NODE_VARIATIONS[2]),
    (identity, NODE_VARIATIONS[3]),
    (identity, NODE_VARIATIONS[4]),
    (identity, NODE_VARIATIONS[5]),                                                              # 35
    (pil(T.RandomPerspective(distortion_scale=0.5, p=1.0)), N_NODES),
    (pil(T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10)), N_NODES),
    (pil(T.ElasticTransform(alpha=50.0, sigma=5.0)), N_NODES),
    (rotate(45), NODE_VARIATIONS[2]),
    (pil(T.GaussianBlur(kernel_size=(5, 5), sigma=(4, 5))), NODE_VARIATIONS[3]),                 # 40
    (pil(T.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.8, 1.0))), N_NODES),
    (pil(T.RandomHorizontalFlip(p=1.0)), N_NODES),
    (pil(T.RandomVerticalFlip(p=1.0)), N_NODES),
    (pil(T.Compose([T.RandomRotation((30, 30)), T.GaussianBlur(kernel_size=(3, 3), sigma=(2, 3))])), NODE_VARIATIONS[4]),
    (pil(T.Compose([T.RandomPerspective(distortion_scale=0.3, p=1.0),                           # 45
                    T.ColorJitter(brightness=0.4, contrast=0.4)])), NODE_VARIATIONS[1]),
    (pil(T.RandomSolarize(threshold=128, p=1.0)), N_NODES),
    (pil(T.RandomPosterize(bits=3, p=1.0)), N_NODES),
    (motion_blur, NODE_VARIATIONS[0]),
    (exposure, NODE_VARIATIONS[2]),
    (contrast, NODE_VARIATIONS[5]),                                                              # 50
]
assert len(AUGMENTATIONS) == 51

# RandomPerspective gives different images in torchvision 0.15 (graph building) and 0.21 (training), so the
# augmented images of these entries are stored with their graphs instead of being rebuilt when loaded.
STORED_AUGMENTATIONS = (36, 45)

# Multiscale entries, with the targets of NODE_VARIATIONS: in a dataset with another base number of
# superpixels (build_dataset.py --nodes) they keep these targets, and the other entries take the base number.
FIXED_TARGET_AUGMENTATIONS = (31, 32, 33, 34, 35, 39, 40, 44, 45, 48, 49, 50)
assert all(n == N_NODES for k, (_, n) in enumerate(AUGMENTATIONS) if k not in FIXED_TARGET_AUGMENTATIONS)


def target_nodes(k, base=N_NODES):
    """DISF target of augmentation k in the dataset whose base number of superpixels is `base`."""
    return AUGMENTATIONS[k][1] if k in FIXED_TARGET_AUGMENTATIONS else base


@contextlib.contextmanager
def seeded(seed):
    """Fixed python/numpy/torch RNG seeds inside the block; the previous RNG states are restored after."""
    py_state, np_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        try:
            yield
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)


def augment(image, seed, k):
    """Augmentation k of an image with fixed seeds; returns (augmented uint8 image, DISF target nodes)."""
    transform, n_nodes = AUGMENTATIONS[k]
    with seeded(seed):
        out = np.ascontiguousarray(transform(image), dtype=np.uint8)
    return out, n_nodes
