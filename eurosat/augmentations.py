"""The 51 training augmentations. Entry k produces training graph k of each image.

Each entry is (transform, n_nodes): transform maps an HxWx3 uint8 array to another one
of the same size, n_nodes is the DISF target (None = the default N_NODES).
"""
import random

import cv2
import numpy as np
import torchvision.transforms as T
from PIL import Image

from config import IMG_SIZE


def add_random_occlusion(image, min_size, max_size):
    h, w = image.shape[:2]
    rw, rh = random.randint(min_size, max_size), random.randint(min_size, max_size)
    x0, y0 = random.randint(0, w - rw), random.randint(0, h - rh)
    out = image.copy()
    out[y0:y0 + rh, x0:x0 + rw] = 0
    return out


def add_salt_and_pepper_noise(image, noise_ratio):
    out = image.copy()
    h, w = out.shape[:2]
    n = int(h * w * noise_ratio)
    rows, cols = np.random.randint(0, h, n), np.random.randint(0, w, n)
    salt = np.random.rand(n) < 0.5
    out[rows[~salt], cols[~salt]] = 0
    out[rows[salt], cols[salt]] = 255
    return out


def add_gaussian_noise(image, mean, std):
    noise = np.random.normal(mean, std, image.shape)
    return np.clip(image + noise, 0, 255).astype(np.uint8)


def adjust_gamma(image):
    gamma = np.random.uniform(0.5, 1.5)
    return np.clip(np.power(image / 255.0, gamma) * 255.0, 0, 255).astype(np.uint8)


def adjust_contrast(image):
    factor = np.random.uniform(0.5, 2.0)
    mean = image.mean(axis=(0, 1), keepdims=True)
    return np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)


def motion_blur(image, kernel_size=15):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[(kernel_size - 1) // 2, :] = 1.0 / kernel_size
    return cv2.filter2D(image, -1, kernel)


def pil(transform):
    """Wrap a torchvision PIL transform so it works on numpy arrays."""
    return lambda image: np.array(transform(Image.fromarray(image)))


def rotate(angle):
    return pil(T.RandomRotation((angle, angle)))


def rotate_occlude(angle, min_size, max_size):
    rot = rotate(angle)
    return lambda image: add_random_occlusion(rot(image), min_size, max_size)


def identity(image):
    return image


NODE_VARIATIONS = [30, 60, 70, 90, 110, 130]

AUGMENTATIONS = [
    (identity, None),                                                            # 0
    (rotate(270), None),
    (rotate(180), None),
    (rotate(90), None),
    (pil(T.GaussianBlur(kernel_size=(3, 3), sigma=(6, 7))), None),
    (pil(T.GaussianBlur(kernel_size=(3, 3), sigma=(2, 9))), None),               # 5
    (pil(T.ColorJitter(brightness=(0.5, 1.5), contrast=3, saturation=(0.3, 1.5), hue=(-0.1, 0.15))), None),
    (pil(T.ColorJitter(brightness=(0.5, 1.5), contrast=2, saturation=1.4, hue=(-0.1, 0.5))), None),
    (lambda im: add_gaussian_noise(im, 0, 25), None),
    (lambda im: add_gaussian_noise(im, 0, 45), None),
    (lambda im: add_gaussian_noise(im, 0, 55), None),                           # 10
    (lambda im: add_salt_and_pepper_noise(im, 0.01), None),
    (lambda im: add_salt_and_pepper_noise(im, 0.015), None),
    (lambda im: add_salt_and_pepper_noise(im, 0.02), None),
    (lambda im: add_salt_and_pepper_noise(im, 0.025), None),
    (lambda im: add_random_occlusion(im, 10, 20), None),                          # 15
    (lambda im: add_random_occlusion(im, 15, 25), None),
    (lambda im: add_random_occlusion(im, 20, 30), None),
    (lambda im: add_random_occlusion(im, 25, 35), None),
    (rotate_occlude(270, 10, 20), None),
    (rotate_occlude(180, 10, 20), None),                                          # 20
    (rotate_occlude(90, 10, 20), None),
    (rotate_occlude(270, 15, 20), None),
    (rotate_occlude(180, 15, 20), None),
    (rotate_occlude(90, 15, 20), None),
    (rotate_occlude(270, 15, 15), None),                                          # 25
    (rotate_occlude(180, 15, 15), None),
    (rotate_occlude(90, 15, 15), None),
    (rotate_occlude(270, 10, 10), None),
    (rotate_occlude(180, 10, 10), None),
    (rotate_occlude(90, 10, 10), None),                                           # 30
    (identity, NODE_VARIATIONS[0]),
    (identity, NODE_VARIATIONS[2]),
    (identity, NODE_VARIATIONS[3]),
    (identity, NODE_VARIATIONS[4]),
    (identity, NODE_VARIATIONS[5]),                                               # 35
    (pil(T.RandomPerspective(distortion_scale=0.5, p=1.0)), None),
    (pil(T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10)), None),
    (pil(T.ElasticTransform(alpha=50.0, sigma=5.0)), None),
    (rotate(45), NODE_VARIATIONS[2]),
    (pil(T.GaussianBlur(kernel_size=(5, 5), sigma=(4, 5))), NODE_VARIATIONS[3]),  # 40
    # Output kept at IMG_SIZE (was 50x50) so images of a batch can be stacked for the VGG.
    (pil(T.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.8, 1.0))), None),
    (pil(T.RandomHorizontalFlip(p=1.0)), None),
    (pil(T.RandomVerticalFlip(p=1.0)), None),
    (pil(T.Compose([T.RandomRotation((30, 30)),
                    T.GaussianBlur(kernel_size=(3, 3), sigma=(2, 3))])), NODE_VARIATIONS[4]),
    (pil(T.Compose([T.RandomPerspective(distortion_scale=0.3, p=1.0),
                    T.ColorJitter(brightness=0.4, contrast=0.4)])), NODE_VARIATIONS[1]),  # 45
    (pil(T.RandomSolarize(threshold=128, p=1.0)), None),
    (pil(T.RandomPosterize(bits=3, p=1.0)), None),
    (motion_blur, NODE_VARIATIONS[0]),
    (adjust_gamma, NODE_VARIATIONS[2]),
    (adjust_contrast, NODE_VARIATIONS[5]),                                        # 50
]
