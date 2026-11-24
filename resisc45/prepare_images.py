"""Decodes the split JPEGs once into uint8 arrays: data/images/{split}.npy [N, 256, 256, 3] and
data/images/{split}_labels.npy [N].

The graphs are built and later loaded from these arrays, so every environment works on the same
pixels (JPEG decoding differs between Pillow/libjpeg versions).

    python prepare_images.py
"""
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

from config import IMAGES_DIR, IMG_SIZE, RAW_DIR, SPLIT_SIZES


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    for split, n in SPLIT_SIZES.items():
        out = os.path.join(IMAGES_DIR, f"{split}.npy")
        if os.path.exists(out):
            print(f"{out} exists, skipped")
            continue
        images = np.lib.format.open_memmap(out + ".tmp.npy", mode="w+", dtype=np.uint8, shape=(n, IMG_SIZE, IMG_SIZE, 3))
        labels = np.empty(n, dtype=np.int64)
        for i in tqdm(range(n), desc=split):
            images[i] = np.asarray(Image.open(os.path.join(RAW_DIR, split, f"{split}_{i}.jpg")).convert("RGB"))
            labels[i] = np.load(os.path.join(RAW_DIR, split, f"{split}_{i}.npz"))["label"].argmax()
        images.flush()
        del images
        np.save(os.path.join(IMAGES_DIR, f"{split}_labels.npy"), labels)
        os.replace(out + ".tmp.npy", out)


if __name__ == "__main__":
    main()
