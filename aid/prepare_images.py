"""Decodes the AID images of each split once into uint8 arrays: data/images/{split}.npy [N, 600, 600, 3] and
data/images/{split}_labels.npy [N], in the order of splits/aid-{split}.txt.

The graphs are built and later loaded from these arrays, so every environment works on the same
pixels (JPEG decoding differs between Pillow/libjpeg versions).

    python prepare_images.py
"""
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

from config import CLASS_NAMES, IMAGES_DIR, IMG_SIZE, RAW_DIR, SPLIT_FILES, SPLIT_SIZES


def image_files():
    """{file name: (path, class index)} of the images in RAW_DIR, the class given by the folder."""
    classes = {name.lower(): c for c, name in enumerate(CLASS_NAMES)}
    files = {}
    for folder, _, names in os.walk(RAW_DIR):
        for name in names:
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".tif")):
                assert name not in files, f"{name} appears twice in {RAW_DIR}"
                files[name] = (os.path.join(folder, name), classes[os.path.basename(folder).lower()])
    return files


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    files = image_files()
    for split, n in SPLIT_SIZES.items():
        out = os.path.join(IMAGES_DIR, f"{split}.npy")
        if os.path.exists(out):
            print(f"{out} exists, skipped")
            continue
        with open(SPLIT_FILES.format(split=split)) as f:
            names = [line.strip() for line in f if line.strip()]
        assert len(names) == n, f"{split}: {len(names)} images in the split file, {n} expected"
        images = np.lib.format.open_memmap(out + ".tmp.npy", mode="w+", dtype=np.uint8, shape=(n, IMG_SIZE, IMG_SIZE, 3))
        labels = np.empty(n, dtype=np.int64)
        for i, name in enumerate(tqdm(names, desc=split)):
            path, labels[i] = files[name]
            assert CLASS_NAMES[labels[i]].lower() == name.rsplit("_", 1)[0], f"{path}: file name and folder disagree"
            images[i] = np.asarray(Image.open(path).convert("RGB"))
        images.flush()
        del images
        np.save(os.path.join(IMAGES_DIR, f"{split}_labels.npy"), labels)
        os.replace(out + ".tmp.npy", out)


if __name__ == "__main__":
    main()
