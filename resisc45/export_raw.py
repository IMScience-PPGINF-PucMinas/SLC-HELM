"""Exports the torchgeo RESISC45 splits (18900 train / 6300 val / 6300 test images, 256x256, 45 classes) to the files
prepare_images.py reads: data/raw/{split}/{split}_{i}.jpg (JPEG, quality 90) and data/raw/{split}/{split}_{i}.npz
(one-hot label), i following the order of the torchgeo split. torchgeo downloads the dataset to data/ first.

    python export_raw.py

Needs torchgeo (pip install torchgeo); run it once, in either environment.
"""
import os

import numpy as np
from PIL import Image
from torchgeo.datasets import RESISC45
from tqdm import tqdm

from config import NUM_CLASSES, RAW_DIR, ROOT, SPLIT_SIZES


def main():
    for split, n in SPLIT_SIZES.items():
        dataset = RESISC45(root=os.path.join(ROOT, "data"), split=split, download=True)
        assert len(dataset) == n, f"{split}: {len(dataset)} images, {n} expected"
        out = os.path.join(RAW_DIR, split)
        os.makedirs(out, exist_ok=True)
        for i in tqdm(range(n), desc=split):
            sample = dataset[i]
            image = sample["image"].numpy().astype(np.uint8).transpose(1, 2, 0)
            Image.fromarray(image).save(os.path.join(out, f"{split}_{i}.jpg"), quality=90)
            label = np.zeros(NUM_CLASSES, dtype=np.float32)
            label[int(sample["label"])] = 1
            np.savez_compressed(os.path.join(out, f"{split}_{i}.npz"), label=label)


if __name__ == "__main__":
    main()
