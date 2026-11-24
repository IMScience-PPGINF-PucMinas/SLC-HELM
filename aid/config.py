"""Paths and constants of the AID VGG-feature pipeline."""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
# AID.zip of the isaaccorley/aid mirror on Hugging Face, extracted (one folder per class: data/raw/AID/<Class>/*.jpg)
RAW_DIR = os.path.join(ROOT, "data", "raw", "AID")
# Split files of the same mirror: AID has no official split, these are the stratified 60/20/20 split of
# torchgeo-bench (seed 0), one file name per line
SPLIT_FILES = os.path.join(ROOT, "splits", "aid-{split}.txt")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")  # one compressed superpixel map per graph (N_NODES)
IMAGES_DIR = os.path.join(ROOT, "data", "images")        # decoded split images (prepare_images.py)
WEIGHTS_DIR = os.path.join(ROOT, "weights")

NUM_CLASSES = 30
IMG_SIZE = 600           # original size of the AID images (not resized)
N_NODES = 100            # target number of DISF superpixels (as in RESISC45)
DISF_INIT_SEEDS = 5000   # DISF initial seeds (as in RESISC45)
N_AUGMENTATIONS = 51     # graphs per training image (see augmentations.py)
SPLIT_SIZES = {"train": 6000, "val": 2000, "test": 2000}

# Classes in alphabetical order of the class folders (as torchgeo does for RESISC45); the file names start with the
# class name in lower case (bareland_1.jpg)
CLASS_NAMES = ["Airport", "BareLand", "BaseballField", "Beach", "Bridge", "Center", "Church", "Commercial",
               "DenseResidential", "Desert", "Farmland", "Forest", "Industrial", "Meadow", "MediumResidential",
               "Mountain", "Park", "Parking", "Playground", "Pond", "Port", "RailwayStation", "Resort", "River",
               "School", "SparseResidential", "Square", "Stadium", "StorageTanks", "Viaduct"]

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


def processed_dir(n_nodes=N_NODES):
    """Maps of the graphs whose base number of superpixels is n_nodes (see augmentations.target_nodes)."""
    return PROCESSED_DIR if n_nodes == N_NODES else f"{PROCESSED_DIR}_n{n_nodes}"


def graph_path(split, idx, n_nodes=N_NODES):
    return os.path.join(processed_dir(n_nodes), f"{split}_{idx}.npz")
