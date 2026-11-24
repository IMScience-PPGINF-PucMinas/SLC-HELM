"""Paths and constants shared by the whole EuroSAT pipeline."""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(ROOT, "data", "raw")              # torchvision EuroSAT root (data/raw/eurosat/2750/<class>/*.jpg)
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")  # one .pt graph per file (N_NODES superpixels)
INDICES_DIR = os.path.join(ROOT, "indices")              # fixed 60/20/20 split (indices into torchvision's EuroSAT)
WEIGHTS_DIR = os.path.join(ROOT, "weights")

NUM_CLASSES = 10
IMG_SIZE = 64
N_NODES = 50             # target number of DISF superpixels
DISF_INIT_SEEDS = 1000   # DISF initial seeds
N_AUGMENTATIONS = 51     # graphs per training image (see augmentations.py)
SPLIT_SIZES = {"train": 16200, "val": 5400, "test": 5400}

CLASS_NAMES = ["Annual Crop", "Forest", "Herbaceous Vegetation", "Highway", "Industrial",
               "Pasture", "Permanent Crop", "Residential", "River", "Sea Lake"]

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


def processed_dir(n_nodes=N_NODES):
    """Graphs whose base number of superpixels is n_nodes (build_dataset.py --nodes)."""
    return PROCESSED_DIR if n_nodes == N_NODES else f"{PROCESSED_DIR}_n{n_nodes}"


def graph_path(split, idx, n_nodes=N_NODES):
    return os.path.join(processed_dir(n_nodes), f"{split}_{idx}.pt")
