"""Paths and constants of the RESISC45 VGG-feature pipeline."""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
# Images of the torchgeo RESISC45 splits exported by export_raw.py: {split}/{split}_{i}.jpg + .npz (one-hot label)
RAW_DIR = os.path.join(ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")  # one compressed superpixel map per graph (N_NODES)
IMAGES_DIR = os.path.join(ROOT, "data", "images")        # decoded split images (prepare_images.py)
WEIGHTS_DIR = os.path.join(ROOT, "weights")

NUM_CLASSES = 45
IMG_SIZE = 256
N_NODES = 100            # target number of DISF superpixels (as in the RESISC hierarchy graphs)
DISF_INIT_SEEDS = 5000   # DISF initial seeds
N_AUGMENTATIONS = 51     # graphs per training image (see augmentations.py)
SPLIT_SIZES = {"train": 18900, "val": 6300, "test": 6300}

# torchgeo's RESISC45 labels classes in alphabetical order of the class folders
CLASS_NAMES = ["airplane", "airport", "baseball_diamond", "basketball_court", "beach", "bridge", "chaparral",
               "church", "circular_farmland", "cloud", "commercial_area", "dense_residential", "desert", "forest",
               "freeway", "golf_course", "ground_track_field", "harbor", "industrial_area", "intersection", "island",
               "lake", "meadow", "medium_residential", "mobile_home_park", "mountain", "overpass", "palace",
               "parking_lot", "railway", "railway_station", "rectangular_farmland", "river", "roundabout", "runway",
               "sea_ice", "ship", "snowberg", "sparse_residential", "stadium", "storage_tank", "tennis_court",
               "terrace", "thermal_power_station", "wetland"]

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


def processed_dir(n_nodes=N_NODES):
    """Maps of the graphs whose base number of superpixels is n_nodes (see augmentations.target_nodes)."""
    return PROCESSED_DIR if n_nodes == N_NODES else f"{PROCESSED_DIR}_n{n_nodes}"


def graph_path(split, idx, n_nodes=N_NODES):
    return os.path.join(processed_dir(n_nodes), f"{split}_{idx}.npz")
