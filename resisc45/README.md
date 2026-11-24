# SLC-HELM on NWPU-RESISC45

NWPU-RESISC45: 31,500 images of 256×256 pixels in 45 classes, with the torchgeo train/val/test splits (18,900 /
6,300 / 6,300 images). See the [main README](../README.md) for the method, the environments and the hyperparameters.

## Data and graphs

```bash
python export_raw.py          # torchgeo download + data/raw/{split}/{split}_{i}.jpg (JPEG q90) and .npz (label); needs torchgeo
NODES=300 BUILD_PYTHON=/path/to/build-env/python TRAIN_PYTHON=python bash build_dataset.sh
```

`build_dataset.sh` decodes the images once (`prepare_images.py` → `data/images/{split}.npy`, 5.8 GB), builds the
superpixel maps of val, test and the 51 augmentations of each training image (`data/processed_n300`, ~20 GB; without
`NODES`, 100 superpixels in `data/processed`) and checks them in the training environment (`check_dataset.py`, must
end with `OK`). Each map stores the superpixel labels, the edges, the label and the MD5 of the augmented image; the
image itself is rebuilt from `data/images` when the graph is loaded. DISF uses 5,000 initial seeds; the 12 multiscale
augmentations keep their own targets (100 to 600 superpixels).

## Training and test

```bash
CONFIG=trial_53 bash run_experiment.sh netvlad lse        # paper: 5 seeds, train + test, in sequence
CONFIG=trial_53 GPUS="0 1 2 3 4" bash run_parallel.sh netvlad lse   # one seed per GPU, in the background
python train.py --seed 1 --config configs/trial_53.json  # a single run; python test.py with the same arguments
```

`configs/trial_53.json` is the configuration selected by the search (300 superpixels, *K* = 16, 4 heads, percentile
35 over the batch, lr 4.1e-4, dropout 0.1, label smoothing 0.2, weight decay 6.3e-7); `trial_19`, `trial_49` and
`trial_55` are the other confirmed configurations. Runs are tagged with the configuration name, e.g.
`weights/SCH_HELM2_netvlad_lse_n300_trial_53_seed=1.pth` and `logs/test_netvlad_lse_trial_53_seed1.log`.
`train.py --backbone resnet50` replaces the VGG16 pixel features by ResNet50 `conv1` features (not used in the paper).

## Hyperparameter search

`bash run_search.sh` (resumable; results in `logs/search/`): superpixel counts 50, 100, 150, 200 and 300, reference =
`train.py` defaults (100 superpixels). The results are in `configs/search_trials.csv` and `configs/search_confirm.csv`
and summarized in the main README.

## Other protocols (not in the paper)

`split_protocol.py` defines the usual 20%/80% protocol over the same graphs (`splits/tr20.npz`: 10% of the labelled
images held out for validation; `tr20_noval.npz`: none). `PROTOCOL=tr20 CONFIG=trial_53 bash run_parallel.sh netvlad
lse` trains on it, and `PROTOCOL=tr20 bash run_search.sh` searches inside its labelled images only.

## MobileNetV3-Small baseline

`bash run_cnn_baseline.sh` or `python cnn_baseline.py --seed 1`: same images and augmentations, lr, weight decay and
label smoothing of `configs/trial_53.json`.
