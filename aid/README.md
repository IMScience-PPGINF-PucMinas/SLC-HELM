# SLC-HELM on AID

AID (Aerial Image Dataset, Xia et al., IEEE TGRS 2017): 10,000 Google Earth images of 600×600 pixels in 30 classes,
used at their original size. AID has no official split; this code uses the stratified 60/20/20 split of
torchgeo-bench (seed 0) published with the Hugging Face mirror
[isaaccorley/aid](https://huggingface.co/datasets/isaaccorley/aid): 6,000 / 2,000 / 2,000 images, file lists in
`splits/aid-{train,val,test}.txt`. See the [main README](../README.md) for the method, the environments and the
hyperparameters.

## Data and graphs

```bash
mkdir -p data/raw && cd data/raw
curl -L -O https://huggingface.co/datasets/isaaccorley/aid/resolve/main/AID.zip
echo "79345ef1766e85f7b92cf556c809c4f32ff24228ae71aadf96780f74f48c26a4  AID.zip" | sha256sum -c
unzip -q AID.zip && cd ../..  # data/raw/AID/<Class>/<class>_<n>.jpg
NODES=300 BUILD_PYTHON=/path/to/build-env/python TRAIN_PYTHON=python bash build_dataset.sh
```

`build_dataset.sh` decodes the images once (`prepare_images.py` → `data/images/{split}.npy`, 10.8 GB, in the order of
the split files), builds the superpixel maps of val, test and the 51 augmentations of each training image
(`data/processed_n300`, 310,000 maps, ~18 GB, ~21 h with 12 workers) and checks them (`check_dataset.py`, must end with
`OK`). DISF takes ~1.6 s per 600×600 image and leaks memory, so each worker process is replaced after ~100 graphs.

## Training and test

```bash
CONFIG=aid_trial_33 bash run_experiment.sh netvlad lse        # paper: 5 seeds, train + test, in sequence
CONFIG=aid_trial_33 GPUS="0 1 2 3 4" bash run_parallel.sh netvlad lse
python train.py --seed 1 --config configs/aid_trial_33.json   # a single run; python test.py with the same arguments
```

`configs/aid_trial_33.json` is the configuration selected by the search (300 superpixels, *K* = 8, 4 heads, percentile
75 per graph, lr 5.5e-4, no dropout, label smoothing 0.2, weight decay 6.9e-7); `configs/trial_53.json` is the RESISC45
configuration, the reference of the AID search. The batch size is 32: with 600×600 images a batch of 64 needs ~18 GB of
GPU memory (`--batch-size`, or `"batch_size"` in a configuration file).

## Hyperparameter search

`bash run_search.sh` (resumable; results in `logs/search/`): superpixel counts 300, 450, 650, 950 and 1,400, embedding
size 32 or 64, reference = RESISC45's `trial_53`. A step of 32 graphs with 1,400 superpixels takes up to ~20.5 GB, so
`search.py` caps the CUDA allocator at 92% of the GPU memory and prunes the trials that run out of memory without
counting them. The results are in `configs/search_trials.csv` and `configs/search_confirm.csv` and summarized in the
main README.

## Other protocols (not in the paper)

`split_protocol.py` defines the usual 50%/50% protocol over the same graphs (`splits/tr50.npz`: 10% of the labelled
images held out for validation; `tr50_noval.npz`: none). `PROTOCOL=tr50 CONFIG=aid_trial_33 bash run_parallel.sh
netvlad lse` trains on it, and `PROTOCOL=tr50 bash run_search.sh` searches inside its labelled images only (superpixel
counts 150 to 650).

## MobileNetV3-Small baseline

`bash run_cnn_baseline.sh` or `python cnn_baseline.py --seed 1`: same images and augmentations, lr, weight decay and
label smoothing of `configs/aid_trial_33.json`, 600×600 input. The images of the stored augmentations are read from
`data/processed_n300`.
