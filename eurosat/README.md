# SLC-HELM on EuroSAT

EuroSAT (RGB version): 27,000 Sentinel-2 images of 64×64 pixels in 10 classes. The fixed 60/20/20 split (16,200 /
5,400 / 5,400 images) is in `indices/{train,val,test}_indices.npy`, as indices into torchvision's `EuroSAT` dataset
(classes and files in alphabetical order). See the [main README](../README.md) for the method, the environments and
the hyperparameters.

## Data and graphs

```bash
python -c "from torchvision.datasets import EuroSAT; EuroSAT('data/raw', download=True)"   # data/raw/eurosat/2750/
BUILD_PYTHON=/path/to/build-env/python bash build_dataset.sh       # 50 superpixels: data/processed/
NODES=100 BUILD_PYTHON=/path/to/build-env/python bash build_dataset.sh   # another count: data/processed_n100/
```

Each graph is a `.pt` file with the image, the pixel-to-superpixel map (int16), the RAG edges and the label (~28 KB;
~23 GB for the 826,200 training graphs). Train graph *k* of image *i* is `train_{i*51 + k}.pt`, built with the
augmentation *k* of `augmentations.py` seeded with `i*51 + k`. DISF uses 1,000 initial seeds.

## Training and test

```bash
export MLFLOW_TRACKING_URI=file:$PWD/mlruns
python train.py --seed 1                  # configuration of the paper (train.py defaults)
python test.py --seed 1                   # same arguments as train.py
bash train_runs.sh && bash test_runs.sh   # seeds 1 123 16 20 58
python train.py --seed 1 --config configs/trial_<n>.json   # a configuration exported by search.py
```

The defaults of `train.py` are the configuration of the paper (`configs/default.json`): 50 superpixels, *K* = 16,
*d* = 64, 2 heads, percentile 50 over the batch, lr 1e-3, weight decay 3e-6, dropout 0.1, label smoothing 0.1, batch
64, readout `lse`, early stopping patience 30. Checkpoints go to `weights/SCH_HELM2_netvlad_lse_seed=<s>.pth`.

## Hyperparameter search

```bash
mkdir -p logs/search
BUILD_PYTHON=/path/to/build-env/python PYTHON=python nohup setsid bash run_search.sh >> logs/search/run.out 2>&1 < /dev/null &
```

Same protocol as RESISC45 and AID (see the main README): superpixel counts 25, 35, 50, 75, 100 and 150
(`search_config.NODE_VALUES`), 60 trials on 10% of the training images (1,621 images × 8 augmentations) validated on
the whole validation set, then the 3 best trials and the reference (`configs/default.json`) with seeds 2–6 on 30% of
the training images. The search graphs go to `data/processed_n{n}`, where `NODES=n bash build_dataset.sh` later
completes the full dataset of the chosen count. Step by step:

```bash
python search_build.py --fraction 0.1      # graph-building environment
python search.py check --fraction 0.1
python search.py run --trials 60
python search.py top --k 3                 # prints the superpixel counts to build for the confirmation
python search_build.py --fraction 0.3 --nodes <counts>
python search.py confirm
python search.py export trial_<n>          # configs/trial_<n>.json
```

## MobileNetV3-Small baseline

`bash run_cnn_baseline.sh` (seeds 1 16 20 58 123) or `python cnn_baseline.py --seed 1`. It decodes the images once into
`data/eurosat_images.npy` and rebuilds the 51 augmentations with the seeds of `build_dataset.py`.
