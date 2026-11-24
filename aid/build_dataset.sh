#!/bin/bash
# Builds the AID graphs: decoded images (prepare_images.py), then the superpixel maps of val, test and
# train (all 51 augmentations), then checks them in the training environment. Resumable: run it again after
# an interruption and it continues where it stopped.
#   bash build_dataset.sh                 # BUILD_PYTHON: environment with DISF (NumPy 1.x)
#                                         # TRAIN_PYTHON: environment used for training (check_dataset.py)
#   NODES=300 bash build_dataset.sh       # base of 300 superpixels, in data/processed_n300
set -e
cd "$(dirname "$0")"
BUILD_PYTHON=${BUILD_PYTHON:-python}
TRAIN_PYTHON=${TRAIN_PYTHON:-python}
WORKERS=${WORKERS:-12}
NODES=${NODES:-100}
DIR=$($TRAIN_PYTHON -c "from config import processed_dir; print(processed_dir($NODES))")
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1  # one thread per worker process
echo "start: $(date)"
$BUILD_PYTHON prepare_images.py
for split in val test train; do
    echo "$split: $(date)"
    $BUILD_PYTHON build_dataset.py --split $split --workers $WORKERS --nodes $NODES
done
echo "graphs built: $(date) | $(ls $DIR | grep -c '\.npz$') graphs | $(du -sh $DIR | cut -f1)"
$TRAIN_PYTHON check_dataset.py --nodes $NODES
echo "done: $(date)"
