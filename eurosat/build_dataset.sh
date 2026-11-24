#!/bin/bash
# Builds the EuroSAT graphs: val, test, then train (all 51 augmentations). Resumable: run it again after an
# interruption and it continues where it stopped.
#   bash build_dataset.sh                 # BUILD_PYTHON: environment with DISF (NumPy 1.x)
#   NODES=100 bash build_dataset.sh       # base of 100 superpixels, in data/processed_n100
set -e
cd "$(dirname "$0")"
BUILD_PYTHON=${BUILD_PYTHON:-python}
WORKERS=${WORKERS:-$(nproc)}
NODES=${NODES:-50}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1  # one thread per worker process
for split in val test train; do
    echo "$split: $(date)"
    $BUILD_PYTHON build_dataset.py --split $split --workers "$WORKERS" --nodes "$NODES"
done
echo "done: $(date)"
