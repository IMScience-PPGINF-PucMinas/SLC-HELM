#!/bin/bash
# Whole hyperparameter search: builds the superpixel maps (300 to 1400 superpixels, search_config.NODE_VALUES),
# searches on 10% of the training images (validation set only, test set not used) and confirms the 3 best
# configurations and the reference (RESISC45's trial_53) with 5 new seeds on 30% of the images. Resumable: run it
# again after an interruption and it continues where it stopped.
#   nohup setsid bash run_search.sh >> logs/search/run.out 2>&1 < /dev/null &
# With PROTOCOL (a protocol of split_protocol.py, e.g. tr50) the search is done inside its labelled images, with the
# superpixel counts of search_config.PROTOCOL_NODE_VALUES (see search.py), in logs/search_$PROTOCOL/; the maps of all
# its folds are built at the start:
#   mkdir -p logs/search_tr50 && PROTOCOL=tr50 nohup setsid bash run_search.sh >> logs/search_tr50/run.out 2>&1 < /dev/null &
set -eu
cd "$(dirname "$0")"
BUILD_PYTHON=${BUILD_PYTHON:-python}
PYTHON=${PYTHON:-python}
TRIALS=${TRIALS:-60}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1  # graph building: one thread per worker process
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-file:$(pwd)/mlruns} PYTHONUNBUFFERED=1
# Less fragmentation: with 1400 superpixels ~1 GB reserved above the allocated memory instead of ~2.7 GB
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
PROTOCOL=${PROTOCOL:-}
if [ -n "$PROTOCOL" ]; then
    [ -f "splits/$PROTOCOL.npz" ] || { echo "splits/$PROTOCOL.npz not found (split_protocol.py)"; exit 1; }
    P="--protocol $PROTOCOL"
    mkdir -p logs/search_$PROTOCOL
    echo "search graphs ($PROTOCOL): $(date)"
    $BUILD_PYTHON search_build.py $P
    unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
    $PYTHON search.py $P check
    echo "search: $(date)"
    $PYTHON search.py $P run --trials "$TRIALS"
    echo "confirmation, superpixels $($PYTHON search.py $P top --k 3 | tail -1): $(date)"
    $PYTHON search.py $P confirm
    echo "done: $(date)"
    exit 0
fi
mkdir -p logs/search
echo "search graphs: $(date)"
$BUILD_PYTHON search_build.py --fraction 0.1
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
$PYTHON search.py check --fraction 0.1
echo "search: $(date)"
$PYTHON search.py run --trials "$TRIALS"
NODES=$($PYTHON search.py top --k 3 | tail -1)
echo "confirmation, superpixels $NODES: $(date)"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $BUILD_PYTHON search_build.py --fraction 0.3 --nodes $NODES
$PYTHON search.py check --fraction 0.3 --nodes $NODES
$PYTHON search.py confirm
echo "done: $(date)"
