#!/bin/bash
# Whole hyperparameter search on EuroSAT: builds the graphs of the search (25 to 150 superpixels,
# search_config.NODE_VALUES), searches on 10% of the training images (validation set only, test set not used) and
# confirms the 3 best configurations and the reference (configs/default.json, the configuration of the paper's runs)
# with 5 new seeds on 30% of the images. Resumable: run it again after an interruption and it continues where it
# stopped.
#   mkdir -p logs/search && nohup setsid bash run_search.sh >> logs/search/run.out 2>&1 < /dev/null &
# BUILD_PYTHON: environment with DISF (graph building); PYTHON: training environment.
set -eu
cd "$(dirname "$0")"
BUILD_PYTHON=${BUILD_PYTHON:-python}
PYTHON=${PYTHON:-python}
TRIALS=${TRIALS:-60}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1  # graph building: one thread per worker process
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-file:$(pwd)/mlruns} PYTHONUNBUFFERED=1
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
