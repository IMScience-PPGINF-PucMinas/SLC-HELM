#!/bin/bash
# CNN baseline (cnn_baseline.py, MobileNetV3-Small from scratch) on RESISC45: the seeds run from a queue in the
# background, so the runs survive the end of the session, PARALLEL at a time on one GPU (default 1: at 256x256
# the rebuilt augmentations keep all CPU cores busy, ~1600 images/s on a 6-core Ryzen, so parallel seeds would only
# split them). Each seed writes logs/cnn_baseline/seed<N>.log, with the test results at the end. Running the script
# again skips finished seeds and resumes interrupted ones.
#
#   bash run_cnn_baseline.sh [seeds...]      # default seeds: 1 16 20 58 123
#   PARALLEL=5 WORKERS=8 GPU=1 bash run_cnn_baseline.sh    # e.g. on a machine with more cores
#   GPUS="0 3 4 5 6" bash run_cnn_baseline.sh             # one seed per GPU, all at once (as run_parallel.sh)
#
# With GPUS, seed i runs on the i-th GPU of the list (at least as many GPUs as seeds); keep
# seeds x (WORKERS + 4) below the number of CPU cores, since rebuilding the augmentations is CPU-bound.
set -u
cd "$(dirname "$0")"
SEEDS=${*:-1 16 20 58 123}
export PYTHON=${PYTHON:-python}
PARALLEL=${PARALLEL:-1}
export WORKERS=${WORKERS:-11}  # data-loading workers per seed
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-file:$(pwd)/mlruns} PYTHONUNBUFFERED=1
export L=logs/cnn_baseline
mkdir -p $L

# Done once before starting the seeds, so they do not race for it: the MLflow experiment.
$PYTHON -c "import cnn_baseline as cb, mlflow; mlflow.set_experiment(cb.parse_args().experiment)" \
    > /dev/null 2>&1 || { echo "setup failed"; exit 1; }

run_seed() {
    [ -f $L/seed$1.done ] && return 0
    # flock: the same seed is never run twice at the same time
    flock -n $L/seed$1.lock bash -c "$PYTHON cnn_baseline.py --seed $1 --workers $WORKERS && touch $L/seed$1.done" \
        >> $L/seed$1.log 2>&1
    echo "seed $1: $(grep -E '^Accuracy' $L/seed$1.log | tail -1 || echo 'stopped, see the log') ($(date))"
}
export -f run_seed
export SEEDS PARALLEL
if [ -n "${GPUS:-}" ]; then  # one seed per GPU, each in its own background process
    read -r -a SEED_LIST <<< "$SEEDS"
    read -r -a GPU_LIST <<< "$GPUS"
    [ ${#GPU_LIST[@]} -ge ${#SEED_LIST[@]} ] || { echo "${#SEED_LIST[@]} seeds but only ${#GPU_LIST[@]} GPUs ($GPUS)"; exit 1; }
    for i in "${!SEED_LIST[@]}"; do
        CUDA_VISIBLE_DEVICES=${GPU_LIST[$i]} nohup setsid bash -c "run_seed ${SEED_LIST[$i]}" >> $L/queue.out 2>&1 < /dev/null &
        echo "seed ${SEED_LIST[$i]} -> GPU ${GPU_LIST[$i]} (pid $!, log $L/seed${SEED_LIST[$i]}.log)"
    done
    echo "follow with: tail -f $L/seed*.log $L/queue.out   (or nvidia-smi)"
    exit 0
fi
nohup setsid bash -c 'echo $SEEDS | tr " " "\n" | xargs -P $PARALLEL -I{} bash -c "run_seed {}"' \
    >> $L/queue.out 2>&1 < /dev/null &
echo "queue of seeds $SEEDS started, $PARALLEL at a time on GPU $CUDA_VISIBLE_DEVICES (pid $!)"
echo "follow with: tail -f $L/seed*.log $L/queue.out"
