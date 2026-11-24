#!/bin/bash
# CNN baseline (cnn_baseline.py, MobileNetV3-Small from scratch): the seeds share one GPU, at most PARALLEL at a time
# (the network is small, so a single process leaves the GPU mostly idle: 3 processes give ~1.8x the throughput of
# one, 5 hardly more), from a queue that runs in the background so the runs survive the end of the session. Each seed
# writes logs/cnn_baseline/seed<N>.log, with the test results at the end. Running the script again skips finished
# seeds and resumes interrupted ones.
#
#   bash run_cnn_baseline.sh [seeds...]      # default seeds: 1 16 20 58 123
#   PARALLEL=5 WORKERS=2 GPU=1 bash run_cnn_baseline.sh
set -u
cd "$(dirname "$0")"
SEEDS=${*:-1 16 20 58 123}
export PYTHON=${PYTHON:-python}
PARALLEL=${PARALLEL:-3}
export WORKERS=${WORKERS:-3}  # data-loading workers per seed
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-file:$(pwd)/mlruns} PYTHONUNBUFFERED=1
export L=logs/cnn_baseline
mkdir -p $L

# Done once before starting the seeds, so they do not race for them: decoded images and MLflow experiment.
$PYTHON -c "import cnn_baseline as cb, mlflow; cb.decode_images(); mlflow.set_experiment(cb.parse_args().experiment)" \
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
nohup setsid bash -c 'echo $SEEDS | tr " " "\n" | xargs -P $PARALLEL -I{} bash -c "run_seed {}"' \
    >> $L/queue.out 2>&1 < /dev/null &
echo "queue of seeds $SEEDS started, $PARALLEL at a time on GPU $CUDA_VISIBLE_DEVICES (pid $!)"
echo "follow with: tail -f $L/seed*.log $L/queue.out"
