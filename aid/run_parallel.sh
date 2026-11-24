#!/bin/bash
# Starts one seed per GPU at once (run_experiment.sh each), in the background so the runs survive
# the end of the SSH session. Each seed writes logs/run_<aggregation>_<readout>_seed<N>.out.
#
#   bash run_parallel.sh <netvlad|mean> <att|lse> [seeds...]        # default seeds: 1 16 20 58 123
#   GPUS="2 3 4 5 6" bash run_parallel.sh netvlad lse                # choose the GPUs
#   THRESHOLD=graph bash run_parallel.sh netvlad lse                 # per-graph pooling threshold
#   CONFIG=trial_53 bash run_parallel.sh netvlad lse                 # configuration of configs/trial_53.json
#   PROTOCOL=tr50 CONFIG=aid_trial_33 bash run_parallel.sh netvlad lse  # 50%/50% protocol (splits/tr50.npz)
#
# GPUs: $GPUS if set, else the ones in $CUDA_VISIBLE_DEVICES (e.g. given by the job scheduler),
# else 0, 1, 2, ... Running the script again skips finished seeds and resumes interrupted ones.
# Environment variables of run_experiment.sh (CONFIG, THRESHOLD, PROTOCOL, PYTHON, WORKERS, MLFLOW_TRACKING_URI) are passed on.
set -u
cd "$(dirname "$0")"
USAGE="usage: bash run_parallel.sh <netvlad|mean> <att|lse> [seeds...]"
AGG=${1:?$USAGE}
READOUT=${2:?$USAGE}
case $AGG in netvlad | mean) ;; *) echo "aggregation must be netvlad or mean"; exit 1 ;; esac
case $READOUT in att | lse) ;; *) echo "readout must be att or lse"; exit 1 ;; esac
shift 2
read -r -a SEEDS <<< "${*:-1 16 20 58 123}"
read -r -a GPU_LIST <<< "${GPUS:-$(echo "${CUDA_VISIBLE_DEVICES:-}" | tr ',' ' ')}"
[ ${#GPU_LIST[@]} -gt 0 ] || read -r -a GPU_LIST <<< "$(seq -s ' ' 0 $((${#SEEDS[@]} - 1)))"
[ ${#GPU_LIST[@]} -ge ${#SEEDS[@]} ] || { echo "${#SEEDS[@]} seeds but only ${#GPU_LIST[@]} GPUs (${GPU_LIST[*]})"; exit 1; }
export THRESHOLD=${THRESHOLD:-}
case $THRESHOLD in "" | batch) PG="" ;; graph) PG="_pg" ;; *) echo "THRESHOLD must be batch or graph"; exit 1 ;; esac
export CONFIG=${CONFIG:-}
CONFIG_ARGS=""
export PROTOCOL=${PROTOCOL:-}
[ -z "$PROTOCOL" ] || [ -f "splits/$PROTOCOL.npz" ] || { echo "splits/$PROTOCOL.npz not found (split_protocol.py)"; exit 1; }
PYTHON=${PYTHON:-python}
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-file:$(pwd)/mlruns}
mkdir -p logs
if [ -n "$CONFIG" ]; then
    [ -f "configs/$CONFIG.json" ] || { echo "configs/$CONFIG.json not found"; exit 1; }
    CONFIG_ARGS="--config configs/$CONFIG.json"
fi
# number of superpixels of the graphs (100, or the one of the configuration)
NODES=$($PYTHON -c "import sys, train; sys.argv = ['train.py'] + sys.argv[1:]; print(train.parse_args().nodes)" $CONFIG_ARGS) \
    || { echo "could not read the configuration"; exit 1; }
[ "$NODES" = 100 ] && CHECK=check_dataset || CHECK=check_dataset_n$NODES
CHECK=$CHECK${PROTOCOL:+_$PROTOCOL}  # with a protocol, only its graphs are checked (e.g. a copy with only them)

# Done once before starting the seeds, so they do not race for them: the dataset check, the VGG16
# weights download and the MLflow experiment creation.
ENV_ID="$(hostname):$($PYTHON -c 'import sys; print(sys.prefix)')"  # the check is redone on another machine or environment
if [ "$(cat logs/$CHECK.ok 2>/dev/null)" != "$ENV_ID" ]; then
    echo "checking the dataset (logs/$CHECK.log)..."
    $PYTHON check_dataset.py --nodes "$NODES" ${PROTOCOL:+--protocol $PROTOCOL} > logs/$CHECK.log 2>&1 \
        || { echo "dataset check failed, see logs/$CHECK.log"; exit 1; }
    echo "$ENV_ID" > logs/$CHECK.ok
fi
$PYTHON - $CONFIG_ARGS ${PROTOCOL:+--protocol $PROTOCOL} <<'PY' || { echo "setup failed (VGG16 weights or MLflow)"; exit 1; }
import sys

import mlflow
from torchvision.models import VGG16_Weights, vgg16

import train

vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
sys.argv = ["train.py"] + sys.argv[1:]
mlflow.set_experiment(train.parse_args().experiment)  # train.py's default experiment (of the configuration)
PY

for i in "${!SEEDS[@]}"; do
    s=${SEEDS[$i]}
    g=${GPU_LIST[$i]}
    out=logs/run_${AGG}_${READOUT}${PG}${CONFIG:+_$CONFIG}${PROTOCOL:+_$PROTOCOL}_seed$s.out
    CUDA_VISIBLE_DEVICES=$g nohup bash run_experiment.sh "$AGG" "$READOUT" "$s" >> "$out" 2>&1 < /dev/null &
    echo "seed $s -> GPU $g (pid $!, log $out)"
    sleep 5
done
echo "follow with: tail -f logs/run_${AGG}_${READOUT}${PG}${CONFIG:+_$CONFIG}${PROTOCOL:+_$PROTOCOL}_seed*.out   (or nvidia-smi)"
