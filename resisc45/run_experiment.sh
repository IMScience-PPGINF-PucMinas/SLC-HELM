#!/bin/bash
# Full training + test of SCH_HELM2 with VGG features on RESISC45, with the protocol of the full VGG
# experiment on EuroSAT: every epoch goes through all 51 augmentations, ReduceLROnPlateau patience 10,
# early stopping patience 30, no EMA, two-branch MGP. Each seed is trained and then tested. Finished
# seeds are skipped, and an interrupted training resumes from its snapshot when the script is run again
# with the same arguments.
#
#   bash run_experiment.sh <netvlad|mean> <att|lse> [seeds...]      # default seeds: 1 16 20 58 123
#   CUDA_VISIBLE_DEVICES=1 bash run_experiment.sh netvlad lse 16     # one seed on one GPU
#   (run_parallel.sh starts one seed per GPU at once)
#
# Environment variables: CONFIG (hyperparameters and number of superpixels of configs/$CONFIG.json, e.g.
# trial_53; default: train.py's defaults), THRESHOLD (pooling percentile over the batch or each graph: batch |
# graph; default: train.py's or the configuration's), PYTHON (default: python), WORKERS (data-loading
# workers, default: 8), MLFLOW_TRACKING_URI (default: file store in ./mlruns), PROTOCOL (train/val/test protocol of
# split_protocol.py, e.g. tr20: 20% of the images for training and validation, the rest for test; default: the
# torchgeo splits), BACKBONE (per-pixel features: vgg16 | resnet50, ResNet50 conv1+bn1+relu interpolated to the image
# size; default: vgg16) and UPSAMPLE (that interpolation: bilinear | bicubic; default: bilinear).
set -u
cd "$(dirname "$0")"
USAGE="usage: bash run_experiment.sh <netvlad|mean> <att|lse> [seeds...]"
AGG=${1:?$USAGE}
READOUT=${2:?$USAGE}
case $AGG in netvlad | mean) ;; *) echo "aggregation must be netvlad or mean"; exit 1 ;; esac
case $READOUT in att | lse) ;; *) echo "readout must be att or lse"; exit 1 ;; esac
shift 2
SEEDS=${*:-1 16 20 58 123}
THRESHOLD=${THRESHOLD:-}
case $THRESHOLD in "" | batch) PG="" ;; graph) PG="_pg" ;; *) echo "THRESHOLD must be batch or graph"; exit 1 ;; esac
CONFIG=${CONFIG:-}
CONFIG_ARGS=""
PROTOCOL=${PROTOCOL:-}
[ -z "$PROTOCOL" ] || [ -f "splits/$PROTOCOL.npz" ] || { echo "splits/$PROTOCOL.npz not found (split_protocol.py)"; exit 1; }
BACKBONE=${BACKBONE:-}
UPSAMPLE=${UPSAMPLE:-}
case $UPSAMPLE in "" | bilinear | bicubic) ;; *) echo "UPSAMPLE must be bilinear or bicubic"; exit 1 ;; esac
case $BACKBONE in
    "" | vgg16) BB=""; [ -z "$UPSAMPLE" ] || { echo "UPSAMPLE applies only to BACKBONE=resnet50"; exit 1; } ;;
    resnet50) BB=resnet50_${UPSAMPLE:-bilinear} ;;  # runs tagged e.g. _resnet50_bicubic (VGG16 runs keep their names)
    *) echo "BACKBONE must be vgg16 or resnet50"; exit 1 ;;
esac
BACKBONE_ARGS="${BACKBONE:+ --backbone $BACKBONE}${UPSAMPLE:+ --upsample $UPSAMPLE}"
PYTHON=${PYTHON:-python}
WORKERS=${WORKERS:-8}
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-file:$(pwd)/mlruns}
export PYTHONUNBUFFERED=1
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

# The graphs (data/processed*) and images (data/images) are checked once in this environment: all files
# present and the images rebuilt here equal to the ones the superpixel maps were built from.
ENV_ID="$(hostname):$($PYTHON -c 'import sys; print(sys.prefix)')"  # the check is redone on another machine or environment
if [ "$(cat logs/$CHECK.ok 2>/dev/null)" != "$ENV_ID" ]; then
    $PYTHON check_dataset.py --nodes "$NODES" ${PROTOCOL:+--protocol $PROTOCOL} > logs/$CHECK.log 2>&1 \
        || { echo "dataset check failed, see logs/$CHECK.log"; exit 1; }
    echo "$ENV_ID" > logs/$CHECK.ok
fi

for s in $SEEDS; do
    tag=${AGG}_${READOUT}${BB:+_$BB}${PG}${CONFIG:+_$CONFIG}${PROTOCOL:+_$PROTOCOL}_seed$s
    args="--seed $s --aggregation $AGG --readout $READOUT $CONFIG_ARGS${THRESHOLD:+ --threshold $THRESHOLD}${PROTOCOL:+ --protocol $PROTOCOL}$BACKBONE_ARGS"
    [ -f logs/$tag.done ] && { echo "$tag: already done"; continue; }
    if command -v flock > /dev/null; then  # the same seed is never run twice at the same time
        exec 9> logs/$tag.lock
        flock -n 9 || { echo "$tag: running in another process, skipped"; continue; }
    fi
    if [ ! -f logs/$tag.trained ]; then
        echo "$tag: training ($(date))"
        $PYTHON train.py $args --workers "$WORKERS" --eval-workers 2 --snapshot-every 1 >> logs/train_$tag.log 2>&1 \
            || { echo "$tag: training failed, see logs/train_$tag.log"; exit 1; }
        touch logs/$tag.trained
    fi
    $PYTHON test.py $args --workers "$WORKERS" > logs/test_$tag.log 2>&1 \
        || { echo "$tag: test failed, see logs/test_$tag.log"; exit 1; }
    touch logs/$tag.done
    echo "$tag: $(grep -E '^Accuracy' logs/test_$tag.log) ($(date))"
    exec 9>&-
done
