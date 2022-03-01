#!/bin/bash
set -e
CUR_DIR=$(cd $(dirname $0);pwd)

# Checkout envs
if [ -z $IMAGENET_TRAIN_DATASET ]; then
    echo "[ERROR] Please set IMAGENET_TRAIN_DATASET Env."
    exit 1
fi

if [ -z $TORCH_HOME ]; then
    echo "[ERROR] Please set TORCH_HOME Env."
    exit 1
fi

# eval
python $CUR_DIR/../classify.py -a resnet50 --data $IMAGENET_TRAIN_DATASET -j 12 --device gpu --jit true --jit_fuse true --device_id 0 --batch_size 64 --input_data_type float32
