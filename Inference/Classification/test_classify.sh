#!/bin/bash
set -e
CUR_DIR=$(cd $(dirname $0);pwd)

function usage
{
    echo "Usage:"
    echo "-------------------------------------------------------------"
    echo "|  $0 [0-1] precision-device-[options...]"
    echo "|  required param1: 0)resnet50"
    echo "|  required param2: precision: fp32, fp16 "
    echo "|                   device: mlu, gpu,           "
    echo "|                   option1(jit mode): jit      "
    echo "|                   option2(jit fuse mode): fuse   "
    echo "|                   option3(run inference on int8/16): int8, int16"
    echo "|                                                         "
    echo "|  eg. ./test_classify.sh 0 fp32-mlu"
    echo "|      which means running resnet50 net with fp32 precision in eager mode."
    echo "|                                                         "
    echo "|  eg. ./test_classify.sh 0 fp16-mlu-jit-fuse"
    echo "|      which means running resnet50 net with fp16 precision in jit fuse mode."
    echo "|                                                         "
    echo "|                                                         "
    echo "|  eg. ./test_classify.sh 0 fp32-mlu-jit-fuse-int8"
    echo "|      which means running resnet50 net with fp32 precision and int8 conv/linear ops in jit fuse mode."
    echo "|                                                         "
    echo "|  eg. ./test_classify.sh 0 fp16-mlu-jit-fuse-int16"
    echo "|      which means running resnet50 net with fp16 precision and int16 conv/linear ops in jit fuse mode."
    echo "-------------------------------------------------------------"
}

# Check numbers of argument
if [ $# -lt 2 ]; then
    echo "[ERROR] Not enough arguments."
    usage
    exit 1
fi

net_index=$1
configs=$2
quantized_iters=$3

# Paramaters check
if [[ $net_index =~ ^[0-9]+$ && $net_index -le 0 &&\
    ($configs =~ ^(fp32|fp16)-(mlu|gpu)(-jit)?(-fuse)?(-int8|-int16)?(-ci_eval)?$ || $configs =~ ^ci[_a-z]*$) ]]; then
    echo "Paramaters Exact."
else
    echo "[ERROR] Unknow Parameter : " $net_index $configs
    usage
    exit 1
fi

# get location of net
net_list=(resnet50)
net_name=${net_list[$net_index]}
net_location=""
case "$net_index" in
    0)  net_location=${CUR_DIR}/vision_classification;;
    ?)      echo "there is unrecognized net index."; $net_index; usage; exit 1;;
esac

# Checkout envs
if [ -z $IMAGENET_TRAIN_DATASET ]; then
    echo "[ERROR] Please set IMAGENET_TRAIN_DATASET."
    exit 1
fi

if [ -z $TORCH_HOME ]; then
    echo "[ERROR] Please set TORCH_HOME."
    exit 1
fi

checkstatus () {
    if (($?!=0)); then
        echo "work failed"
        exit -1
    fi
}

# Set dataset info
dataset_name="ImageNet2012"
export DATASET_NAME=$dataset_name

if [[ $net_index -le 1 ]]; then
    # vision_classification支持多个网络，需要给定net名称
    run_cmd="pushd ${net_location}/cambricon/test; bash test_benchmark.sh ${net_name}-${configs}; checkstatus; popd"
else
    run_cmd="pushd ${net_location}/cambricon/test; bash test_benchmark.sh ${configs}; checkstatus; popd"
fi

echo $run_cmd
eval $run_cmd
