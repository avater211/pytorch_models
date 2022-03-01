#!/bin/bash
set -e
CUR_DIR=$(cd $(dirname $0);pwd)

function usage
{
    echo "Usage:"
    echo "-------------------------------------------------------------"
    echo "|  $0 [0-8] precision-device-[options...]"
    echo "|  required param1: 0)ssd_vgg16 "
    echo "|  required param2: precision: fp32"
    echo "|                   device: mlu, gpu            "
    echo "|                   option1: ddp                "
    echo "|  eg. ./test_detection.sh 0 fp32-mlu"
    echo "|      which means running ssd_vgg16 on single MLU card with fp32 precision."
    echo "|  eg. export MLU_VISIBLE_DEVICES=0,1,2,3 && ./test_detection.sh 0 fp32-mlu-ddp"
    echo "|      which means running ssd_vgg16 on 4 MLU cards with fp32 precision."
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

# Paramaters check
if [[ $net_index =~ ^[0-8]$ && $net_index -le 0 &&\
 ($configs =~ ^(fp32)-(mlu|gpu)(-ddp)?(-ci.*)?$ || $configs =~ ^ci[_a-z]*$) ]]; then
    echo "Paramaters Exact."
else
    echo "[ERROR] Unknow Parameter : " $net_index $configs
    usage
    exit 1
fi
# get location of net
net_list=(ssd_vgg16)
net_name=${net_list[$net_index]}
net_location=""
case "$net_index" in
    0) net_location="${CUR_DIR}/ssd_vgg16";;
    ?) echo "there is unrecognized net index ."; $net_index; usage; exit 1;;
esac

# Checkout envs
if [ -z $PYTORCH_TRAIN_DATASET ]; then
    echo "[ERROR] Please set PYTORCH_TRAIN_DATASET."
    exit 1
fi

if [ -z $PYTORCH_TRAIN_CHECKPOINT ]; then
    echo "[ERROR] Please set PYTORCH_TRAIN_CHECKPOINT."
    exit 1
fi

checkstatus () {
    if (($?!=0)); then
        echo "work failed"
        exit -1
    fi
}

# Set dataset name
dataset_name=("VOC")
export DATASET_NAME=${dataset_name[$net_index]}

run_cmd="pushd ${net_location}/cambricon/test; bash test_benchmark.sh ${configs}; checkstatus; popd"

eval $run_cmd
