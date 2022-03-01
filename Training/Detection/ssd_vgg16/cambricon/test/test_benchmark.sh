#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)

# 帮助函数
function usage () {
    echo -e "\033[32m Usage : \033[0m"
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
    echo "|  bash $0 precision-device-[options...]"
    echo "|      Supported options:"
    echo "|             precision: fp32, O0, O1, O2, O3, amp"
    echo "|             device: mlu, gpu"
    echo "|             option1(multicards): ddp"
    echo "|                                                   "
    echo "|  eg.1. bash test_benchmark.sh fp32-mlu"
    echo "|      which means running ssd_vgg16 on single MLU card with fp32 precision."
    echo "|                                                   "
    echo "|  eg.2. export MLU_VISIBLE_DEVICES=0,1,2,3 && bash test_benchmark.sh fp32-mlu-ddp"
    echo "|      which means running ssd_vgg16 on 4 MLU cards with fp32 precision."
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
}

while getopts 'h:' opt; do
   case "$opt" in
       h)  usage ; exit 1 ;;
       ?)  echo "unrecognized optional arg : "; $opt; usage; exit 1;;
   esac
done

## 加载参数配置
config=$1
source ${CUR_DIR}/params_config.sh
set_configs "$config"

# config配置到网络脚本的转换
main() {
    train_cmd="OMP_NUM_THREADS=1 python train.py --start_iter 60000 --dataset VOC --seed 42 --dataset_root $dataset --device $device --dist-backend $dist_backend --resume $ckpt $distributed --world-size 1 --rank 0 --iters $train_iters --batch_size $batch_size --lr $lr"

    # 配置混合精度相关参数
    if [[ $precision != "fp32" ]]; then
        if [[ $precision == "amp" ]]; then
            train_cmd="${train_cmd} --pyamp"
        else
            echo "SSD_VGG16 have not supported CNMIX yet, please run precision fp32 or AMP."
            exit 1
        fi
    fi

    # 运行训练脚本
    echo "cmd---------------------------------------"
    echo "train_cmd: $train_cmd"
    eval "$train_cmd"
    echo "cmd---------------------------------------"

    # 是否要推理
    if [[ $evaluate == "True" ]]; then
        eval_cmd="python eval.py --trained_model ./${device}_weights/ssd300_COCO_60002.pth --voc_root $dataset --device $device --eval_iters $eval_iters"
        # 运行推理脚本
        echo "cmd---------------------------------------"
        echo "eval_cmd: $eval_cmd"
        eval "$eval_cmd"
        echo "cmd---------------------------------------"
    fi
}

pip install -r ../perf_requirements.txt
pushd $CUR_DIR/../../
main
popd