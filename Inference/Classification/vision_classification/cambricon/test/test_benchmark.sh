#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)

# 帮助函数
function usage () {
    echo -e "\033[32m Usage : \033[0m"
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
    echo "|  bash $0 net-precision-device-[options...]"
    echo "|      Supported options:"
    echo "|             net: resnet50"
    echo "|             precision: fp32, fp16"
    echo "|             device: mlu, gpu"
    echo "|             option1(jit mode): jit"
    echo "|             option2(jit fuse mode): fuse"
    echo "|             option3(run inference on int8/16): int8, int16"
    echo "|                                                   "
    echo "|  eg.1. bash test_benchmark.sh resnet50-fp32-mlu"
    echo "|      which means running resnet50 net with fp32 precision in eager mode."
    echo "|                                                   "
    echo "|  eg.2. bash test_benchmark.sh resnet50-fp16-mlu-jit-fuse"
    echo "|      which means running resnet50 net with fp16 precision in jit fuse mode."
    echo "|                                                   "
    echo "|  eq.3. bash test_benchmark.sh resnet50-fp32-mlu-jit-fuse-int8"
    echo "|      which means running resnet50 net with fp32 precision and int8 conv/linear ops in jit fuse mode."
    echo "|                                                   "
    echo "|  eq.4. bash test_benchmark.sh vgg19-fp16-mlu-jit-fuse-int16"
    echo "|      which means running vgg19 net with fp16 precision and int16 conv/linear ops in jit fuse mode."
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
}

# 获取用户指定config函数并执行,得到对应config的参数配置
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
    run_cmd="python \
             $CUR_DIR/../../classify.py \
             -a $net \
             --batch_size $batch_size \
             --device $device \
             --data $IMAGENET_TRAIN_DATASET \
             -j $num_workers \
             --jit $jit \
             --jit_fuse $jit_fuse \
             --input_data_type $input_data_type \
             --qint $qint \
             --quantized_iters $quantized_iters \
             --first_conv $first_conv"

    # 配置迭代次数
    if [[ $iters ]]; then
        run_cmd="${run_cmd} --iters ${iters}"
    fi

    # 参数配置完毕，运行脚本
    echo "cmd---------------------------------------"
    echo "$run_cmd"
    eval "${run_cmd}"
    echo "cmd---------------------------------------"
}


pushd $CUR_DIR
main
popd
