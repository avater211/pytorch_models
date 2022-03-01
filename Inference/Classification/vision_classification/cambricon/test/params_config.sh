#!/bin/bash

CONFIG_DIR=$(cd $(dirname $0);pwd)

resnet50_base_params () {
    net="resnet50"
    device="mlu"

    batch_size="64"
    jit="False"
    jit_fuse="False"
    input_data_type="float32"
    num_workers="32"
    qint="no_quant"
    quantized_iters="5"
    first_conv="False"

    benchmark_mode="True"
}

set_configs () {
    args=$1
    echo $args
    # 获取网络和参数字段
    net=${args%%-*}
    params=${args#*-}

    # 调用相应网络的base_params
    ${net}_base_params

    # 根据每个字段的功能, overide对应参数
    params_array=(${params//-/ })
    for var in ${params_array[@]}
    do
        case "$var" in
            fp32)            input_data_type="float32" ;;
            fp16)            input_data_type="float16" ;;
            mlu)             ;;
            gpu)             device="gpu" ;;
            jit)             jit="True" ;;
            fuse)            jit_fuse="True" ;;
            int8)            qint="int8" ;;
            int16)           qint="int16" ;;
            ci_eval)         benchmark_mode=False;
                             iters=2;
                             ;;
            *) echo "Unrecognized option: " $var; exit 1;;
        esac
    done
}

