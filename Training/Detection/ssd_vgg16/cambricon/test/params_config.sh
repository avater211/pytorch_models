#!/bin/bash

CONFIG_DIR=$(cd $(dirname $0);pwd)

base_params () {
    device="mlu"
    dist_backend="cncl"

    batch_size="32"      # when batch_size==64, GPU quantize simulate will be OOM, so we use 32
    train_iters="2"
    eval_iters="2"
    lr="1e-3"
    precision="fp32"
    ddp="False"
    card_num=1;
    distributed="--device_id 0"
    evaluate="False";
    benchmark_mode="True"

    max_batch_size_MLU290="32"
    max_batch_size_MLU370="18"
    max_batch_size_MLU370_ECC="18"
    max_batch_size_V100="32"

    dataset="${PYTORCH_TRAIN_DATASET}/VOCdevkit"
    ckpt="${PYTORCH_TRAIN_CHECKPOINT}/ssd_vgg16/checkpoints_fp/ssd300_COCO_60000.pth"
}

set_configs () {
    params=$1

    # 调用网络的base_params
    base_params

    # 根据每个字段的功能, overide对应参数
    params_array=(${params//-/ })
    for var in ${params_array[@]}
    do
        case "$var" in
            fp32)   ;;
            O[0-3]) precision=$var ;;
            amp)    precision="amp" ;;
            mlu)    ;;
            gpu)    device="gpu";
                    dist_backend="nccl" ;;
            ddp)    ddp="True";
                    distributed="--multiprocessing-distributed";
                    batch_size="16";
                    lr="5e-4" ;;
            ci)     benchmark_mode="False";
                    evaluate="True" ;;
            *) echo "Unrecognized option: " $var; exit 1;;
        esac
    done

    ## 加载公用方法
    source ${CONFIG_DIR}/../../../../tools/utils/common_utils.sh

    # 处理benchmark_mode所需的参数
    if [[ $benchmark_mode == "True" ]]; then

        ## 获取benchmark_mode计数规则,配置迭代数
        eval_iters=-1
        perf_iters_rule train_iters

        ## 设置benchmark_mode log路径
        export BENCHMARK_LOG=${CONFIG_DIR}/../../../../benchmark_log

        ## 获取平台类型，配置最大batch_size
        cur_platform=""
        get_platform cur_platform
        mbs_name=max_batch_size_${cur_platform}

        cur_ecc_status=""
        get_ecc_status cur_ecc_status
        if [[ ${cur_ecc_status} == "ON" ]]; then
            mbs_name=max_batch_size_${cur_platform}_ECC
        fi
        batch_size=${!mbs_name}

        ## 检查性能模式软硬件环境
        pushd ${CONFIG_DIR}/../../../../tools/mlu_performance_check/; ./check_mlu_perf.sh; popd;
    fi

    if [[ $ddp == "True" ]]; then
        get_visible_cards card_num
    fi

    if [[ $card_num -le 0 ]]; then
        echo "Invalid card number ${card_num} !!!"
        exit 1
    fi

    batch_size=`expr ${card_num} \* ${batch_size}`
}
