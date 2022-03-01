#!/bin/bash

CONFIG_DIR=$(cd $(dirname $0);pwd)

resnet50_base_params () {
    net="resnet50"
    device="mlu"

    batch_size="64"
    lr="0.025"
    weight_decay="1e-4"
    momentum="0.9"
    seed="42"
    epochs=75
    precision="fp32"
    num_workers="12"
    ddp="False"

    benchmark_mode="True"
    max_batch_size_MLU290="256"
    max_batch_size_MLU370="224"
    max_batch_size_MLU370_ECC="224"
    max_batch_size_V100="128"

    resume="${IMAGENET_TRAIN_CHECKPOINT}/resnet50/epoch_74.pth"
}

set_configs () {
    args=$1

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
            fp32)   ;;
            O[0-3]) precision=$var ;;
            amp)    precision="pyamp" ;;
            mlu)    ;;
            gpu)    device="gpu" ;;
            ddp)    ddp="True" ;;
            ci_train)  benchmark_mode=False;
                       iters=2;
                       resume_multi_device="True";
                       ;;
            ci_eval)   benchmark_mode=False;
                       iters=2;
                       resume_multi_device="True";
                       evaluate="True";
                       ;;
            *) echo "Unrecognized option: " $var; exit 1;;
        esac
    done

    # 处理benchmark_mode所需的参数
    if [[ $benchmark_mode == "True" ]]; then
        ## 加载公用方法
        source ${CONFIG_DIR}/../../../../tools/utils/common_utils.sh

        ## 获取benchmark_mode计数规则,配置迭代数
        iters=-1
        perf_iters_rule iters

        ## 设置benchmark_mode log路径
        export BENCHMARK_LOG=${CUR_DIR}/../../../../benchmark_log

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

        epochs=1
        resume=""

        visible_cards=-1
        get_visible_cards visible_cards
        ## 检查多卡时是否设置VISIBLE_DEVICES环境变量
        if [[ $ddp == "True" ]]; then
            if [ $visible_cards -eq -1 ]; then
                # echo "Please set env MLU_VISIBLE_DEVICES before running multicards."
                exit 1
            fi
        fi

        ## set num_workers for different platforms and cardnums
        num_workers="32"
        if [[ ${visible_cards} -eq 8 ]]; then
            num_workers="12"
        fi
        if [[ ${cur_platform} == "MLU370" ]]; then
            if [[ ${visible_cards} -eq 16 ]]; then
                num_workers="7"
            elif [[ ${visible_cards} -eq 4 ]]; then
                num_workers="16"
            fi
        fi

        ## 检查性能模式软硬件环境
        pushd ${CONFIG_DIR}/../../../../tools/mlu_performance_check/; ./check_mlu_perf.sh; popd;
    fi
}

