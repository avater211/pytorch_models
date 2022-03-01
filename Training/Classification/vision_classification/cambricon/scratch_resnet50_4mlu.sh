#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)
VIS_DIR=$(cd ${CUR_DIR}/../;pwd)
pushd $VIS_DIR

export MLU_VISIBLE_DEVICES=0,1,2,3
# train
python classify.py -a resnet50 --iters -1 --batch-size 64 --lr 0.1 --device mlu --momentum 0.9  --wd 1e-4  --seed 42 --data $IMAGENET_TRAIN_DATASET --logdir resnet50_four_card_log --epochs 100 --save_ckp --ckpdir resnet50_four_card_ckps  --multiprocessing-distributed -j8 --dist-backend cncl --world-size 1 --rank 0 

# eval
python classify.py -a resnet50 -e --batch-size 64 --device mlu --logdir resnet50_four_card_log --seed 42 --data $IMAGENET_TRAIN_DATASET --iters -1 --resume resnet50_four_card_ckps/resnet50_100.pth

popd
