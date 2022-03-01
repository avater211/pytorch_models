CUR_DIR=$(cd $(dirname $0);pwd)
pushd "${CUR_DIR}/../"

if [ -z $PYTORCH_TRAIN_DATASET ]; then
    echo "[ERROR] Please set PYTORCH_TRAIN_DATASET."
    exit 1
fi

if [ -z $PYTORCH_TRAIN_CHECKPOINT ]; then
    echo "[ERROR] Please set PYTORCH_TRAIN_CHECKPOINT."
    exit 1
fi

export MLU_VISIBLE_DEVICES=0,1,2,3

cmd="python train.py --dataset_root ${PYTORCH_TRAIN_DATASET}/VOCdevkit --save_folder ${PYTORCH_TRAIN_CHECKPOINT}/ssd_vgg16/basenet/ --dataset VOC --seed 42 --iters 60000 --device mlu --multiprocessing-distributed --world-size 1 --rank 0 --batch_size 64 --lr 2e-3 --dist-backend cncl"

echo "scratch train cmd: $cmd"
eval "$cmd"
if (($?!=0)); then
    echo "work on train failed"
    exit 1
fi

cmd="python eval.py --trained_model ./mlu_weights/ssd300_COCO_60000.pth --voc_root ${PYTORCH_TRAIN_DATASET}/VOCdevkit --device mlu"
echo "scratch eval cmd: $cmd"
eval "$cmd"
if (($?!=0)); then
    echo "work on eval failed"
    exit 1
fi

popd
