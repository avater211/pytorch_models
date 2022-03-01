# Cambricon PyTorch Deep Learning ModelZoo

## 介绍
为了让用户快速方便的体验寒武纪芯片的优势，了解PyTorch模型在寒武纪软件栈上的适配流程，我们提供了各个领域典型网络的训练/推理demo以供参考，且后续还会持续增加。如果您对寒武纪pytorch_models有任何意见，欢迎给我们提需求，我们会尽快完善。

## 依赖

1. Cambricon Catch.
2. PyTorch v1.6.0和torchvision v0.7.0版本.
3. Python3环境

## 目录

### Training
Training目录提供以下领域典型网络的训练demo。

**图像分类（Image Classification）**

| 网络             | 板卡  | 单机多卡 |
| ------------------ | ------- | -------- |
| [resnet50](Training/Classification/vision_classification)           | 370 | yes      |

**目标检测（Object Detection）**

| 网络       | 板卡  | 单机多卡 |
| ------------ | ------- | -------- |
| [ssd_vgg16](Training/Detection/ssd_vgg16)                           | 370 | yes      |

#### 网络性能测试流程

** 多卡性能测试 **

多卡性能测试场景，通过 MLU_VISIBLE_DEVICE 设置MLU卡数。

例如：``export MLU_VISIBLE_DEVICE=0,1,2,3 `` 表示运行4卡DDP。

** 测试环境 **

发布Docker中已预装dllogger.如果是wheel包或者源码编译的方式运行Cambricon PyTorch，需要使用下面的命令单独安装 ``dllogger`` 。

pip --no-cache-dir --no-cache install git+https://github.com/NVIDIA/dllogger

### Inference
Inference目录提供以下领域典型网络在寒武纪推理引擎MagicMind上的demo。

**图像分类（Image Classification）**

| 网络             | 板卡  | 单机单卡 | 数据类型支持 |
| ------------------ | ------- | -------- | -------- |
| [resnet50](Inference/Classification/vision_classification)           | 370 | yes      | fp32/fp16/int8/int16    |

## FAQ
