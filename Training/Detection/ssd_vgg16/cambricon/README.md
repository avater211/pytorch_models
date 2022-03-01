# SSD VGG16的MLU和GPU训练
  本项目关于Pytorch的SSD VGG16网络的MLU和GPU训练。

## 训练准备
   本项目需要下载初始化权重 (wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth)，使用VOC数据集。
   提前准备好MLU或者GPU的环境。

## 注意
* 该网络的训练数据预处理流程里的SSDAugmentation功能包含随机性，当DataLoader开多个worker进程时，会导致worker进程里的预处理结果具有随机性，可能会导致daily测试结果超GPU精度，所以daily测试时num_workers需要设成0。

## 遗留问题
* 目前MLU不支持像GPU一样torch.set_default_tensor_type('torch.cuda.FloatTensor')设置默认Tensor的机制，为保持GPU和MLU计算一致，暂时改成torch.set_default_tensor_type('torch.FloatTensor')




