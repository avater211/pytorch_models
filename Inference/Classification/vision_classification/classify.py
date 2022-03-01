from __future__ import division
import sys
import time
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torchvision import datasets

import os
import numpy as np
from PIL import Image
import logging
import argparse
import json

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../tools/utils/")
from common_utils import Timer

torch.set_grad_enabled(False)

#configure logging path
logging.basicConfig(level=logging.INFO,
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger("TestNets")

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

abs_path=sys.path[0]+"/"
parser = argparse.ArgumentParser(description='Pre-checkin and Daily test script.')
parser.add_argument("--data", dest = 'data', help =
                    "The path of imagenet dataset.",
                    default = "./imagenet", type = str)
parser.add_argument('-j', default=4, type=int, metavar='N',
                    help='number of data loading works (default: 4)')
parser.add_argument("--batch_size", dest = "batch_size", help =
                    "batch size for one inference.",
                    default = 1,type = int)
parser.add_argument('--device', default='cpu', type=str,
                    help='Use cpu gpu or mlu device')
parser.add_argument('--device_id', default=None, type=int,
                    help='Use specified device for training, useless in multiprocessing distributed training')
parser.add_argument("--jit", default=True, type=str2bool,
                    help="if use jit trace")
parser.add_argument("--jit_fuse", default=True, type=str2bool,
                    help="if use jit fuse mode")
parser.add_argument("--input_data_type", dest = 'input_data_type', help =
                    "the input data type, float32 or float16, default float32.",
                    default = 'float32', type = str)
parser.add_argument("--qint", default='no_quant', dest = 'qint', help =
                    "the quantized data type for conv/linear ops, float or int8 or int16, default float.")
parser.add_argument('--iters', type=int, default=60000, metavar='N',
                    help='iters per epoch')
parser.add_argument('--dummy_test', dest='dummy_test', action='store_true',
                    help='use fake data to inferrence')
parser.add_argument('-p', '--print-freq', default=1, type=int,
                    metavar='N', help='print frequency (default: 1)')
parser.add_argument("--quantized_iters", default = 5, dest = 'quantized_iters', type = int,
                    help = "Set image numbers to evaluate quantized params, default is 5.")
parser.add_argument("--first_conv", default = False, dest = 'first_conv', type = str2bool,
                    help = "Enable First Conv for infernce on MLU, default False.")

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
args = parser.parse_args()
print(args)
if args.device == 'mlu':
    import torch_mlu
    import torch_mlu.core.mlu_model as ct
    if args.qint == 'int8' or 'int16':
        import torch_mlu.core.mlu_quantize as mlu_quantize

class dummy_data_loader():
    def __init__(self, len = 0, images_size = (3, 224, 224), batch_size = 1, num_classes = 1000):
        self.len = len
        self.images = torch.normal(mean = -0.03 , std = 1.24, size = (batch_size,)+images_size)
        self.target = torch.randint(low = 0, high = num_classes, size = (batch_size,))
        self.data = 0
    def __iter__(self):
        return self
    def __len__(self):
        return self.len
    def __next__(self):
        if self.data > self.len:
            raise StopIteration
        else:
            self.data += 1
            return self.images, self.target

def saveResult(imageNum,batch_size,top1,top5,meanAp,hardwaretime,endToEndTime,threads=1):
    if not os.getenv('OUTPUT_JSON_FILE'):
        return
    TIME=-1
    hardwareFps=-1
    endToEndFps=-1
    latencytime=-1
    if hardwaretime!=TIME:
        hardwareFps=imageNum/hardwaretime
        latencytime = hardwaretime / (imageNum / batch_size) * 1000
    if endToEndTime!=TIME:
        endToEndFps=imageNum/endToEndTime
    if top1!=TIME:
        top1=top1/imageNum
    if top5!=TIME:
        top5=top5/imageNum
    result={
            "Output":{
                "Accuracy":{
                    "top1":'%.2f'%top1,
                    "top5":'%.2f'%top5,
                    "meanAp":'%.2f'%meanAp
                    },
                "HostLatency(ms)":{
                    "average":'%.2f'%latencytime,
                    "throughput(fps)":'%.2f'%endToEndFps,
                    }
                }
            }
    with open(os.getenv("OUTPUT_JSON_FILE"),"a") as outputfile:
        json.dump(result,outputfile,indent=4,sort_keys=True)
        outputfile.write('\n')
        outputfile.close()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def main():

    if args.device_id is None:
        args.device_id = 0  # Default Device is 0

    # Create the network
    print("processing: ", args.arch)
    net = getattr(models, args.arch)(pretrained=True)
    net.eval().float()

    # Load dataset for quantization/inferencing.
    print ("=> loading dataset...")
    resize = 256
    crop = 224
    data_scale = 1.0
    use_avg = False
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    valdir = os.path.join(args.data, 'val')
    val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(crop),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
                ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.j, pin_memory=True)
    total_image_number = len(val_loader) * args.batch_size

    if args.dummy_test:
        val_loader = dummy_data_loader(len = len(val_loader), batch_size = args.batch_size)

    # Converting model to int8/int16 quantized model and evaluate q_params
    if args.qint != 'no_quant' and args.device == 'mlu':
        # First evaluating quantized params
        print ("=> evaluating quantized param using {} images...".format(args.quantized_iters))
        if args.first_conv is True:
            qconfig = {'use_avg': False, 'data_scale': 1.0, 'mean': mean, 'std': std,
                       'firstconv': True, 'per_channel': False, 'iteration': args.quantized_iters}
        else:
            qconfig = {'use_avg': False, 'data_scale': 1.0, 'mean': None, 'std': None,
                       'firstconv': False, 'per_channel': False, 'iteration': args.quantized_iters}
        quantized_model = mlu_quantize.quantize_dynamic_mlu(net, qconfig, dtype = args.qint, gen_quant = True)

        # Preparing input images for quantization
        icount = 0
        for i, (images, target) in enumerate(val_loader):
            for img in images:
                _ = quantized_model(img.unsqueeze(0))
                icount += 1
                if icount == args.quantized_iters:
                    break
            if icount == args.quantized_iters:
                    break
        checkpoint = quantized_model.state_dict()
        # Second converting model into quantized_model and load the params.
        net = mlu_quantize.quantize_dynamic_mlu(net)
        net.load_state_dict(checkpoint)

    # prepare the example input for jit.trace()
    in_h = 224
    in_w = 224
    model = None
    example_input = torch.randn(args.batch_size, 3, in_h, in_w, dtype=torch.float)
    if args.jit:
        logger.info('Test in jit mode')
        if args.device == "mlu":
            ct._jit_override_can_fuse_on_mlu(args.jit_fuse)
        elif args.device == "gpu":
           torch._C._jit_override_can_fuse_on_gpu(args.jit_fuse)

        if args.qint != "no_quant" and args.device == "mlu":
            model = torch.jit.trace(net.to('mlu'), example_input.to('mlu'), check_trace=False)
        else:
            model = torch.jit.trace(net, example_input, check_trace=False)
    else:
        logger.info('Test in eager mode')
        model = net

    if args.input_data_type == 'float16':
        model.half()
        example_input = example_input.half()

    if args.device == "mlu":
        # Set the MLU device id
        ct.set_device(args.device_id)
        print("Use MLU{} for inferrence".format(args.device_id))

        model.to(ct.mlu_device())
        example_input = example_input.to(ct.mlu_device(), non_blocking=True)
    elif args.device == "gpu":
        # Set the GPU device id
        torch.cuda.set_device(args.device_id)
        print("Use GPU{} for inferrence".format(args.device_id))

        model.to(torch.device("cuda"))
        example_input = example_input.cuda(args.device_id, non_blocking=True)

    if args.jit and args.jit_fuse:
        print("=> warmuping JIT Fuse mode...")
        output = model(example_input)

    batch_time = AverageMeter('Time' , ':6.3f')
    data_time  = AverageMeter('Data' , ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time, top1, top5],
        prefix='Test: ')

    # Doing inference
    if args.first_conv is True and args.qint != "no_quant" and args.device == 'mlu':
        print("==> reload the dataset for MLU First Conv inference...")
        mean_mlu = [0.0, 0.0, 0.0]
        std_mlu = [1/255,1/255, 1/255]
        val_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(valdir, transforms.Compose([
                    transforms.Resize(resize),
                    transforms.CenterCrop(crop),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean_mlu, std=std_mlu),
                    ])),
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.j, pin_memory=True)
        print("==> finished reloading...")
    print ("=> inferring process...")
    total_e2e = 0
    model_forward_time = 0
    end = time.time()
    total = time.time()
    for i, (images, target) in enumerate(val_loader):

        if i == args.iters:
            total_image_number = args.iters * args.batch_size
            break

        if images.size(0) < args.batch_size:
            for index in range(images.size(0) + 1, args.batch_size + 1):
                images = torch.cat((images, images[0].unsqueeze(0)), 0)
                target = torch.cat((target, target[0].unsqueeze(0)), 0)

        data_time.update(time.time() - end)

        if args.input_data_type == 'float16':
            images = images.half()

        if args.device == 'gpu':
            images = images.cuda(args.device_id, non_blocking=True)
            target = target.cuda(args.device_id, non_blocking=True)
        elif args.device == 'mlu':
            images = images.to(ct.mlu_device(), non_blocking=True)
            target = target.to(ct.mlu_device(), non_blocking=True)
        if args.dummy_test:
            run_start = time.time()
        output = model(images)
        if args.dummy_test:
            current_device = torch_mlu._MLUC._get_device()
            current_qu = torch_mlu._MLUC._getCurrentQueue(current_device)
            current_qu.synchronize()
            model_forward_time += time.time() - run_start

        acc1, acc5 = accuracy(output.float(), target, topk=(1, 5))

        # measure elapsed time
        if not args.dummy_test:
            batch_time.update(time.time() - end)
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        if i % args.print_freq == 0:
            progress.display(i)

        end = time.time()

    if not args.dummy_test:
        total_e2e = time.time() - total

    logger.info('Global accuracy:')
    logger.info('accuracy1: ' +  str(top1.avg.item()))
    logger.info('accuracy5: ' +  str(top5.avg.item()))
    if total_image_number > args.batch_size:
        if not args.dummy_test:
            logger.info('latency: ' + str(total_e2e / total_image_number * args.batch_size * 1000))
            logger.info('throughput: ' + str(total_image_number / total_e2e))
            saveResult(total_image_number,args.batch_size,top1.avg.item(),top5.avg.item(),-1,-1,total_e2e)
        else:
            logger.info('latency: ' + str(model_forward_time / total_image_number * args.batch_size * 1000))
            logger.info('throughput: ' + str(total_image_number / model_forward_time))
            saveResult(total_image_number,args.batch_size,top1.avg.item(),top5.avg.item(),-1,model_forward_time,-1)
    else:
        saveResult(total_image_number,args.batch_size,top1.avg.item(),top5.avg.item(),-1,0,0)

if __name__ == '__main__':

    dataset_path = args.data
    assert dataset_path != "", "imagenet dataset should be provided."

    # set pretrained model path
    TORCH_HOME = os.getenv('TORCH_HOME')
    if TORCH_HOME == None:
        print("Warning: please set environment variable TORCH_HOME such as $PWD/models/pytorch")
        exit(1)

    torch.hub.set_dir(os.getenv('TORCH_HOME'))
    print("TORCH_HOME:  ", TORCH_HOME)

    main()
