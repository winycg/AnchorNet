
import os
import sys
import time
import math

import operator
from functools import reduce
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init
from bisect import bisect_right
import numpy as np


def cal_param_size(model):
    return sum([i.numel() for i in model.parameters()])


count_ops = 0
def measure_layer(layer, x, multi_add=1):
    delta_ops = 0
    type_name = str(layer)[:str(layer).find('(')].strip()

    if type_name in ['Conv2d']:
        out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) //
                    layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) //
                    layer.stride[1] + 1)
        delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] *  \
                layer.kernel_size[1] * out_h * out_w // layer.groups * multi_add

    ### ops_linear
    elif type_name in ['Linear']:
        weight_ops = layer.weight.numel() * multi_add
        bias_ops = 0
        delta_ops = weight_ops + bias_ops

    global count_ops
    count_ops += delta_ops
    return


def is_leaf(module):
    return sum(1 for x in module.children()) == 0


def should_measure(module):
    if is_leaf(module):
        return True
    return False


def cal_multi_adds(model, shape=(1,3,32,32)):
    global count_ops
    count_ops = 0
    data = torch.zeros(shape)

    def new_forward(m):
        def lambda_forward(x):
            measure_layer(m, x)
            return m.old_forward(x)
        return lambda_forward

    def modify_forward(model):
        for child in model.children():
            if should_measure(child):
                child.old_forward = child.forward
                child.forward = new_forward(child)
            else:
                modify_forward(child)

    def restore_forward(model):
        for child in model.children():
            if is_leaf(child) and hasattr(child, 'old_forward'):
                child.forward = child.old_forward
                child.old_forward = None
            else:
                restore_forward(child)

    modify_forward(model)
    model.forward(data)
    restore_forward(model)

    return count_ops


def confidence(logits, uncertainty_type='entropy'):
    probability = F.softmax(logits, dim=1)
    if uncertainty_type == 'entropy':
        return -torch.sum(probability * torch.log(probability),dim=1)
    elif uncertainty_type == 'confidence':
        return torch.max(probability, dim=1)[0]


def correct_num(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    correct = pred.eq(target.view(-1, 1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:, :k].float().sum()
        res.append(correct_k.item())
    return res

 
def IOU(bboxA, bboxB):
    x1 = bboxA[0]
    y1 = bboxA[1]
    width1 = bboxA[2] - bboxA[0]
    height1 = bboxA[3] - bboxA[1]

    x2 = bboxB[0]
    y2 = bboxB[1]
    width2 = bboxB[2] - bboxB[0]
    height2 = bboxB[3] - bboxB[1]

    endx = max(x1 + width1, x2 + width2)
    startx = min(x1, x2)
    width = width1 + width2 - (endx - startx)

    endy = max(y1 + height1, y2 + height2)
    starty = min(y1, y2)
    height = height1 + height2 - (endy - starty)

    if width <= 0 or height <= 0:
        ratio = 0
    else:
        Area = width * height
        Area1 = width1 * height1
        Area2 = width2 * height2
        ratio = Area * 1. / (Area1 + Area2 - Area)

    return ratio


def localize_bbox(pos_list, cam, patch_size, stride, topk, iou_thresholds):
    value = cam.flatten()
    argmaxx = value.argsort(descending=True)

    coordinate = []
    final_bbox = []
    count = 0
    for i in argmaxx:
        pt1 = (pos_list[i][0] * stride, pos_list[i][1] * stride)
        bbox = (pt1[0], pt1[1], pt1[0] + patch_size - 1, pt1[1] + patch_size - 1)
        if i == 0:
            count = count + 1
            final_bbox.append(bbox)
            if count == topk:
                return final_bbox
        else:
            flag = True
            for j in range(len(final_bbox)):
                if IOU(final_bbox[j], bbox) > iou_thresholds[count-1]:
                    flag = False
            if flag:
                count = count + 1
                final_bbox.append(bbox)
                if count == topk:
                    return final_bbox
    return final_bbox


def crop_patches(pos_list, inputs, cam, patch_size, stride, topk, iou_thresholds):
    patches = []
    for i in range(cam.size(0)):
        final_bbox = localize_bbox(pos_list, cam[i], patch_size, stride, topk, iou_thresholds)
        for bbox in final_bbox:
            patches.append(inputs[i, :, bbox[0]: bbox[2]+1, bbox[1]: bbox[3]+1])
    patches = torch.stack(patches, dim=0)
    return patches


def adjust_lr(optimizer, epoch, args):
    cur_lr = 0.
    if args.lr_type == 'multistep':
        cur_lr = args.init_lr * 0.1 ** bisect_right(args.milestones, epoch)
    elif args.lr_type == 'cosine':
        cur_lr = args.init_lr * 0.5 * (1. + math.cos(np.pi * epoch / args.epochs))

    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr

        return cur_lr



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