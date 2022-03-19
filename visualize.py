import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import os
import shutil
import argparse
import numpy as np
from PIL import Image
from PIL import ImageDraw

import models
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from utils import cal_param_size, cal_multi_adds, IOU


from bisect import bisect_right
import time
import math


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default='/dev/shm/ImageNet/', type=str, help='trainset directory')
parser.add_argument('--dataset', default='ImageNet', type=str, help='Dataset name')
parser.add_argument('--patch-size', default=95, type=int, help='patch size')
parser.add_argument('--iou-threshold', default=0.5, type=float, help='IOU threshold')
parser.add_argument('--stride', default=8, type=int, help='accumulated stride')
parser.add_argument('--topk', default=4, type=int, help='maximum number of patches')
parser.add_argument('--batch-size', type=int, default=512, help='batch size')
parser.add_argument('--num-workers', type=int, default=16, help='number workers')
parser.add_argument('--pretrained-anchornet', default='', type=str, help='pretrained weights')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--manual_seed', type=int, default=0)
parser.add_argument('--save-dir', default='./visualization/', type=str, help='checkpoint dir')


# global hyperparameter set
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

if not os.path.isdir(args.save_dir):
    os.makedirs(args.save_dir)

num_classes = 1000
np.random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)
print('==> Building model..')
anchornet = getattr(models, 'anchornet')(num_classes=num_classes)
print('AnchorNet_inference Params: %.2fM, Multi-adds: %.2fG'
      % (cal_param_size(anchornet)/1e6, cal_multi_adds(anchornet, (1, 3, 224, 224))/1e9))
del(anchornet)

anchornet = getattr(models, 'anchornet')(num_classes=num_classes).cuda()
print('load pre-trained weights for anchornet')
anchornet_checkpoint = torch.load(args.pretrained_anchornet, 
                                  map_location=torch.device('cpu'))
anchornet.load_state_dict(anchornet_checkpoint)
anchornet.eval()
print('load AnchorNet successfully!')

cudnn.benchmark = True


def localize_bbox(cam, patch_size, stride, topk):
    global pos_list

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
                if IOU(final_bbox[j], bbox) > args.iou_threshold:
                    flag = False
            if flag:
                count = count + 1
                final_bbox.append(bbox)
                if count == topk:
                    return final_bbox


def crop_patches(inputs, cam, patch_size, stride, topk):
    final_bbox = []
    for i in range(cam.size(0)):
        final_bbox = localize_bbox(cam[i], patch_size, stride, topk)
    return final_bbox


def pos_lists():
    pos_list_95 = []
    size_95 = 17
        
    for i in range(size_95):
        for j in range(size_95):
            pos_list_95.append((i, j))

    return pos_list_95


if __name__ == '__main__':
    pos_list = pos_lists()

    idx = 0
    image_dir = os.path.join(args.data, 'val')
    dirs = []
    for dir in os.listdir(image_dir):
        dirs.append(dir)
    dirs.sort()

    trans1 = transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224)])

    trans2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    label = -1
    with torch.no_grad():
        for dir in dirs:
            cur_dir = os.path.join(image_dir, dir)
            label = label + 1
            for jpeg in os.listdir(cur_dir):
                idx = idx + 1
                raw_image = Image.open(os.path.join(cur_dir, jpeg))
                if len(raw_image.split()) != 3:
                    continue

                raw_image = trans1(raw_image)
                image = trans2(raw_image.copy())
                image = torch.unsqueeze(image, 0).cuda()

                feas, fc_weights, logits = anchornet(image)
                cam_weight = torch.index_select(fc_weights, 0, logits.argmax(dim=1))
                cam_weight = cam_weight.unsqueeze(-1).unsqueeze(-1)
                cam = torch.mean(feas * cam_weight, dim=1)
                final_bbox = crop_patches(image, cam, patch_size=args.patch_size, stride=8, topk=args.topk)
                
                raw_image = raw_image.resize((224, 224))
                draw = ImageDraw.ImageDraw(raw_image)
                for bbox in final_bbox:
                    draw.rectangle(((bbox[1], bbox[0]),(bbox[3], bbox[2])), fill=None, outline='blue', width=5)
                raw_image.save(os.path.join(args.save_dir, str(idx)+'.jpg'))
                print('Localized Semantic Patches of the ' + str(idx)+ '-th Image')