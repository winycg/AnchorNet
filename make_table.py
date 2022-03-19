import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import os
import shutil
import argparse
import numpy as np


import models
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from utils import cal_param_size, cal_multi_adds, correct_num, IOU


from bisect import bisect_right
import time
import math


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default='/dev/shm/ImageNet/', type=str, help='trainset directory')
parser.add_argument('--dataset', default='ImageNet', type=str, help='Dataset name')
parser.add_argument('--arch', default='resnet50', type=str, help='network architecture')
parser.add_argument('--patch-size', default=95, type=int, help='patch size')
parser.add_argument('--iou-thresholds', type=float, nargs='+', default=[0.,0.1,0.2])
parser.add_argument('--stride', default=8, type=int, help='accumulated stride')
parser.add_argument('--topk', default=5, type=int, help='maximum number of patches')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=512, help='batch size')
parser.add_argument('--num-workers', type=int, default=16, help='number workers')
parser.add_argument('--pretrained-anchornet', default='', type=str, help='pretrained weights')
parser.add_argument('--resized-pretrained', default='', type=str, help='pretrained weights')
parser.add_argument('--patch-pretrained', default='', type=str, help='pretrained weights')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--manual_seed', type=int, default=0)
parser.add_argument('--save-dir', type=str, default='./tables/')

# global hyperparameter set
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


log_txt = 'result/'+ str(os.path.basename(__file__).split('.')[0]) + '_'+\
          'arch'+args.arch + '_'+\
          'seed'+ str(args.manual_seed) +'.txt'

log_dir = str(os.path.basename(__file__).split('.')[0]) + '_'+\
          'arch'+args.arch + '_'+\
          'seed'+ str(args.manual_seed)


np.random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)


if not os.path.isdir(args.save_dir):
    os.makedirs(args.save_dir)


num_classes = 1000
test_set = datasets.ImageFolder(
    os.path.join(args.data, 'val'),
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
]))

testloader = torch.utils.data.DataLoader(
    test_set, batch_size=args.batch_size, shuffle=False,
    num_workers=args.num_workers, pin_memory=True)
# --------------------------------------------------------------------------------------------

# Model
print('==> Building model..')
anchornet = getattr(models, 'anchornet')(num_classes=num_classes)
print('AnchorNet_inference Params: %.2fM, Multi-adds: %.3fG'
      % (cal_param_size(anchornet)/1e6, cal_multi_adds(anchornet, (1, 3, 224, 224))/1e9))
del(anchornet)

down_net = getattr(models, args.arch)(num_classes=num_classes).eval()
print(args.arch + ' Params: %.2fM, Multi-adds: %.3fG'
      % (cal_param_size(down_net)/1e6, cal_multi_adds(down_net, (1, 3, 224, 224))/1e9))
del(down_net)

anchornet = getattr(models, 'anchornet')(num_classes=num_classes).cuda()
print('load pre-trained weights for anchornet')
anchornet_checkpoint = torch.load(args.pretrained_anchornet, 
                                  map_location=torch.device('cpu'))
anchornet.load_state_dict(anchornet_checkpoint)
anchornet.eval()
print('load AnchorNet successfully!')

down_net = getattr(models, args.arch)(num_classes=num_classes).cuda()
print('load pre-trained weights for '+ args.arch)
checkpoint = torch.load(args.patch_pretrained, map_location=torch.device('cpu'))['net']
down_net.load_state_dict(checkpoint)
down_net.eval()
print('load ' + args.arch +' successfully!')

resized_down_net = getattr(models, args.arch)(num_classes=num_classes).cuda()
print('load pre-trained weights for resized '+ args.arch)
checkpoint = torch.load(args.resized_pretrained, map_location=torch.device('cpu'))['net']
resized_down_net.load_state_dict(checkpoint)
resized_down_net.eval()
print('load ' + args.arch +' successfully!')

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
                if IOU(final_bbox[j], bbox) > args.iou_thresholds[count-1]:
                    flag = False
            if flag:
                count = count + 1
                final_bbox.append(bbox)
                if count == topk:
                    return final_bbox
    return final_bbox


def crop_patches(inputs, cam, patch_size, stride, topk):
    patches = []
    for i in range(cam.size(0)):
        final_bbox = localize_bbox(cam[i], patch_size, stride, topk)
        for bbox in final_bbox:
            patches.append(inputs[i, :, bbox[0]: bbox[2]+1, bbox[1]: bbox[3]+1])
    patches = torch.stack(patches, dim=0)
    return patches


def pos_lists():
    pos_list_95 = []
    size_95 = 17
        
    for i in range(size_95):
        for j in range(size_95):
            pos_list_95.append((i, j))

    return pos_list_95
    
    

if __name__ == '__main__':
    pos_list = pos_lists()

    thresholds = []
    predictions = []


    correct1 = 0
    correct5 = 0
    total = 0
    with torch.no_grad():
        batch_start_time = time.time()
        for batch_idx, (inputs, target) in enumerate(testloader):
            inputs, target = inputs.cuda(), target.cuda()
            # resized 
            resized_inputs = F.interpolate(inputs, size=[args.patch_size, args.patch_size], mode='bicubic', align_corners=True)
            resized_logits = resized_down_net(resized_inputs)

            threshold, argmax_preds = F.softmax(resized_logits,dim=1).max(dim=1)

            threshold = threshold.view(-1, 1)
            pred = argmax_preds.eq(target).view(-1, 1)

            feas, fc_weights, logits = anchornet(inputs)
            cam_weight = torch.index_select(fc_weights, 0, logits.argmax(dim=1))
            cam_weight = cam_weight.unsqueeze(-1).unsqueeze(-1)
            cam = torch.mean(feas * cam_weight, dim=1)
            patches = crop_patches(inputs, cam, patch_size=args.patch_size, stride=8, topk=args.topk-1)
            print(patches.size())
            logits = down_net(patches)

            batch_size = target.size(0)
            cumsum_logits = resized_logits
            logits = logits.view(-1, args.topk-1, num_classes)
            for j in range(args.topk-1):
                cumsum_logits += logits[:, j, :]
                tmp_threshold, tmp_argmax_preds = F.softmax(cumsum_logits,dim=1).max(dim=1)
                tmp_pred = tmp_argmax_preds.eq(target).view(-1, 1)
                tmp_threshold = tmp_threshold.view(-1, 1)
                threshold = torch.cat([threshold, tmp_threshold], dim=1)
                pred = torch.cat([pred, tmp_pred], dim=1)

            thresholds.append(threshold)
            predictions.append(pred)

            logits = cumsum_logits
            prec1, prec5 = correct_num(logits, target, topk=(1, 5))
            correct1 += prec1
            correct5 += prec5
            total += target.size(0)
            
            print('batch_idx:{}/{}, Duration:{:.2f}, Mini-batch Top-1 ACC:{:.4f}, Overall Top-1 ACC:{:.4f}'
                .format(batch_idx, len(testloader), time.time()-batch_start_time, prec1/target.size(0), correct1/total))
            batch_start_time = time.time()

        thresholds = torch.cat(thresholds, dim=0)
        predictions = torch.cat(predictions, dim=0)
        dict_ = {'setup':args.arch+'_'+str(args.patch_size)+'_'+str(args.iou_thresholds), 'thresholds': thresholds, 'predictions':predictions}
        
        torch.save(dict_,
            os.path.join(args.save_dir,args.arch+'_'+str(args.patch_size)+'_'+str(args.manual_seed)+'.pth'))
        
        print('Size of threshold Table', thresholds.size())
        print('Size of prediction Table', predictions.size())

        acc1 = round(correct1/total, 4)
        acc5 = round(correct5/total, 4)
        
        print('Test accuracy_1:{:.4f}\n'
              'Test accuracy_5:{:.4f}\n'.format(acc1, acc5))


