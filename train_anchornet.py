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
from utils import cal_param_size, cal_multi_adds, correct_num, adjust_lr, AverageMeter


from bisect import bisect_right
import time
import math


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default='/dev/shm/ImageNet/', type=str, help='trainset directory')
parser.add_argument('--dataset', default='ImageNet', type=str, help='Dataset name')
parser.add_argument('--arch', default='anchornet', type=str, help='network architecture')
parser.add_argument('--init-lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lr-type', default='multistep', type=str, help='learning rate strategy')
parser.add_argument('--milestones', default=[30, 60, 90], type=list, help='milestones for lr-multistep')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=256, help='batch size')
parser.add_argument('--num-workers', type=int, default=16, help='batch size')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--manual_seed', type=int, default=0)
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--evaluate', '-e', action='store_true', help='evaluate model')
parser.add_argument('--checkpoint-dir', default='./checkpoint', type=str, help='checkpoint dir')



# global hyperparameter set
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


log_txt = 'result/'+ str(os.path.basename(__file__).split('.')[0]) + '_'+\
          'arch'+args.arch + '_'+\
          'seed'+ str(args.manual_seed) +'.txt'

log_dir = str(os.path.basename(__file__).split('.')[0]) + '_'+\
          'arch'+args.arch + '_'+\
          'seed'+ str(args.manual_seed)

with open(log_txt, 'a+') as f:
    f.write("==========\nArgs:{}\n==========".format(args) + '\n')
    
np.random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)
best_acc = 0.  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch



args.checkpoint_dir = os.path.join(args.checkpoint_dir, log_dir)
if not os.path.isdir(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)
if not os.path.isdir(os.path.join(args.checkpoint_dir, 'result')):
    os.makedirs(os.path.join(args.checkpoint_dir, 'result'))
log_txt = os.path.join(args.checkpoint_dir, log_txt)

num_classes = 1000
train_set = datasets.ImageFolder(
    os.path.join(args.data, 'train'),
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]))

test_set = datasets.ImageFolder(
    os.path.join(args.data, 'val'),
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
]))

trainloader = torch.utils.data.DataLoader(
    train_set, batch_size=args.batch_size, shuffle=True,
    num_workers=args.num_workers, pin_memory=True)

testloader = torch.utils.data.DataLoader(
    test_set, batch_size=args.batch_size, shuffle=False,
    num_workers=args.num_workers, pin_memory=True)
# --------------------------------------------------------------------------------------------

# Model
print('==> Building model..')
model = getattr(models, args.arch)
net = model(num_classes=num_classes)
print('Params: %.2fM, Multi-adds: %.2fG'
      % (cal_param_size(net)/1e6, cal_multi_adds(net, (2, 3, 224, 224))/1e9))
del(net)

net = model(num_classes=num_classes).cuda()
net = torch.nn.DataParallel(net)


cudnn.benchmark = True



# Training
def train(epoch, criterion_list, optimizer):
    train_loss_cls = AverageMeter('train_loss_cls', ':.4e')
    correct_1 = 0
    correct_5 = 0
    total = 0
    
    lr = adjust_lr(optimizer, epoch, args)
    start_time = time.time()
    criterion_cls = criterion_list[0]

    net.train()
    batch_start_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.cuda()
        targets = targets.cuda()

        optimizer.zero_grad()
        _, _, logits = net(inputs)

        loss_cls = criterion_cls(logits, targets)
        
        loss_cls.backward()
        optimizer.step()

        train_loss_cls.update(loss_cls.item(), inputs.size(0))

        prec1, prec5 = correct_num(logits, targets, topk=(1, 5))
        correct_1 += prec1
        correct_5 += prec5
        total += targets.size(0)

        print('Epoch:{}, batch_idx:{}/{}, lr:{:.5f}, Duration:{:.2f}'
              .format(epoch, batch_idx, len(trainloader),
              lr, time.time()-batch_start_time))
        batch_start_time = time.time()


    with open(log_txt, 'a+') as f:
        f.write('Epoch:{}\t lr:{:.4f}\t duration:{:.3f}'
                '\ttrain_loss_cls:{:.5f}'
                '\nTrain accuracy_1: {:.4f}\nTrain accuracy_5: {:.4f}\n'
                .format(epoch, lr, time.time() - start_time,
                        train_loss_cls.avg,
                        correct_1/total,
                        correct_5/total))


def test(epoch, criterion_cls):
    net.eval()
    test_loss_cls = AverageMeter('test_loss_cls', ':.4e')

    correct_1 = 0
    correct_5 = 0
    total = 0

    with torch.no_grad():
        batch_start_time = time.time()
        for batch_idx, (inputs, target) in enumerate(testloader):
            inputs, target = inputs.cuda(), target.cuda()
            _, _, logits = net(inputs)
            loss_cls = criterion_cls(logits, target)

            test_loss_cls.update(loss_cls.item(), inputs.size(0))


            print('Epoch:{}, batch_idx:{}/{}, Duration:{:.2f}'.format(epoch, batch_idx, \
                                len(testloader), time.time()-batch_start_time))
            batch_start_time = time.time()

            prec1, prec5 = correct_num(logits, target, topk=(1, 5))
            correct_1 += prec1
            correct_5 += prec5
            total += target.size(0)
        
        with open(log_txt, 'a+') as f:
            f.write('test epoch:{}\t'
                    'test_loss_cls:{:.5f}\t'
                    'Test accuracy_1:{}\n'
                    'Test accuracy_5:{}\n'
                    .format(epoch, test_loss_cls.avg, 
                            correct_1/total,
                            correct_5/total))

    return correct_1/total


if __name__ == '__main__':
    criterion_cls = nn.CrossEntropyLoss()

    if args.evaluate: 
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar'),
                                map_location=torch.device('cpu'))
        net.module.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
        test(start_epoch, criterion_cls)
    else:
        trainable_list = nn.ModuleList([])
        trainable_list.append(net)
        optimizer = optim.SGD(trainable_list.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)

        criterion_list = nn.ModuleList([])
        criterion_list.append(criterion_cls)  # classification loss
        criterion_list.cuda()

        if args.resume: 
            checkpoint = torch.load(os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'),
                                    map_location=torch.device('cpu'))
            net.module.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch']

        for epoch in range(start_epoch, args.epochs):
            train(epoch, criterion_list, optimizer)
            acc = test(epoch, criterion_cls)

            state = {
                'net': net.module.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'))

            is_best = False
            if best_acc < acc:
                best_acc = acc
                is_best = True

            if is_best:
                shutil.copyfile(os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'),
                                os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar'))
        print('Evaluate the best model:')
        print('load pre-trained weights from: {}'.format(os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar')))
        args.evaluate = True
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar'),
                                map_location=torch.device('cpu'))
        net.module.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']

        
        top1_acc = test(start_epoch, criterion_cls)

        torch.save(checkpoint['net'], os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth'))
        with open(log_txt, 'a+') as f:
            f.write('best_accuracy: {} \n'.format(top1_acc))
        print('best_accuracy: {} \n'.format(top1_acc))
