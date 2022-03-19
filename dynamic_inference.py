import torch
import numpy as np
import argparse
import os


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--table-file', default='densenet201_95_0.3_0.pth', type=str, help='Table files')
parser.add_argument('--checkpoint-dir', default='./checkpoint', type=str, help='checkpoint dir')
parser.add_argument('--arch', default='densenet201', type=str, help='network architecture')
args = parser.parse_args()


if not os.path.isdir(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)
if not os.path.isdir(os.path.join(args.checkpoint_dir, 'result')):
    os.makedirs(os.path.join(args.checkpoint_dir, 'result'))

log_txt = 'result/dynamic_inference_'+args.arch+'.txt'
log_txt = os.path.join(args.checkpoint_dir, log_txt)

dict_ = torch.load(args.table_file)
thresholds_table = dict_['thresholds'].cpu()

setup = dict_['setup']

print(setup)

predictions_table = dict_['predictions'].cpu()


# densenet: 0.790
# res50: 0.752
if args.arch == 'densenet201':
    meta_computation = 0.790
elif args.arch == 'resnet50':
    meta_computation = 0.752
else:
    raise ValueError('not support this network architecture')

def dynamic_infer(predefined_thresholds):
    global thresholds_table
    global predictions_table

    thresholds = thresholds_table.clone()
    predictions = predictions_table.clone()
    computation = 0.
    
    stage_num = []
    stage_true_num = []
    for i in range(len(predefined_thresholds)):
        mask = thresholds[:, i] > predefined_thresholds[i]
        stage_num.append(mask.sum().item())
        stage_true_num.append(predictions[mask, i].sum().item())
        unmask = ~mask
        thresholds = thresholds[unmask]
        predictions = predictions[unmask]


    stage_true_num = np.array(stage_true_num)
    stage_num = np.array(stage_num)

    acc = stage_true_num.sum()/stage_num.sum()

    for i in range(len(stage_num)):
        if i == 0:
            computation += stage_num[i] * (meta_computation * (i+1))
        else:
            computation += (0.06 + (meta_computation * (i+1))) * stage_num[i]

    print('Acc:{:.4f}, FLOPs:{:.2f}G, stage_true_num:{}, stage_num:{}'.format(acc, computation/50000, str(stage_true_num),str(stage_num)))
    with open(log_txt, 'a+') as f:
        f.write('Acc:{:.4f}, FLOPs:{:.2f}G, stage_true_num:{}, stage_num:{}'.format(acc, computation/50000, str(stage_true_num),str(stage_num))+'\n')


all_predefined_thresholds = [
[0., 1.1, 1.1, 1.1, 1.1],
[1.1, 0, 1.1, 1.1, 1.1],
[1.1, 1.1, 0, 1, 1],
[1.1, 1.1, 1.1, 0, 1],
[1.1, 1.1, 1.1, 1.1, 0],
]
for v in all_predefined_thresholds:
    print(v)
    dynamic_infer(v)