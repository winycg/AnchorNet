import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


__all__ = ['anchornet']

def swish(x):
    return x * x.sigmoid()


class SE(nn.Module):
    '''Squeeze-and-Excitation block with Swish.'''

    def __init__(self, in_channels, se_channels):
        super(SE, self).__init__()
        self.se1 = nn.Conv2d(in_channels, se_channels,
                             kernel_size=1, bias=True)
        self.se2 = nn.Conv2d(se_channels, in_channels,
                             kernel_size=1, bias=True)

    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, (1, 1))
        out = swish(self.se1(out))
        out = self.se2(out).sigmoid()
        out = x * out
        return out


class Block(nn.Module):
    '''expansion + depthwise + pointwise + squeeze-excitation'''
    def __init__(self,
                 kernel_size,
                 in_channels,
                 expand_ratio,
                 out_channels,
                 stride,
                 se_ratio=0.25,
                 drop_rate=0.):
        super(Block, self).__init__()
        self.stride = stride
        self.drop_rate = drop_rate
        self.expand_ratio = expand_ratio

        # Expansion
        channels = int(expand_ratio * in_channels)
        self.conv1 = nn.Conv2d(in_channels,
                               channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        # Depthwise conv
        self.conv2 = nn.Conv2d(channels,
                               channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=0,
                               groups=channels,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        # SE layers
        se_channels = int(in_channels * se_ratio)
        self.se = SE(channels, se_channels)

        # Output
        self.conv3 = nn.Conv2d(channels,
                               out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Skip connection if in and out shapes are the same (MV-V2 style)
        self.has_skip = (stride == 1) and (in_channels == out_channels)

    def forward(self, x):
        out = x if self.expand_ratio == 1 else swish(self.bn1(self.conv1(x)))
        out = swish(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))

        if x.size(-1) != out.size(-1):
            diff = x.size(-1) - out.size(-1)
            x = x[:, :, :-diff, :-diff]

        if self.has_skip:
            out = out + x
        return out


class AnchorNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AnchorNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.bneck = nn.Sequential(
            Block(3, 16, 1, 16, 2, 0.25),
            Block(3, 16, 3, 24, 2, 0.25),
            Block(3, 24, 4, 24, 1, 0.25),
            Block(3, 24, 4, 48, 1, 0.25),
            Block(3, 48, 2, 96, 1, 0.25),
            Block(3, 96, 1.5, 96, 1),
            Block(3, 96, 1.5, 96, 1),
        )

        self.exp_conv = nn.Conv2d(96, 512, kernel_size=1, padding=0, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Linear(512, 1000)

        self.init_params()


    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = swish(self.bn1(self.conv1(x)))
        out = self.bneck(out)


        fea = self.exp_conv(out)
        
        out = self.avg_pool(fea)

        logit = self.fc(out.view(out.size(0), -1))

        return fea, self.fc.weight, logit


def anchornet(num_classes=1000):
    model = AnchorNet(num_classes=num_classes)
    return model


if __name__ == '__main__':
    net = anchornet_inference()
    y = net(torch.randn((2, 3, 224, 224)))
    print(y[0].size())
    from utils import cal_param_size, cal_multi_adds
    print('Params: %.2fM, Multi-adds: %.3fM'
          % (cal_param_size(net) / 1e6, cal_multi_adds(net, (2, 3, 224, 224)) / 1e6))

