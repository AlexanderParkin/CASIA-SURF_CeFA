'''LiteMobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
from torch.nn import init
import torch.nn as nn
import math
import torch
import collections
import torch.nn.functional as F


class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1, use_bn=True):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,
                               in_planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               groups=in_planes,
                               bias=(not use_bn))
        if use_bn:
            self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes,
                               out_planes,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=(not use_bn))
        if use_bn:
            self.bn2 = nn.BatchNorm2d(out_planes)
        self.use_bn = use_bn

    def forward(self, x):
        if self.use_bn:
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
        else:
            out = F.relu(self.conv1(x))
            out = F.relu(self.conv2(out))
        return out


class LiteMobileNet(nn.Module):
    def __init__(self, input_channels=3, pretrained=None, use_bn=True, classifier=None):
        super(LiteMobileNet, self).__init__()
        self.cfg_lite_mobilenet = [(64, 2), 64, (128, 2), 128, (128, 2), 128, (256, 2), 256]
        self.conv1 = nn.Conv2d(input_channels,
                               32,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=(not use_bn))
        self.use_bn = use_bn
        if use_bn:
            self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32, cfg=self.cfg_lite_mobilenet)
        self.embedding_size = 256
        if pretrained is not None:
            self._load_pretrained_weights_path(pretrained)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = classifier

    def _make_layers(self, in_planes, cfg):
        layers = []
        for x in cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride, self.use_bn))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.use_bn:
            out = F.relu(self.bn1(self.conv1(x)))
        else:
            out = F.relu(self.conv1(x))
        out = self.layers(out)
        out = self.avg(out)
        out = out.view(out.size(0), -1)
        if self.classifier is not None:
            out = self.classifier(out)
        return out

    def _load_pretrained_weights_path(self, path):
        weights = torch.load(path, map_location='cpu')
        pretrained_weights = self.state_dict()
        for k, v in pretrained_weights.items():
            if k in weights:
                pretrained_weights[k] = weights[k]
        self.load_state_dict(pretrained_weights)
        print('Weights loaded.')


def test():
    net = LiteMobileNet(input_channels=1)
    x = torch.randn(2, 1, 112, 112)
    y = net(x)
    print (net)
    print(y.size())
