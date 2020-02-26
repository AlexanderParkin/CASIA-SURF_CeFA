'''SimpleNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F
import torch

cfg = {
    'simple': [8,16,32,64],
    'v2': [16,32,64,64]
}

class SimpleBlock(nn.Module):
    def __init__(self, in_planes, planes, bias=False, stride=1):
        super(SimpleBlock, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, bias=bias, stride=self.stride)
        self.bn1 = nn.BatchNorm2d(planes)
    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = F.relu(out)
        if self.stride == 1:
            out = F.max_pool2d(out, 2)
        return out

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.features = self._make_layers(cfg['v2'])
        #self.classifier = nn.Linear(128, 2)

    def forward(self, x):
        out = F.relu(self.features(x))
        out = out.view(out.size(0), -1)
        #out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            layers.append(SimpleBlock(in_channels, x, True, stride=1))
            in_channels = x
        layers.append(nn.Conv2d(in_channels, 2 * in_channels, kernel_size=5))
        return nn.Sequential(*layers)
