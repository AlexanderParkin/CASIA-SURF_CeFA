'''SimpleNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F
import torch

cfg = {
    'v1': [ 8, 16, 32,  64],
    'v2': [16, 32, 64,  64],
    'v3': [16, 32, 64, 128]
}

class SimpleBlock(nn.Module):
    def __init__(self, in_planes, planes, bias=False, stride=1, merged=False):
        super(SimpleBlock, self).__init__()
        self.merged = merged
        self.stride = stride
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, bias=bias, stride=self.stride)
        
        if not self.merged:
            self.bn1 = nn.BatchNorm2d(planes)
    
    def forward(self, x):
        out = self.conv1(x)
        
        if not self.merged:
            out = self.bn1(out)
            
        out = F.relu(out)
        if self.stride == 1:
            out = F.max_pool2d(out, 2)
        return out

class SimpleNet112(nn.Module):
    def __init__(self):
        super(SimpleNet112, self).__init__()
        self.features = self._make_layers(cfg['v3'])

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)

        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            layers.append(SimpleBlock(in_channels, x, True, stride=1, merged=False))
            in_channels = x
        layers.append(nn.Conv2d(in_channels, 2 * in_channels, kernel_size=5))
        return nn.Sequential(*layers)
    

class SimpleNetMerged(nn.Module):
    def __init__(self):
        super(SimpleNetMerged, self).__init__()
        self.features = self._make_layers(cfg['v3'])

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)

        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            layers.append(SimpleBlock(in_channels, x, True, stride=1, merged=True))
            in_channels = x
        layers.append(nn.Conv2d(in_channels, 2 * in_channels, kernel_size=5))
        return nn.Sequential(*layers)
