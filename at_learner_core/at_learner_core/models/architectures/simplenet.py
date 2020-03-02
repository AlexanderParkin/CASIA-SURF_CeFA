'''SimpleNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F
import torch

cfg = {
    'v1': [8, 16, 32,  64],
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
    def __init__(self, pretrained, in_channels = 3):
        super(SimpleNet112, self).__init__()
        self.in_channels = in_channels
        self.features = self._make_layers(cfg['v3'])

        if pretrained:
            pretrained_weights = torch.load(pretrained,
                                            map_location='cpu')
            pretrained_weights = pretrained_weights['state_dict']
            new_weights = self.state_dict()
            for k, v in new_weights.items():
                new_weights[k] = pretrained_weights['backbone.' + k]
            self.load_state_dict(new_weights)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.in_channels
        for x in cfg:
            layers.append(SimpleBlock(in_channels, x, True, stride=1, merged=False))
            in_channels = x
        layers.append(nn.Conv2d(in_channels, 2 * in_channels, kernel_size=5))
        return nn.Sequential(*layers)
    
def get_simplenet_block(block_number, inplanes=None, planes=None):
    planes_arr = [16, 32, 64, 128]

    if block_number == 0:
        inplanes = 3 if inplanes is None else inplanes
        planes = 16 if planes is None else planes

        block = SimpleBlock(inplanes, planes, True)
    elif 1 <= block_number <= 3:
        if block_number == 1:
            stride = 1
        else:
            stride = 1
        inplanes = planes_arr[block_number - 1] if inplanes is None else inplanes
        planes = planes_arr[block_number] if planes is None else planes
        block = SimpleBlock(inplanes, planes, True)
    return block
