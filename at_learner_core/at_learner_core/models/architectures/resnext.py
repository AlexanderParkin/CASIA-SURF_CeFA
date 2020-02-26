import torch
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore

iChannels = 64
bottleneck_ = None

pretrained_weights_dict = {
    'FaceNet': '/media3/a.parkin/codes/replayliveness/models/pretrained/resnext50_facenet.pth'
}

class ResNeXtBottleneck(nn.Module):
    def __init__(self, in_channels, stride, cardinality, widen_factor,
                 thin_mode=False, k=1, ch8_per_group=False):
        super(ResNeXtBottleneck, self).__init__()
        
        global iChannels
        nInputPlane = iChannels
        iChannels = in_channels * 4

        D = in_channels * widen_factor // 64
        C = cardinality

        if thin_mode:
            reduce_ch = int(D*C/2)
        else:
            reduce_ch = D*C

        groups = int(C*k)
        if ch8_per_group:
            scale = 8 / (reduce_ch // groups)
            if scale > 1:
                groups /= scale
            groups = int(groups)

        self.conv_reduce = nn.Conv2d(nInputPlane, reduce_ch, kernel_size=1, stride=1,
                                     padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(reduce_ch)
        self.conv_conv = nn.Conv2d(reduce_ch, D*C, kernel_size=3, stride=stride,
                                   padding=1, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(D*C)
        self.conv_expand = nn.Conv2d(D*C, in_channels * 4, kernel_size=1, stride=1,
                                     padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(in_channels * 4)

        self.shortcut = nn.Sequential()
        if nInputPlane != in_channels * 4:
            self.shortcut.add_module('shortcut_conv', nn.Conv2d(nInputPlane, in_channels * 4,
                                                                kernel_size=1, stride=stride,
                                                                padding=0, bias=False))
            self.shortcut.add_module('shortcut_bn', nn.BatchNorm2d(in_channels * 4))

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.relu(self.bn_reduce.forward(bottleneck), inplace=True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.relu(self.bn.forward(bottleneck), inplace=True)
        bottleneck = self.conv_expand.forward(bottleneck)
        bottleneck = self.bn_expand.forward(bottleneck)
        residual = self.shortcut.forward(x)
        return F.relu(residual + bottleneck, inplace=True)


class ResNeXt(nn.Module):
    def __init__(self, cardinality, depth, widen_factor, k=1, bottleneck_type='bottleneck',
                 version=1, split_output=False, split_size=1024,
                 ch8_per_group=True, pretrained=None):
        super(ResNeXt, self).__init__()

        print(' | ResNeXt-' + str(depth))

        cfg = {
            26: [[1, 2, 3, 2], 2048],
            35: [[1, 2, 5, 3], 2048],
            38: [[2, 2, 5, 3], 2048],
            50: [[3, 4, 6, 3], 2048],
            101: [[3, 4, 23, 3], 2048],
            152: [[3, 8, 36, 3], 2048],
        }

        thin_mode = (depth == 152)
        if thin_mode:
            print('THIN MODE', thin_mode)
        global bottleneck_
        if bottleneck_type == 'bottleneck':
            bottleneck_ = ResNeXtBottleneck
        else:
            raise Exception('Wrong bottleneck type!')

        cfg_depth, nfeat = cfg[depth]

        self.cardinality = cardinality
        self.widen_factor = widen_factor
        self.version = version
        self.split_output = split_output
        self.split_size = split_size
        global iChannels
        iChannels = round(64 * k)

        self.conv_1_3x3 = nn.Conv2d(3, round(32 * k), 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(round(32 * k))
        self.conv_2_3x3 = nn.Conv2d(round(32 * k), round(64 * k), 3, 1, 1, bias=False)
        self.bn_2 = nn.BatchNorm2d(round(64 * k))

        self.maxpool_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stages = [round(64 * k), round(128 * k), round(256 * k), round(512 * k)]

        self.stage_1 = self.block('stage_1', self.stages[0], cfg_depth[0], 1,
                                  thin_mode, k, ch8_per_group)
        self.stage_2 = self.block('stage_2', self.stages[1], cfg_depth[1], 2,
                                  thin_mode, k, ch8_per_group)
        self.stage_3 = self.block('stage_3', self.stages[2], cfg_depth[2], 2,
                                  thin_mode, k, ch8_per_group)
        self.stage_4 = self.block('stage_4', self.stages[3], cfg_depth[3], 2,
                                  thin_mode, k, ch8_per_group)

        if self.version == 1:
            print('ResneXt V1')
            self.last_channel = round(7 * 7 * 2048 * k)
        elif self.version == 2:
            print('ResneXt V2')
            self.last_channel  =round(2048 * k)
        
        self.qan_inp = False
        self.qan_split_inp = None

        if pretrained is not None:
            pretrained_weights = torch.load(pretrained_weights_dict[pretrained], 
                                            map_location='cpu')
            self.load_state_dict(pretrained_weights)

    def block(self, name, in_channels, block_depth, pool_stride=2,
              thin_mode=False, k=1, ch8_per_group=False):
        global bottleneck_

        block = nn.Sequential()
        for bottleneck in range(block_depth):
            name_ = '%s_bottleneck_%d' % (name, bottleneck)
            if bottleneck == 0:
                block.add_module(name_, bottleneck_(in_channels, pool_stride, self.cardinality,
                                                    self.widen_factor, thin_mode, k, ch8_per_group))
            else:
                block.add_module(name_,
                                 bottleneck_(in_channels, 1, self.cardinality, self.widen_factor,
                                             thin_mode, k, ch8_per_group))
        return block

    def forward(self, x):
        x = self.conv_1_3x3.forward(x)
        x = F.relu(self.bn_1.forward(x), inplace=True)
        x = self.conv_2_3x3.forward(x)
        x = F.relu(self.bn_2.forward(x), inplace=True)
        x = self.maxpool_1(x)
        x = self.stage_1.forward(x)
        x = self.stage_2.forward(x)
        x = self.stage_3.forward(x)        
        if self.qan_inp and self.qan_split_inp==3:
            x1 = F.avg_pool2d(x, 14, 1)
            xq = x1.view(x1.size(0), -1)
        x = self.stage_4.forward(x)
        if self.version == 1:
            x = x.view(x.size(0), -1)

        elif self.version == 2:
            x = F.adaptive_avg_pool2d(x, (1,1))            
            x = x.view(x.size(0), -1)

        return x
