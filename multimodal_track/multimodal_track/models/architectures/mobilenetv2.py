from torch.nn import init
import torch.nn as nn
import math
import torch.nn.functional as F




class mobilenetv2(nn.Module):
    def __init__(self, width_mult=1.4, descriptor_size=256):
        super(mobilenetv2, self).__init__()
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 1], 
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        self.pool_size = 7
        input_channel = int(32 * width_mult // 8 * 8) 
        self.last_channel = int(1280 * width_mult // 8 * 8)  
       
        self.features = [conv_bn(3, input_channel, 2)] 
        
        # building inverted residual blocks
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult // 8 * 8) 
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, t))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, t))
                input_channel = output_channel
                
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        
        self.features = nn.Sequential(*self.features)

        self.avg_pool = nn.AvgPool2d(self.pool_size,count_include_pad=False)

        # building classifier
        self.classifier = nn.Sequential(nn.Linear(self.last_channel, descriptor_size))
        #self.final_bn = nn.BatchNorm1d(descriptor_size)
        
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        return x


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                init.constant_(m.bias, 0)




def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )



class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup
        self.expand_inp = (inp * expand_ratio) // 8 * 8
        self.conv = nn.Sequential(
            
            # pw
            nn.Conv2d(inp, self.expand_inp, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.expand_inp),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(self.expand_inp, self.expand_inp, 3, stride, 1, groups=self.expand_inp, bias=False),
            nn.BatchNorm2d(self.expand_inp),
            nn.ReLU(inplace=True),
            # pw-linear
            nn.Conv2d(self.expand_inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)