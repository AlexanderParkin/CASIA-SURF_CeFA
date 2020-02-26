from torch.nn import init
import torch.nn as nn
import math
import torch
import collections


class MLP(nn.Module):
    def __init__(self, in_size = 512, out_size = 1, layers = [512], dropout_p = 0.2):
        super(MLP, self).__init__()
        in_s = in_size
        seq_layers = collections.OrderedDict()
        for i in range(len(layers)+1):
            if i > 0:
                seq_layers['prelu'+str(i)] = nn.PReLU(in_s)
            seq_layers['dropout'+str(i)] = nn.Dropout(p=dropout_p)
            out_s = out_size
            if i < len(layers):
                out_s = layers[i]
            seq_layers['fc'+str(i)] = nn.Linear(in_s, out_s)
            in_s = out_s
        self.layers = nn.Sequential(seq_layers)

    def forward(self, x, **kwargs):
        return self.layers(x)