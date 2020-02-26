import torch.nn as nn
"""
TODO:
-[ ] make to and to_parallel for each attribute of class
"""


class Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        pass

    def predict(self, x):
        pass

    def to(self, device):
        for attribute, attr_value in self.__dict__['_modules'].items():
            if isinstance(attr_value, nn.Module):
                setattr(self.__dict__['_modules'], attribute, attr_value.to(device))
        return self

    def to_parallel(self, parallel_class):
        pass
