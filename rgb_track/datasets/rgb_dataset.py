from PIL import Image
from at_learner_core.datasets.df2dict_dataset import Df2DictDataset
from collections import OrderedDict
import torch


def img_loader(path, channels=3):
    img = Image.open(path)
    if channels == 1 and img.mode != 'L':
        img = img.convert('L')
    elif channels == 3 and img.mode != 'RGB':
        img = img.convert('RGB')
    return img


class RGBDataset(Df2DictDataset):
    def __init__(self, datalist_config, transforms):
        super().__init__(datalist_config, transforms)
        self.input_channels = 3

    def __getitem__(self, index):
        item_dict = OrderedDict()
        for column, column_name in self.data_columns:
            item_dict[column_name] = img_loader(self.df[column].values[index],
                                                channels=self.input_channels)
        if self.transforms is not None:
            item_dict = self.transforms(item_dict)

        for column, column_name in self.target_columns:
            item_dict[column_name] = torch.Tensor([self.df[column].values[index]])

        return item_dict
