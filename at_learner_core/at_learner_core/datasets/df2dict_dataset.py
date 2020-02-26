from PIL import Image
import pandas as pd
import torch.utils.data
from collections import OrderedDict
import torch


def rgb_loader(path):
    return Image.open(path).convert('RGB')


class Df2DictDataset(torch.utils.data.Dataset):
    def __init__(self, datalist_config, transforms):
        self.datalist_config = datalist_config
        self.transforms = transforms
        self.df = self._read_list()

    def __getitem__(self, index):
        item_dict = OrderedDict()
        for column, column_name in self.data_columns:
            item_dict[column_name] = rgb_loader(self.df[column].values[index])
        if self.transforms is not None:
            item_dict = self.transforms(item_dict)

        for column, column_name in self.target_columns:
            item_dict[column_name] = torch.Tensor([self.df[column].values[index]])

        return item_dict

    def __len__(self):
        return len(self.df)

    def _read_list(self):
        data_df = pd.read_csv(self.datalist_config.datalist_path)
        data_df = data_df[data_df[self.datalist_config.protocol_name]]

        if isinstance(self.datalist_config.data_columns, list):
            self.data_columns = self.datalist_config.data_columns
        elif isinstance(self.datalist_config.data_columns, tuple):
            self.data_columns = [self.datalist_config.data_columns]
        elif isinstance(self.datalist_config.data_columns, str):
            self.data_columns = [(self.datalist_config.data_columns,
                                  self.datalist_config.data_columns)]
        else:
            raise Exception('Unknown columns types in dataset')

        if isinstance(self.datalist_config.target_columns, list):
            self.target_columns = self.datalist_config.target_columns
        elif isinstance(self.datalist_config.target_columns, tuple):
            self.target_columns = [self.datalist_config.target_columns]
        elif isinstance(self.datalist_config.target_columns, str):
            self.target_columns = [(self.datalist_config.target_columns,
                                    self.datalist_config.target_columns)]
        else:
            raise Exception('Unknown columns types in dataset')

        needed_columns = [x[0] for x in self.data_columns]
        needed_columns = needed_columns + [x[0] for x in self.target_columns]
        needed_columns = list(set(needed_columns))
        data_df = data_df[needed_columns]
        return data_df
