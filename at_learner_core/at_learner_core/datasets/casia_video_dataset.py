from PIL import Image
import pandas as pd
import numpy as np
import torch.utils.data
from collections import OrderedDict
import torch


def rgb_loader(path):
    return Image.open(path).convert('RGB')


class VideoDataset(torch.utils.data.Dataset):

    def __init__(self, datalist_config, transforms):
        '''
        Dataset to generate image sequences by group_column
        :param datalist_config:
        :param transforms:

        video_len, group_column, random_start
        '''
        self.datalist_config = datalist_config
        self.group_column = self.datalist_config.group_column
        self.transforms = transforms
        self.seq_transforms = self.datalist_config.sequence_transforms
        self.df = self._read_list()
        self._init_groups()

    def _init_groups(self):
        group_ids = self.df[self.group_column].unique()
        for batch_idx, group_id in enumerate(group_ids):
            subdf = self.df[self.df[self.group_column] == group_id]
            self.df.loc[subdf.index.values, 'batch_idx'] = batch_idx

    def __getitem__(self, index):
        item_dict = OrderedDict()
        video_subdf = self.df[self.df.batch_idx == index]

        for column, column_name in self.data_columns:
            all_images = []
            for img_path in video_subdf[column].values:
                all_images.append(img_path)
            item_dict[column_name] = all_images

        for column, column_name in self.target_columns:
            item_dict[column_name] = video_subdf[column].iloc[0]

        if self.seq_transforms is not None:
            item_dict = self.seq_transforms(item_dict)

        for _, column_name in self.data_columns:
            item_dict[column_name] = [rgb_loader(x) for x in item_dict[column_name]]

        for _, column_name in self.target_columns:
            item_dict[column_name] = torch.Tensor([item_dict[column_name]])

        if self.transforms is not None:
            item_dict = self.transforms(item_dict)

        return item_dict

    def __len__(self):
        return len(self.df[self.group_column].unique())

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
        needed_columns.append(self.group_column)
        data_df = data_df[needed_columns]
        return data_df
