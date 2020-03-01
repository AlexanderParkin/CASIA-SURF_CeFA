from PIL import Image
import pandas as pd
from .df2dict_dataset import Df2DictDataset


def rgb_loader(path):
    return Image.open(path).convert('RGB')


class FrameDataset(Df2DictDataset):

    def __init__(self, datalist_config, transforms):
        '''
        Dataset to get images by column
        :param datalist_config:
        :param transforms:
        '''
        self.datalist_config = datalist_config
        self.transforms = transforms
        self.df = self._read_list()

    def _create_index2class(self, class_column):
        self.index2class_dict = {}
        for idx, class_v in enumerate(self.df[class_column].values):
            self.index2class_dict[idx] = class_v

    def _get_class(self, index):
        return self.index2class_dict[index]

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
