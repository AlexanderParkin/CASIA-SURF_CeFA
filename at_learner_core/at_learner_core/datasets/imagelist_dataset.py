from PIL import Image
import pandas as pd
import torch.utils.data


def rgb_loader(path):
    return Image.open(path)


class ImageListDataset(torch.utils.data.Dataset):
    def __init__(self, datalist_config, transform):
        self.datalist_config = datalist_config
        self.transform = transform
        self.df = self._read_list()
        self.loader = rgb_loader

    def __getitem__(self, index):
        item_data = self.loader(self.df.path.values[index])
        item_label = self.df.label.values[index]

        if self.transform is not None:
            item_data = self.transform(item_data)
        return item_data, item_label

    def __len__(self):
        return len(self.df.label.values)

    def _read_list(self):
        data_df = pd.read_csv(self.datalist_config.datalist_path)
        if hasattr(self.datalist_config, 'used_columns'):
            data_df = self.df[self.datalist_config.used_columns]
        return data_df
