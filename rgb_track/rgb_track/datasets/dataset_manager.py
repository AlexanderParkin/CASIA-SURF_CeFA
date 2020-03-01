import torch.utils.data
from at_learner_core import datasets
from .rgb_dataset import RGBDataset


class DatasetManager(object):
    def __init__(self):
        super().__init__()
        pass

    @staticmethod
    def _get_dataset(dataset_config):
        if dataset_config.dataset_name == 'Df2DictDataset':
            return datasets.df2dict_dataset.Df2DictDataset(dataset_config, dataset_config.transforms)
        elif dataset_config.dataset_name == 'RGBDataset':
            return RGBDataset(dataset_config, dataset_config.transforms)
        else:
            raise Exception('Unknown dataset type')

    @staticmethod
    def get_dataloader(dataset_config, train_process_config, shuffle=True):
        sampler = None
        dataset = DatasetManager._get_dataset(dataset_config)
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=train_process_config.batchsize,
                                                  shuffle=shuffle,
                                                  num_workers=train_process_config.nthreads,
                                                  sampler=sampler)
        return data_loader
