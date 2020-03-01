import torch
import torch.utils.data
from .imagelist_dataset import ImageListDataset
from .df2dict_dataset import Df2DictDataset
from .casia_video_dataset import VideoDataset
from .casia_frame_dataset import FrameDataset


class DatasetManager(object):
    def __init__(self):
        pass

    @staticmethod
    def _get_dataset(dataset_config):
        if dataset_config.dataset_name == 'ImageListDataset':
            return ImageListDataset(dataset_config, dataset_config.transforms)
        elif dataset_config.dataset_name == 'Df2DictDataset':
            return Df2DictDataset(dataset_config, dataset_config.transforms)
        elif dataset_config.dataset_name == 'VideoDataset':
            return VideoDataset(dataset_config, dataset_config.transforms)
        elif dataset_config.dataset_name == 'FrameDataset':
            return FrameDataset(dataset_config, dataset_config.transforms)
        else:
            raise Exception('Unknown dataset type')

    @staticmethod
    def _get_sampler(sampler_config, dataset):
        if sampler_config.name == 'ClassProbability':
            dataset._create_index2class(sampler_config.class_column)
            needed_probs = sampler_config.class_probability
            class_probs = [0]*len(needed_probs)
            for index in range(len(dataset)):
                class_probs[dataset._get_class(index)] += 1
            class_probs = [x/sum(class_probs) for x in class_probs]
            weights = [x/y for x, y in zip(needed_probs, class_probs)]
            weights = torch.Tensor(weights)

            if sampler_config.num_elem_per_epoch is None:
                num_elements = len(dataset)
            elif type(sampler_config.num_elem_per_epoch) == int:
                num_elements = sampler_config.num_elem_per_epoch
            elif type(sampler_config.num_elem_per_epoch) == float:
                num_elements = int(sampler_config.num_elem_per_epoch * len(dataset))

            sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_elements)
        elif sampler_config.name == 'NumElements':
            if type(sampler_config.num_elem_per_epoch) == int:
                num_elements = sampler_config.num_elem_per_epoch
            elif type(sampler_config.num_elem_per_epoch) == float:
                num_elements = int(sampler_config.num_elem_per_epoch * len(dataset))

            if num_elements > len(dataset):
                replacement = True
            else:
                replacement = False
                num_elements = None
            sampler = torch.utils.data.sampler.RandomSampler(dataset,
                                                             replacement,
                                                             num_elements)
        return sampler

    @staticmethod
    def get_dataloader(dataset_config, train_process_config, shuffle=True):
        dataset = DatasetManager._get_dataset(dataset_config)
        if hasattr(dataset_config, 'sampler_config'):
            sampler = DatasetManager._get_sampler(dataset_config.sampler_config, dataset)
            shuffle = False
        else:
            sampler = None
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=train_process_config.batchsize,
                                                  shuffle=shuffle,
                                                  num_workers=train_process_config.nthreads,
                                                  sampler=sampler)
        return data_loader

    @staticmethod
    def get_dataloader_by_args(dataset, batch_size, num_workers=8, shuffle=False, sampler=None):
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers,
                                                  sampler=sampler)
        return data_loader
