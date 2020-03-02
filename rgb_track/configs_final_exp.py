import argparse
import os
import torch
import torchvision as tv
from at_learner_core.utils import transforms
from at_learner_core.configs import dict_to_namespace
from at_learner_core.utils import transforms as transforms

from at_learner_core.utils import joint_transforms as j_transforms
from at_learner_core.utils import sequence_transforms as s_transforms
from PIL import Image

L = 16
image_size = 112
modality_list = ['stat_r1000', 'stat_r1']
of_modality_list = ['optical_flow', 'optical_flow_start']

test_seq_transform = tv.transforms.Compose([
    s_transforms.LinspaceTransform(L, key_list=['data']),
])

train_seq_transform = tv.transforms.Compose([
    tv.transforms.RandomApply([
        s_transforms.DuplicateElements(1, False, ['data'], 'target', 1, True)
    ], p=0.5),
    s_transforms.LinspaceTransform(L, key_list=['data'], max_start_index=0),
])

preprocess_transform = transforms.Transform4EachElement([
    transforms.RemoveBlackBorders(),
    transforms.SquarePad(),
    tv.transforms.Resize(image_size),
])

postprocess_transform = tv.transforms.Compose([
    transforms.CreateNewItem(transforms.RankPooling(C=1000), 'data', 'stat_r1000'),
    transforms.CreateNewItem(transforms.RankPooling(C=1), 'data', 'stat_r1'),

    transforms.DeleteKeys(['data']),

    transforms.Transform4EachKey([
        transforms.Transform4EachElement([
            # tv.transforms.Resize(112),
            tv.transforms.ToTensor(),
        ]),
        transforms.StackTensors(squeeze=True),
        tv.transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5])
    ], key_list=of_modality_list),

    transforms.Transform4EachKey([
        tv.transforms.Resize(112),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.5], std=[0.5])],
        key_list=modality_list)
])

train_image_transform = tv.transforms.Compose([
    transforms.Transform4EachKey([
        preprocess_transform,
        tv.transforms.RandomApply([j_transforms.ColorJitter(0.2, 0.2, 0.2, 0.2)], p=0.5),
    ], key_list=['data']),

    transforms.Transform4EachKey([
        tv.transforms.RandomApply([
            transforms.Transform4EachElement([
                tv.transforms.RandomApply([
                    tv.transforms.RandomRotation(5)
                ], p=0.5)
            ])], p=0.5),
        tv.transforms.RandomApply([
            transforms.Transform4EachElement([
                tv.transforms.RandomApply([
                    tv.transforms.RandomCrop(image_size, padding=5, pad_if_needed=True)
                ], p=0.5)
            ])
        ], p=0.5),
        tv.transforms.RandomApply([
            transforms.Transform4EachElement([
                tv.transforms.RandomApply([
                    tv.transforms.ColorJitter(0.05, 0.05, 0.05, 0.00)
                ], p=0.5)
            ])
        ], p=0.5),
    ], key_list=['data']),
    transforms.CreateNewItem(transforms.LiuOpticalFlowTransform((0, 4), (L - 4, L)), 'data', 'optical_flow'),
    transforms.CreateNewItem(transforms.LiuOpticalFlowTransform((0, 1), (2, 4)), 'data', 'optical_flow_start'),
    postprocess_transform

])

test_image_transform = tv.transforms.Compose([
    transforms.Transform4EachKey([
        preprocess_transform,
    ], key_list=['data']),
    transforms.CreateNewItem(transforms.LiuOpticalFlowTransform((0, 0), (L - 1, L)), 'data', 'optical_flow'),
    transforms.CreateNewItem(transforms.LiuOpticalFlowTransform(0, 1), 'data', 'optical_flow_start'),

    postprocess_transform
])


def get_config():
    config = {
        'head_config': {
            'task_name': 'rgb_track',
            'exp_name': 'exp1_protocol41',
            'text_comment': '',
        },

        'checkpoint_config': {
            'out_path': None,
            'save_frequency': 1,
        },

        'datalist_config': {
            'trainlist_config': {
                'dataset_name': 'VideoDataset',
                'datalist_path': '/ssd/a.parkin/media/CASIA-SURF_CeFA/train_list.txt',
                'protocol_name': 'protocol_4_1',
                'data_columns': [('rgb_path', 'data')],
                'target_columns': ('label', 'target'),
                'group_column': 'video_id',
                'sampler_config': {
                    'name': 'NumElements',
                    'class_column': 'label',
                    'num_elem_per_epoch': 20.0,
                },
                'sequence_transforms': train_seq_transform,
                'transforms': train_image_transform,

            },
            'testlist_configs': {
                'dataset_name': 'VideoDataset',
                'datalist_path': '/ssd/a.parkin/media/CASIA-SURF_CeFA/dev_list.txt',
                'protocol_name': 'protocol_4_1',
                'data_columns': [('rgb_path', 'data')],
                'target_columns': ('label', 'target'),
                'group_column': 'video_id',
                'sequence_transforms': test_seq_transform,
                'transforms': test_image_transform,
            }
        },

        'train_process_config': {
            'nthreads': 8,
            'ngpu': 1,
            'batchsize': 32,
            'nepochs': 10,
            'resume': None,
            'optimizer_config': {
                'name': 'Adam',
                'lr_config': {
                    'lr_type': 'StepLR',
                    'lr': 0.0001,
                    'lr_decay_period': 5,
                    'lr_decay_lvl': 0.5,
                },
                'weight_decay': 1e-05,
            },
        },

        'test_process_config': {
            'run_frequency': 1,
            'metric': {
                'name': 'acer',
                'target_column': 'target',
            }
        },

        'wrapper_config': {
            'wrapper_name': 'MultiModalWrapper',
            'input_modalities': modality_list + of_modality_list,
            'backbone': 'simplenet112',
            'nclasses': 1,
            'loss': 'BCE',
            'pretrained': None,
            'freeze_weights': None,
        },

        'logger_config': {
            'logger_type': 'log_combiner',
            'loggers': [
                {'logger_type': 'terminal',
                 'log_batch_interval': 5,
                 'show_metrics': {
                     'name': 'acer',
                     'fpr': 0.01,
                 }},
            ]

        },
        'manual_seed': 42,
        'resume': None,
    }

    ns_conf = argparse.Namespace()
    dict_to_namespace(ns_conf, config)
    return ns_conf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--savepath',
                        type=str,
                        default='experiments/',
                        help='Path to save options')
    args = parser.parse_args()
    configs = get_config()
    out_path = os.path.join(args.savepath,
                            configs.head_config.task_name,
                            configs.head_config.exp_name)
    os.makedirs(out_path, exist_ok=True)
    if configs.checkpoint_config.out_path is None:
        configs.checkpoint_config.out_path = out_path
    filename = os.path.join(out_path,
                            configs.head_config.task_name + '_' + configs.head_config.exp_name + '.config')

    torch.save(configs, filename)
    print('Options file was saved to ' + filename)