import argparse
import os
import torch
import torchvision as tv
from at_learner_core.utils import transforms
from at_learner_core.configs import dict_to_namespace
from at_learner_core.utils import transforms as transforms

from at_learner_core.utils import joint_transforms as j_transforms
from at_learner_core.utils import sequence_transforms as s_transforms

def get_config():
    config = {
        'head_config': {
            'task_name': 'rgb_liveness',
            'exp_name': 'test_exp',
            'text_comment': '',
        },

        'checkpoint_config': {
            'out_path': None,
            'save_frequency': 1,
        },

        'datalist_config': {
            'trainlist_config': {
                'dataset_name': 'VideoDataset',
                'datalist_path': '/path/to/train/list/',
                'protocol_name': 'protocol_4_2',
                'data_columns': [('rgb_path', 'data')],
                'target_columns': ('label', 'target'),
                'group_column': 'video_id',
                'sequence_transforms': tv.transforms.Compose([
                                            tv.transforms.RandomApply([
                                                s_transforms.ShuffleTransform(key_list=['data'])
                                            ], p=0.5),
                                            s_transforms.LinspaceTransform(16, key_list=['data']),
                                        ]),
                'transforms': transforms.Transform4EachKey(
                                tv.transforms.Compose([
                                transforms.Transform4EachElement(
                                    tv.transforms.Compose([
                                        tv.transforms.Resize((224, 224)),
                                        tv.transforms.RandomHorizontalFlip(p=0.5),
                                        tv.transforms.RandomRotation(degrees=(-15, 15),
                                                                     resample=False,
                                                                     expand=False),
                                        tv.transforms.CenterCrop(size=(200, 200)),
                                        tv.transforms.RandomResizedCrop(size=(160, 160)),
                                        tv.transforms.Resize((112, 112)),
                                        tv.transforms.ToTensor(),
                                        tv.transforms.Normalize(mean=[0.5],
                                                                std=[0.5])
                                    ])
                                ),
                                transforms.StackTensors()
                                ]),
                                key_list=['data'])
            },
            'testlist_configs': {
                'dataset_name': 'VideoDataset',
                'datalist_path': '/path/to/dev/list/',
                'protocol_name': 'protocol_4_2',
                'data_columns': [('rgb_path', 'data')],
                'target_columns': ('label', 'target'),
                'group_column': 'id',
                'sequence_transforms': tv.transforms.Compose([
                    s_transforms.LinspaceTransform(16, key_list=['data'])
                ]),
                'transforms': transforms.Transform4EachKey(
                                tv.transforms.Compose([
                                transforms.Transform4EachElement(
                                    tv.transforms.Compose([
                                        tv.transforms.Resize((224, 224)),
                                        tv.transforms.CenterCrop(160),
                                        tv.transforms.Resize((112, 112)),
                                        tv.transforms.ToTensor(),
                                        tv.transforms.Normalize(mean=[0.5],
                                                                std=[0.5])
                                    ])
                                ),
                                transforms.StackTensors(),
                                ]),
                                key_list=['data'])
                }
        },

        'train_process_config': {
            'nthreads': 8,
            'ngpu': 1,
            'batchsize': 32,
            'nepochs': 100,
            'resume': None,
            'optimizer_config': {
                'name': 'Adam',
                'lr_config': {
                    'lr_type': 'StepLR',
                    'lr': 0.001,
                    #'eta_min': 1e-6,
                    #'t_max': 100,
                    'lr_decay_period': 20,
                    'lr_decay_lvl': 0.5,
                },
                'weight_decay': 0.001,
            },
        },

        'test_process_config': {
            'metric': {
                'name': 'acer',
                'target_column': 'target',
            }
        },

        'wrapper_config': {
            'wrapper_name': 'RGBVideoWrapper',
            'backbone': 'MobilenetV2',
            'nclasses': 1,
            'loss': 'focal_loss',
            'loss_config': {
                'gamma': 2.0,
                'alpha': 0.25,
            },
            'pretrained': None,
        },

        'logger_config': {
            'logger_type': 'log_combiner',
            'loggers': [
                {'logger_type': 'terminal',
                 'log_batch_interval': 5,
                 'show_metrics': {
                     'name': 'acer',
                     'fpr': 0.01
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