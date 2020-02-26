import argparse
import os
import torch
import torchvision as tv


def get_config():
    test_config = {
        'test_config_name': None,
        'out_path': None,
        'ngpu': 1,
        'dataset_configs': {
            'dataset_name': 'DepthDataset',
            'datalist_path': '/path/to/test1/list/',
            'data_columns': ['rgb_path', 'ir_path'],
            'target_columns': 'label',
            'transform_source': 'model_config',
            'test_process_config': {
                'metrics': {
                    'name': 'roc-curve',
                }
            },
            'nthreads': 8,
            'batch_size': 128,
        },

        'logger_config': {
            'logger_type': 'tensorboard',
            'show_metrics': {
                'name': 'tpr@fpr',
                'fpr': 0.01
            },
        }
    }

    ns_conf = argparse.Namespace()
    return ns_conf
