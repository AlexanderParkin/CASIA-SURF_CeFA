import argparse
import argparse
import os
import torch
import torchvision as tv


def get_config():    
    config = {
        'head_config': {
            'task_name': 'Test task',
            'exp_name': 'Test exp 0',
        },

        'checkpoint_config': {
            'out_path': None,
            'save_frequency': 1,
        },

        'datalist_config': {
            'trainlist_config': {
                'dataset_name': 'ImageListDataset',
                'datalist_path': '/path/to/train/list/',
                'data_columns': ['rgb_path', 'ir_path'],
                'target_columns': 'label',
                'transforms': tv.transforms.Compose([
                                tv.transforms.CenterCrop(224),
                                tv.transforms.RandomResizedCrop(size=112,
                                                                scale=(0.8, 1.0),
                                                                ratio=(0.9, 1.1111)),
                                tv.transforms.RandomHorizontalFlip(),
                                tv.transforms.ToTensor(),
                                tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])])
            },
            'testlist_configs': {
                'datalist_path': '/path/to/test1/list/',
                'transforms': tv.transforms.Compose([
                                tv.transforms.CenterCrop(224),
                                tv.transforms.Resize(112),
                                tv.transforms.ToTensor(),
                                tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])])

                }
        },

        'train_process_config': {
            'nthreads': 16,
            'ngpu': 1,
            'batchsize': 256,
            'nepochs': 30,
            'resume': None,
            'optimizer_config': {
                'name': 'Adam',
                'lr_config': {
                    'lr_type': 'StepLR',
                    'lr': 0.001,
                    'lr_decay_period': 20,
                    'lr_decay_lvl': 0.5,
                },
            },
        },

        'test_process_config': {
            'metric': {
                'name': 'tpr@fpr'
            }
        },

        'wrapper_config': {
            'wrapper_name': 'SimpleClassifierWrapper',
            'backbone': 'MobilenetV2',
            'nclasses': 1,
            'loss': 'BCE',
            'pretrained': None,
        },

        'logger_config': {
            'logger_type': 'terminal',
            'log_batch_interval': 5,
            'show_metrics': {
                'name': 'tpr@fpr',
                'fpr': 0.01
            },
        },

        'manual_seed': 42,
        'resume': None,
    }

    ns_conf = argparse.Namespace()
    dict_to_namespace(ns_conf, config)
    return ns_conf


def list_to_namespace_list(l):
    new_list = []
    for list_v in l:
        if isinstance(list_v, dict):
            list_ns = argparse.Namespace()
            dict_to_namespace(list_ns, list_v)
            new_list.append(list_ns)
        else:
            new_list.append(list_v)
    return new_list


def dict_to_namespace(ns, d):
    for k, v in d.items():
        if isinstance(v, dict):
            leaf_ns = argparse.Namespace()
            ns.__dict__[k] = leaf_ns
            dict_to_namespace(leaf_ns, v)
        elif isinstance(v, list):
            ns.__dict__[k] = list_to_namespace_list(v)
        else:
            ns.__dict__[k] = v


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--savepath', 
                        type=str, 
                        default='test_configs/', 
                        help='Path to save options')
    args = parser.parse_args()
    configs = get_config()
    out_path = os.path.join(args.savepath,
                            configs.head_config.task_name,
                            configs.head_config.exp_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if configs.checkpoint_config.out_path is None:
        configs.checkpoint_config.out_path = out_path
    filename = os.path.join(out_path,
                            configs.head_config.task_name + '_' + configs.head_config.exp_name + '.config')

    torch.save(configs, filename)
    print('Options file was saved to ' + filename)
