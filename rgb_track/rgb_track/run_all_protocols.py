import argparse
import os
import torch
import torchvision as tv

from at_learner_core.utils import transforms
from at_learner_core.configs import dict_to_namespace
from at_learner_core.utils import transforms as transforms

from at_learner_core.utils import joint_transforms as j_transforms
from at_learner_core.utils import sequence_transforms as s_transforms

def get_config(protocol_name):
    config = {
        'head_config': {
            'task_name': 'rgb_liveness',
            'exp_name': f'simplenet_pretrained',
            'protocol_name': protocol_name,
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
                'protocol_name': protocol_name,
                'data_columns': [('rgb_path', 'data')],
                'target_columns': ('label', 'target'),
                'group_column': 'video_id',
                'sampler_config': {
                    'name': 'NumElements',
                    'class_probability': [0.5, 0.5],
                    'class_column': 'label',
                    'num_elem_per_epoch': 1.0,
                },
                'sequence_transforms': tv.transforms.Compose([
                                            tv.transforms.RandomApply([
                                                s_transforms.DuplicateElements(4, False, ['data'], 'target', 0, False)
                                            ], p=0.3),
                                            s_transforms.LinspaceTransform(16, max_start_index=0.5, key_list=['data']),
                                        ]),
                'transforms': transforms.Transform4EachKey(
                                tv.transforms.Compose([
                                    j_transforms.Compose([
                                        transforms.RemoveBlackBorders(),
                                        transforms.SquarePad(),
                                        j_transforms.Resize((112, 112)),
                                        # j_transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                                        j_transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.MeanSubtraction(),
                                        j_transforms.ToTensor(),
                                        j_transforms.Normalize(mean=[0.5],
                                                               std=[0.5])
                                    ]),
                                    transforms.StackTensors()
                                ]),
                                key_list=['data'])
            },
            'testlist_configs': {
                'dataset_name': 'VideoDataset',
                'datalist_path': '/ssd/a.parkin/media/CASIA-SURF_CeFA/dev2_list.txt',
                'protocol_name': protocol_name,
                'data_columns': [('rgb_path', 'data')],
                'target_columns': ('label', 'target'),
                'group_column': 'video_id',
                'sequence_transforms': tv.transforms.Compose([
                    s_transforms.LinspaceTransform(16, key_list=['data'])
                ]),
                'transforms': transforms.Transform4EachKey(
                                tv.transforms.Compose([
                                    j_transforms.Compose([
                                        transforms.RemoveBlackBorders(),
                                        transforms.SquarePad(),
                                        j_transforms.Resize((112, 112)),
                                        transforms.MeanSubtraction(),
                                        j_transforms.ToTensor(),
                                        j_transforms.Normalize(mean=[0.5],
                                                               std=[0.5])
                                    ]),
                                transforms.StackTensors(),
                                ]),
                                key_list=['data'])
                }
        },

        'train_process_config': {
            'nthreads': 8,
            'ngpu': 1,
            'batchsize': 8,
            'nepochs': 10,
            'resume': None,
            'optimizer_config': {
                'name': 'Adam',
                'lr_config': {
                    'lr_type': 'StepLR',
                    'lr': 0.00001,
                    'lr_decay_period': 20,
                    'lr_decay_lvl': 0.5,
                },
                'weight_decay': 0.0001,
            },
        },

        'test_process_config': {
            'run_frequency': 2,
            'metric': {
                'name': 'acer',
                'target_column': 'target',
            }
        },

        'wrapper_config': {
            'wrapper_name': 'RGBVideoWrapper',
            'backbone': 'simplenet112',
            'nclasses': 1,
            'loss': 'BCE',
            'loss_config': {
                'gamma': 2.0,
                'alpha': 0.25,
            },
            'pretrained': '/media3/a.parkin/codes/CASIA_CeFA/CASIA_CeFA_challenge/rgb_track/experiments/rgb_liveness/pretrain_simplenet_recognition_exp/checkpoints/model_99.pth',
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
 #               {'logger_type': 'tensorboard',
 #                'log_batch_interval': 10,
 #                'show_metrics': {
 #                    'name': 'tpr@fpr',
 #                    'fpr': 0.01
 #                }}
            ]

        },
        'manual_seed': 42,
        'resume': None,
    }

    ns_conf = argparse.Namespace()
    dict_to_namespace(ns_conf, config)
    return ns_conf


def run_configs_creating(args):
    config_file_path_arr = []
    for protocol_name in ['protocol_4_1', 'protocol_4_2', 'protocol_4_3']:
        configs = get_config(protocol_name)
        out_path = os.path.join(args.savepath,
                                configs.head_config.task_name,
                                configs.head_config.exp_name,
                                configs.head_config.protocol_name)

        os.makedirs(out_path, exist_ok=True)
        if configs.checkpoint_config.out_path is None:
            configs.checkpoint_config.out_path = out_path
            config_file_path = os.path.join(out_path,
                                            configs.head_config.task_name + '_' + \
                                            configs.head_config.exp_name + '_' + \
                                            configs.head_config.protocol_name + '.config')
            torch.save(configs, config_file_path)
            config_file_path_arr.append(config_file_path)

    experiments_dir = '/'.join(config_file_path.split('/')[:-2])
    print('Config files were saved to ' + experiments_dir + '/')
    return experiments_dir, config_file_path_arr


def run_training_process(args, config_file_path_arr):
    import random
    import os

    import numpy as np
    import torch
    import torch.backends.cudnn as cudnn
    from rgb_trainer import RGBRunner

    for idx, protocol_name in enumerate(['protocol_4_1', 'protocol_4_2', 'protocol_4_3']):
        config = torch.load(config_file_path_arr[idx]) if not args.skip_configs else get_config(protocol_name)

        print('===Options==')
        d = vars(config)
        for k in d.keys():
            print(k, ':', d[k])

        """ Fix random seed """
        random.seed(config.manual_seed)
        np.random.seed(config.manual_seed)
        torch.manual_seed(config.manual_seed)
        torch.cuda.manual_seed_all(config.manual_seed)
        cudnn.benchmark = True

        # Create working directories
        os.makedirs(config.checkpoint_config.out_path, exist_ok=True)
        os.makedirs(os.path.join(config.checkpoint_config.out_path, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(config.checkpoint_config.out_path, 'log_files'), exist_ok=True)
        print('Directory {} was successfully created.'.format(config.checkpoint_config.out_path))

        # Training
        runner = RGBRunner(config)
        runner.train()
        del runner
        torch.cuda.empty_cache()
        print('\n' + ''.join(['='] * 10) + f'\n{protocol_name} training is done\n' + ''.join(['='] * 10) + '\n')


def run_summary(args, experiments_dir):
    from glob import glob
    import numpy as np
    full_meters = {'thrs': [], 'acer': [], 'apcer': [], 'bpcer': []}
    for protocol_name in ['protocol_4_1', 'protocol_4_2', 'protocol_4_3']:
        best_models = glob(os.path.join(experiments_dir, protocol_name, 'checkpoints', 'best_*.pth'))
        sorted(best_models)
        best_checkpoint = torch.load(best_models[-1], map_location='cpu')
        test_info = best_checkpoint['test_info'].metric.get_all_metrics(200, 0.5)
        print(f'{protocol_name}')
        print(f'epoch: {best_checkpoint["epoch"]}')
        for thr, metrics in test_info.items():
            acer, apcer, bpcer = metrics["acer"], metrics["apcer"], metrics["bpcer"]
            print(f'thr: {thr:.4f}, ACER: {acer:.4f}, APCER: {apcer:.4f}, BPCER: {bpcer:.4f}')
            if thr != 0.5:
                full_meters['thrs'].append(thr)
                full_meters['acer'].append(acer)
                full_meters['apcer'].append(apcer)
                full_meters['bpcer'].append(bpcer)

    print('=== Summary ===')
    print (experiments_dir)
    avg_acer = np.mean(full_meters['acer'])
    avg_apcer = np.mean(full_meters['apcer'])
    avg_bpcer = np.mean(full_meters['bpcer'])

    std_acer = np.std(full_meters['acer'])
    std_apcer = np.std(full_meters['apcer'])
    std_bpcer = np.std(full_meters['bpcer'])
    print(f'Avg ACER: {avg_acer:.4f}({std_acer:.4f})')
    print(f'Avg APCER: {avg_apcer:.4f}({std_apcer:.4f})')
    print(f'Avg BPCER {avg_bpcer:.4f}({std_bpcer:.4f})')


def main():
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--savepath',
                        type=str,
                        default='experiments/',
                        help='Path to save options')
    parser.add_argument('--skip_configs', '-c',
                        action='store_true',
                        help='Skip configs creating if exist')
    parser.add_argument('--experiments_dir', '-d',
                        type=str,
                        help='Path to experiments dir, is used if you have skip_config flag')
    parser.add_argument('--skip_train', '-t',
                        action='store_true',
                        help='Skip training models if exist')
    parser.add_argument('--skip_summary', '-s',
                        action='store_true',
                        help='Skip summary string if exist')
    args = parser.parse_args()

    # RUN All protocols creating
    if not args.skip_configs:
        experiments_dir, config_file_path_arr = run_configs_creating(args)
    else:
        experiments_dir = args.experiments_dir

    # RUN ALL training
    if not args.skip_train:
        run_training_process(args, config_file_path_arr)

    # Print All three metrics and AVG metric
    if not args.skip_summary:
        run_summary(args, experiments_dir)


if __name__ == '__main__':
    main()









