import argparse
import random
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from rgb_trainer import RGBRunner
from configs import get_config


def main():
    # Load options
    parser = argparse.ArgumentParser(description='rgb liveness')
    parser.add_argument('--config',
                        type=str,
                        help='Path to config .config file. Leave blank if loading from configs.py')
    
    arg_conf = parser.parse_args()
    config = torch.load(arg_conf.config) if arg_conf.config else get_config()
    
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


if __name__ == '__main__':
    main()


