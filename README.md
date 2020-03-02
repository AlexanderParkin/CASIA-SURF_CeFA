# Chalearn CeFA Face Anti-Spoofing challenge

## First step. 
Install at_learner_core
```
cd /path/to/new/pip/environment
python -m venv casia_cefa
source casia_cefa/bin/activate
pip install -e /path/to/at_learner_core/repository/
```


## Second step.
Put pyflow to at_learner_core/utils and install
Original github repository: https://github.com/pathak22/pyflow
Please see installation instructions there.

## Third step.
Train, dev lists creating.
Replace root path to images in train_list.txt, dev_list.txt, dev_test_list.txt

## Fourth step.
```bash
cd rgb_track
python configs_final_exp.py
CUDA_VISIBLE_DEVICES=0 python main.py --config experiments/rgb_track/exp1_protocol41/rgb_track_exp1_protocol41.config
```

After training process run rgb_predictor
```
python test_config.py
CUDA_VISIBLE_DEVICES=0 python rgb_predictor.py --test_config experiment_tests/protocol_4_1/protocol_4_1.config \
 --model_config_path experiments/rgb_track/exp1_protocol41/rgb_track_exp1_protocol41.config \
 --checkpoint_path experiments/rgb_track/exp1_protocol41/checkpoints/model_9.pth
```

