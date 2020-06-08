# Chalearn CeFA Face Anti-Spoofing challenge
This is code of our solution for Chalearn Single-modal face anti spofing attack detection challenge at CVPR 2020.

Our solution based on two types of artificial transforms: rank pooling[1] and optical flow[2], combined in end-to-end pipeline for spoof detection with sequence augmentation to enrich the collection of fake tracks.

![Alt text](figures/pipeline.png?raw=true "Network Architecture")

## References
[1] Basura  Fernando,  Efstratios  Gavves,  Jose    Oramas,  AmirGhodrati,  and  Tinne  Tuytelaars.Rank  pooling  for  actionrecognition.TPAMI, 39(4):773–787, 201

[2] C. Liu. Beyond pixels: Exploring new representations and applications for motion analysis. Doctoral Thesis. MIT, 2009.

## Training steps
### Step 1. 
Install at_learner_core
```
cd /path/to/new/pip/environment
python -m venv casia_cefa
source casia_cefa/bin/activate
pip install -e /path/to/at_learner_core/repository/
```

### Step 2.
Put pyflow to at_learner_core/utils, replace OpticalFlow.cpp to remove logs spamming to console and build
Original github repository: https://github.com/pathak22/pyflow

```bash
cd at_learner_core/at_learner_core/utils
git clone https://github.com/pathak22/pyflow.git
cd pyflow/
cp ../../../../data/OpticalFlow.cpp src/OpticalFlow.cpp #Remove logs spamming to console
pip install cython
python setup.py build_ext -i
python demo.py # 
```

Please see installation instructions there.

### Step 3.
Train, dev lists creating.
Replace root path to images in train_list.txt, dev_list.txt, dev_test_list.txt
```python
cd data
python prepare_lists.py --data_path /path/to/casia/surf/cefa/directory
cd ..
```

### Step 4.
```bash
cd ../rgb_track
python configs_final_exp.py
CUDA_VISIBLE_DEVICES=0 python main.py --config experiments/rgb_track/exp1_protocol_4_1/rgb_track_exp1_protocol_4_1.config;
CUDA_VISIBLE_DEVICES=0 python main.py --config experiments/rgb_track/exp1_protocol_4_2/rgb_track_exp1_protocol_4_2.config;
CUDA_VISIBLE_DEVICES=0 python main.py --config experiments/rgb_track/exp1_protocol_4_3/rgb_track_exp1_protocol_4_3.config
```

### Step 5
After training process run rgb_predictor
```
python test_config.py
CUDA_VISIBLE_DEVICES=0 python rgb_predictor.py --test_config experiment_tests/protocol_4_1/protocol_4_1.config \
 --model_config_path experiments/rgb_track/exp1_protocol_4_1/rgb_track_exp1_protocol_4_1.config \
 --checkpoint_path experiments/rgb_track/exp1_protocol4_1/checkpoints/model_4.pth
CUDA_VISIBLE_DEVICES=0 python rgb_predictor.py --test_config experiment_tests/protocol_4_2/protocol_4_2.config \
 --model_config_path experiments/rgb_track/exp1_protocol_4_2/rgb_track_exp1_protocol_4_2.config \
 --checkpoint_path experiments/rgb_track/exp1_protocol_4_2/checkpoints/model_4.pth
CUDA_VISIBLE_DEVICES=0 python rgb_predictor.py --test_config experiment_tests/protocol_4_3/protocol_4_3.config \
 --model_config_path experiments/rgb_track/exp1_protocol_4_3/rgb_track_exp1_protocol_4_3.config \
 --checkpoint_path experiments/rgb_track/exp1_protocol_4_3/checkpoints/model_4.pth
```
### Step 6
Compile submit_file
```bash
python compile_submit_file.py
```

