from at_learner_core.predictor import Predictor
from models.wrappers.multi_modal_wrapper_img import MultiModalWrapper
from at_learner_core.datasets.dataset_manager import DatasetManager
import argparse
import torch
import os


class MultiModalPredictor(Predictor):
    def __init__(self, test_config, model_config, checkpoint_path):
        super().__init__(test_config, model_config, checkpoint_path)

    def _init_wrapper(self, checkpoint):
        self.wrapper = MultiModalWrapper(self.model_config.wrapper_config)
        self.wrapper.load_state_dict(checkpoint['state_dict'])
        self.wrapper = self.wrapper.to(self.device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--test_config_path',
                        type=str,
                        help='Path to test config')
    parser.add_argument('--model_config_path',
                        type=str,
                        help='Path to model config')
    parser.add_argument('--checkpoint_path',
                        type=str,
                        help='Path to checkpoint')
    args = parser.parse_args()

    test_config = torch.load(args.test_config_path)
    model_config = torch.load(args.model_config_path)

    # change out_path in test_config
    out_path = os.path.join(test_config.out_path,
                            model_config.head_config.task_name,
                            model_config.head_config.exp_name)
    os.makedirs(out_path, exist_ok=True)
    test_config.out_path = out_path

    predictor = MultiModalPredictor(test_config, model_config, args.checkpoint_path)
    predictor.run_predict()
