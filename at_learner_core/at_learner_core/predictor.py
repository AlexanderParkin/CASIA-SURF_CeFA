from collections import OrderedDict
from tqdm import tqdm
import torch
from . import models
from . import utils
from . import loggers
from .datasets.dataset_manager import DatasetManager


class Predictor(object):
    def __init__(self, test_config, model_config, checkpoint_path):
        self.test_config = test_config
        self.model_config = model_config
        self.device = torch.device("cuda" if self.test_config.ngpu else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        self._init_wrapper(checkpoint)
        self._init_loaders()
        self._init_metrics()
        self._init_logger()

    def _init_logger(self):
        logger_config = self.test_config.logger_config
        self.logger = loggers.get_logger(self, logger_config)

    def _init_wrapper(self, checkpoint):
        self.wrapper = models.get_wrapper(self.model_config)
        self.wrapper.load_state_dict(checkpoint['state_dict'])
        self.wrapper = self.wrapper.to(self.device)

    def _init_loaders(self):
        dataset_config = self.test_config.dataset_configs
        if dataset_config.transform_source == 'model_config':
            transforms = self.model_config.datalist_config.testlist_configs.transforms
            setattr(dataset_config, 'transforms', transforms)
        dataset = DatasetManager._get_dataset(dataset_config)
        self.test_loader = DatasetManager.get_dataloader_by_args(dataset=dataset,
                                                                 batch_size=dataset_config.batch_size,
                                                                 num_workers=dataset_config.nthreads)

    def _init_metrics(self):
        self.test_info = utils.LossMetricsMeter(self.test_config.dataset_configs.test_process_config)

    def run_predict(self):
        self.wrapper.eval()
        self.test_info.reset()
        with torch.no_grad():
            for batch_idx, data in tqdm(enumerate(self.test_loader),
                                        total=len(self.test_loader)):
                if isinstance(data, dict) or isinstance(data, OrderedDict):
                    for k, v in data.items():
                        data[k] = v.to(self.device)
                else:
                    data = data.to(self.device)

                output_dict, batch_loss = self.wrapper(data)

                self.test_info.update((batch_loss,
                                       output_dict[self.test_info.target_column],
                                       output_dict['output']))
                self.logger.log_batch(batch_idx)

        self.logger.log_epoch()
        self.logger.close()
