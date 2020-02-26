from at_learner_core.trainer import Runner
from models.wrappers.rgb_simple_wrapper import RGBSimpleWrapper
from models.wrappers.rgb_video_wrapper import RGBVideoWrapper

from at_learner_core.datasets.dataset_manager import DatasetManager


class RGBRunner(Runner):
    def __init__(self, config, train=True):
        super().__init__(config, train=train)

    def _init_wrapper(self):
        self.wrapper = RGBVideoWrapper(self.config.wrapper_config)
        self.wrapper = self.wrapper.to(self.device)

    def _init_loaders(self):
        self.train_loader = DatasetManager.get_dataloader(self.config.datalist_config.trainlist_config,
                                                          self.config.train_process_config)

        self.val_loader = DatasetManager.get_dataloader(self.config.datalist_config.testlist_configs,
                                                        self.config.train_process_config,
                                                        shuffle=False)
