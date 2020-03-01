from at_learner_core.trainer import Runner
from models.wrappers.multi_modal_wrapper import MultiModalWrapper

from at_learner_core.datasets.dataset_manager import DatasetManager


class MultiModalRunner(Runner):
    def __init__(self, config, train=True):
        super().__init__(config, train=train)

    def _init_wrapper(self):
        if self.config.wrapper_config.wrapper_name == 'MultiModalWrapper':
            self.wrapper = MultiModalWrapper(self.config.wrapper_config)

        if hasattr(self.config.wrapper_config, 'freeze_weights') and self.config.wrapper_config.freeze_weights:
            import re
            if type(self.config.wrapper_config.freeze_weights) == str:
                regexes = [self.config.wrapper_config.freeze_weights]
            else:
                regexes = self.config.wrapper_config.freeze_weights

            for param_name, param in self.wrapper.named_parameters():
                for regex in regexes:
                    if re.search(regex, param_name):
                        param.requires_grad = False
                        break

        self.wrapper = self.wrapper.to(self.device)

    def _init_loaders(self):
        self.train_loader = DatasetManager.get_dataloader(self.config.datalist_config.trainlist_config,
                                                          self.config.train_process_config)

        self.val_loader = DatasetManager.get_dataloader(self.config.datalist_config.testlist_configs,
                                                        self.config.train_process_config,
                                                        shuffle=False)
