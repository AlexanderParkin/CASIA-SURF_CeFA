import time
from collections import OrderedDict
import torch
from . import models
from .datasets.dataset_manager import DatasetManager
from . import utils
from . import loggers


class Runner(object):
    def __init__(self, config, train=True):
        self.config = config
        self.device = torch.device("cuda" if config.train_process_config.ngpu else "cpu")

        self._init_wrapper()
        if train:
            """
            TODO:
            * add parameters, optimizer and scheduler to session iterations
            * add get_optimizer and get_lr_scheduler to 'models' directory. I need discuss.
            * add utils.Metrics with self.config 
            """
            self._init_optimizer()
            self._init_logger()
            self.state = utils.State(self)
            self._init_metrics()
            self._init_loaders()
            self.epoch = 0
            self.best_epoch = False
            self.training = False

    def _init_wrapper(self):
        self.wrapper = models.get_wrapper(self.config)

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

    def _init_optimizer(self):
        parameters = [{'params': self.wrapper.parameters()}]
        self.optimizer = utils.get_optimizer(parameters,
                                             self.config.train_process_config.optimizer_config)
        self.lr_scheduler = utils.get_lr_scheduler(self.config.train_process_config.optimizer_config.lr_config,
                                                   self.optimizer)

    def _init_logger(self):
        self.logger = loggers.get_logger(self)

    def _init_metrics(self):
        self.train_info = utils.LossMetricsMeter(self.config.test_process_config)
        self.test_info = utils.LossMetricsMeter(self.config.test_process_config)
        self.best_test_info = utils.meters.AverageMeter()
        self.batch_time = utils.meters.AverageMeter()
        return

    def _check_best_epoch(self):
        self.best_epoch = False

    def _init_loaders(self):
        self.train_loader = DatasetManager.get_dataloader(self.config.datalist_config.trainlist_config,
                                                          self.config.train_process_config)

        self.val_loader = DatasetManager.get_dataloader(self.config.datalist_config.testlist_configs,
                                                        self.config.train_process_config,
                                                        shuffle=False)

    def train(self):
        if self.config.resume:
            self.state.load_checkpoint()

        if self.config.train_process_config.ngpu > 1:
            self.wrapper.to_parallel(torch.nn.DataParallel)
        """
        TODO:
        * Add iteration by sessions
        * Add freeze parameters
        """
        for epoch in range(self.epoch, self.config.train_process_config.nepochs):
            self.epoch = epoch
            self._process_on_epoch_start()
            self._train_epoch()
            self.lr_scheduler.step()

            if hasattr(self.config.test_process_config, 'run_frequency'):
                if self.config.test_process_config.run_frequency == -1:
                    pass
                elif epoch % self.config.test_process_config.run_frequency == 0:
                    self._test_epoch()
            else:
                self._test_epoch()

            self._check_best_epoch()
            self.logger.log_epoch()
            self.state.create()
            if self.best_epoch:
                self.state.save_checkpoint('best_model_{epoch}.pth'.format(epoch=str(self.epoch).zfill(4)))
            else:
                self.state.save()

    def _process_on_epoch_start(self):
        """
        This method was created for preprocessing before the beginning of epoch.
        For example, change dataset indices.
        :return:
        """
        pass

    def _train_epoch(self):
        self.wrapper.train()
        self.training = True
        self.train_info.reset()
        self.batch_time.reset()
        time_stamp = time.time()
        for batch_idx, data in enumerate(self.train_loader):
            if isinstance(data, dict) or isinstance(data, OrderedDict):
                for k, v in data.items():
                    if isinstance(v, list):
                        data[k] = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in v]
                    else:
                        if isinstance(v, torch.Tensor):
                            data[k] = v.to(self.device)
            else:
                data = data.to(self.device)

            self.optimizer.zero_grad()
            output_dict, batch_loss = self.wrapper(data)  # TODO
            batch_loss.backward()
            self.optimizer.step()
            
            self.train_info.update(batch_loss, output_dict)
            self.batch_time.update(time.time() - time_stamp)
            time_stamp = time.time()
            
            self.logger.log_batch(batch_idx)

    def _test_epoch(self):
        self.wrapper.eval()
        self.training = False
        self.batch_time.reset()
        self.test_info.reset()
        time_stamp = time.time()

        with torch.no_grad():
            for batch_idx, data in enumerate(self.val_loader):
                if isinstance(data, dict) or isinstance(data, OrderedDict):
                    for k, v in data.items():
                        if isinstance(v, list):
                            data[k] = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in v]
                        else:
                            if isinstance(v, torch.Tensor):
                                data[k] = v.to(self.device)
                else:
                    data = data.to(self.device)

                output_dict, batch_loss = self.wrapper(data)
                                
                self.test_info.update(batch_loss, output_dict)
                self.batch_time.update(time.time() - time_stamp)
                time_stamp = time.time()
                self.logger.log_batch(batch_idx)
