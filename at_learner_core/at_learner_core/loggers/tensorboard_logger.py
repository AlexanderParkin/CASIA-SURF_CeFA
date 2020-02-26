from torch.utils.tensorboard import SummaryWriter
from . import logger


class TensorboardLogger(logger.Logger):
    def __init__(self, root, logger_config):
        super().__init__()
        self.root = root
        self.logger_config = logger_config
        out_path = self.root.config.checkpoint_config.out_path
        self.writer = SummaryWriter(log_dir=out_path)

    def log_batch(self, batch_idx):
        if batch_idx % self.logger_config.log_batch_interval == 0:
            cur_len = len(self.root.train_loader) if self.root.training else len(self.root.val_loader)
            cur_loss = self.root.train_info.loss if self.root.training else self.root.test_info.loss

            output_string = 'Train ' if self.root.training else 'Test '
            self.writer.add_scalar('Iteration loss/' + output_string,
                                   cur_loss.val,
                                   self.root.epoch + batch_idx / cur_len)

            if not self.root.training:
                if self.logger_config.show_metrics.name == 'tpr@fpr':
                    pass

    def log_epoch(self):
        self.writer.add_scalars('Loss',
                                {'train': self.root.train_info.loss.avg,
                                 'test': self.root.test_info.loss.avg},
                                self.root.epoch)

        if self.logger_config.show_metrics.name == 'tpr@fpr':
            fpr = self.logger_config.show_metrics.fpr
            self.writer.add_scalar(f'TPR@FPR={fpr}',
                                   self.root.test_info.metric.get_tpr(fpr),
                                   self.root.epoch)
        elif self.logger_config.show_metrics.name == 'accuracy':
            self.writer.add_scalars('Accuracy',
                                    {'Train': self.root.train_info.metric.get_accuracy(),
                                     'Test': self.root.test_info.metric.get_accuracy()},
                                    self.root.epoch)
        self.writer.flush()

    def close(self):
        self.writer.close()


class TestTensorboardLogger(logger.Logger):
    def __init__(self, root, logger_config):
        super().__init__()
        self.root = root
        self.logger_config = logger_config
        out_path = self.root.test_config.out_path
        self.writer = SummaryWriter(log_dir=out_path)

    def log_batch(self, batch_idx):
        pass

    def log_epoch(self):
        if self.logger_config.show_metrics.name == 'roc-curve':
            fpr_arr, tpr_arr, thr_arr = self.root.test_info.metric.get_roc_curve()
            for fpr, tpr in zip(fpr_arr, tpr_arr):
                self.writer.add_scalar(f'ROC curve',
                                       tpr,
                                       fpr*100)
        self.writer.flush()

    def close(self):
        self.writer.close()
