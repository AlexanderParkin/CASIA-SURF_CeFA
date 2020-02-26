import os
from . import logger


class TerminalLogger(logger.Logger):
    def __init__(self, root, logger_config):
        super().__init__()
        self.root = root
        self.logger_config = logger_config

    def log_batch(self, batch_idx):
        if batch_idx % self.logger_config.log_batch_interval == 0:
            cur_len = len(self.root.train_loader) if self.root.training else len(self.root.val_loader)
            cur_loss = self.root.train_info.loss if self.root.training else self.root.test_info.loss

            output_string = 'Train ' if self.root.training else 'Test '
            output_string += 'Epoch {}[{:.2f}%]: [{:.2f}({:.3f}) s]\t'.format(self.root.epoch,
                                                                              100. * batch_idx / cur_len,
                                                                              self.root.batch_time.val,
                                                                              self.root.batch_time.avg)

            loss_i_string = 'Loss: {:.3f}({:.4f})\t'.format(cur_loss.val,
                                                            cur_loss.avg)
            output_string += loss_i_string
            if not self.root.training:
                if self.logger_config.show_metrics.name == 'tpr@fpr':
                    pass

            print(output_string)

    def _log_msg(self, msg=''):
        mode = 'a' if msg else 'w'
        out_root = self.root.config.checkpoint_config.out_path
        f = open(os.path.join(out_root, 'log_files', 'train_log.txt'), mode)
        f.write(msg)
        f.close()

    def log_epoch(self):
        """ Epoch results log string"""
        out_train = 'Train Loss: '
        out_test = 'Test Loss:  '

        loss_i_string = 'Loss: {:.4f}\t'.format(self.root.train_info.loss.avg)
        out_train += loss_i_string
        loss_i_string = 'Loss: {:.4f}\t'.format(self.root.test_info.loss.avg)
        out_test += loss_i_string

        out_train += '\nTrain Metric: '
        out_test += '\nTest Metric:  '
        if self.logger_config.show_metrics.name == 'tpr@fpr':
            fpr = self.logger_config.show_metrics.fpr
            metrics_i_string = 'TPR@FPR {fpr}: {tpr:.3f}\t'.format(fpr=fpr,
                                                                   tpr=self.root.train_info.metric.get_tpr(fpr))
            out_train += metrics_i_string

            metrics_i_string = 'TPR@FPR {fpr}: {tpr:.3f}\t'.format(fpr=fpr,
                                                                   tpr=self.root.test_info.metric.get_tpr(fpr))
            out_test += metrics_i_string
        elif self.logger_config.show_metrics.name == 'accuracy':
            metrics_i_string = 'Acc {acc:.3f}\t'.format(acc=self.root.train_info.metric.get_accuracy())
            out_train += metrics_i_string

            metrics_i_string = 'Acc {acc:.3f}\t'.format(acc=self.root.test_info.metric.get_accuracy())
            out_test += metrics_i_string
        elif self.logger_config.show_metrics.name == 'acer':
            acer, apcer, bpcer = self.root.train_info.metric.get_all_metrics(0.5)
            metrics_i_string = f'ACER: {acer:.3f}, APCER: {apcer:.3f}, BPCER: {bpcer:.3f}\t'
            out_train += metrics_i_string

            acer, apcer, bpcer = self.root.test_info.metric.get_all_metrics(0.5)
            metrics_i_string = f'ACER: {acer:.3f}, APCER: {apcer:.3f}, BPCER: {bpcer:.3f}\t'
            out_test += metrics_i_string

        is_best = ''
        out_res = is_best + 'Epoch {} results:\n'.format(self.root.epoch) + out_train + '\n' + out_test + '\n'
        print(out_res)
        self._log_msg(out_res)

    def close(self):
        pass
