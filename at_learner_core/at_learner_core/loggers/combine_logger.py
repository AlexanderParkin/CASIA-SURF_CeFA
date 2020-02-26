from . import logger
from . import logger_manager


class CombineLogger(logger.Logger):
    def __init__(self, root, logger_config=None, log_manager=logger_manager):
        super().__init__()
        self.root = root
        self.logger_config = logger_config
        self.loggers = []
        for logger_config in self.logger_config.loggers:
            self.loggers.append(log_manager.get_logger(root, logger_config))

    def log_batch(self, batch_idx):
        for curr_logger in self.loggers:
            curr_logger.log_batch(batch_idx)

    def log_epoch(self):
        for curr_logger in self.loggers:
            curr_logger.log_epoch()

    def close(self):
        for curr_logger in self.loggers:
            curr_logger.close()
