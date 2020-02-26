class Logger(object):
    """
    Abstract class for all loggers
    """
    def __init__(self):
        pass

    def log_batch(self, batch_idx):
        pass

    def log_epoch(self):
        pass

    def close(self):
        pass
