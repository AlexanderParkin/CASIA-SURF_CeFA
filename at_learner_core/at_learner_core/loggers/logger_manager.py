from . import terminal_logger
from . import tensorboard_logger
from . import combine_logger
from . import file_logger


def get_logger(model, logger_config=None):
    logger_config = model.config.logger_config if logger_config is None else logger_config
    if logger_config.logger_type == 'tensorboard':
        return tensorboard_logger.TensorboardLogger(model, logger_config)
    elif logger_config.logger_type == 'terminal':
        return terminal_logger.TerminalLogger(model, logger_config)
    elif logger_config.logger_type == 'log_combiner':
        return combine_logger.CombineLogger(model, logger_config)
    elif logger_config.logger_type == 'test_tensorboard':
        return tensorboard_logger.TestTensorboardLogger(model, logger_config)
    elif logger_config.logger_type == 'test_filelogger':
        return file_logger.TestFileLogger(model, logger_config)
    else:
        raise Exception('Unknown logger type')