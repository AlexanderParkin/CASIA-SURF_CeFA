import os
import pandas as pd
from . import logger


class TestFileLogger(logger.Logger):
    def __init__(self, root, logger_config):
        super().__init__()
        self.root = root
        self.logger_config = logger_config
        self.out_path = os.path.join(self.root.test_config.out_path,
                                'TestFileLogger', f'output_{self.root.epoch}.csv')
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)

    def log_batch(self, batch_idx):
        pass

    def log_epoch(self):
        if self.logger_config.show_metrics.name == 'roc-curve':
            output = self.root.test_info.metric.output
            df = pd.DataFrame()
            df['output_score'] = output
            df.to_csv(self.out_path, index=False)
        elif self.logger_config.show_metrics.name == 'acer':
            output = self.root.test_info.metric.output
            df = pd.DataFrame()
            df['output_score'] = output
            if hasattr(self.root.test_info.metric, 'other_info'):
                for k, v in self.root.test_info.metric.other_info.items():
                    df[k] = v
            df.to_csv(self.out_path, index=False)

    def close(self):
        pass
