import torch.nn as nn
from .losses import get_loss
from ..architectures import get_backbone
from .wrapper import Wrapper


class SimpleClassifierWrapper(Wrapper):
    def __init__(self, wrapper_config):
        super().__init__()
        self.backbone = None
        self.classifier = None
        self._init_modules(wrapper_config)
        self._init_loss(wrapper_config)
        self.batch_info = {}

    def _init_modules(self, wrapper_config):
        self.backbone, feature_size = get_backbone(wrapper_config.backbone,
                                                   pretrained=wrapper_config.pretrained,
                                                   get_feature_size=True)
        self.classifier = nn.Linear(feature_size, wrapper_config.nclasses)

    def _init_loss(self, wrapper_config):
        loss_config = None
        if hasattr(wrapper_config, 'loss_config'):
            loss_config = wrapper_config.loss_config
        self.loss = get_loss(wrapper_config.loss, loss_config)

    def forward(self, x):
        features = self.backbone(x['data'])
        output = self.classifier(features)
        if isinstance(self.loss, nn.modules.loss.CrossEntropyLoss):
            x['target'] = x['target'].squeeze()
        output_dict = {'output': output.detach().cpu().numpy(),
                       'target': x['target'].detach().cpu().numpy()}
        loss = self.loss(output, x['target'])
        return output_dict, loss

    def predict(self, x):
        features = self.backbone(x['data'])
        output = self.classifier(features)
        output_dict = {'output': output.detach().cpu()}
        return output_dict

    def to_parallel(self, parallel_class):
        self.backbone = parallel_class(self.backbone)
        self.classifier = parallel_class(self.classifier)
        return self
