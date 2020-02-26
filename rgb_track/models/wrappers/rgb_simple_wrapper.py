import torch.nn as nn
import torch
from at_learner_core.models.wrappers.losses import get_loss
from at_learner_core.models.wrappers.simple_classifier_wrapper import SimpleClassifierWrapper
from at_learner_core.models.architectures import get_backbone


class RGBSimpleWrapper(SimpleClassifierWrapper):
    def __init__(self, wrapper_config):
        super().__init__(wrapper_config)

    def _init_modules(self, wrapper_config):
        self.backbone, feature_size = get_backbone(wrapper_config.backbone,
                                                   pretrained=wrapper_config.pretrained,
                                                   get_feature_size=True)
        self.classifier = nn.Linear(feature_size, wrapper_config.nclasses)

    def predict(self, x):
        features = self.backbone(x['data'])
        output = self.classifier(features)
        if isinstance(self.loss, (nn.BCELoss, nn.BCEWithLogitsLoss)):
            output = torch.sigmoid(output)
        elif isinstance(self.loss, nn.CrossEntropyLoss):
            output = nn.functional.softmax(output, dim=0)[:, 1]
        output_dict = {'output': output.detach().cpu()}
        return output_dict


class RGBSimpleInferenceWrapper(RGBSimpleWrapper):
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        output = torch.sigmoid(output)
        return output
