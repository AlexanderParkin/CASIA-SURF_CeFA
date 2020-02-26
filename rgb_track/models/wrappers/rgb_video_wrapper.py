import torch.nn as nn
import torch
from at_learner_core.models.wrappers.losses import get_loss
from at_learner_core.models.wrappers.simple_classifier_wrapper import SimpleClassifierWrapper
from at_learner_core.models.architectures import get_backbone


class RGBVideoWrapper(SimpleClassifierWrapper):
    def __init__(self, wrapper_config):
        super().__init__(wrapper_config)

    def _init_modules(self, wrapper_config):
        self.backbone, feature_size = get_backbone(wrapper_config.backbone,
                                                   pretrained=wrapper_config.pretrained,
                                                   get_feature_size=True)
        self.pooling = nn.AdaptiveAvgPool2d((1, feature_size))
        self.classifier = nn.Linear(feature_size, wrapper_config.nclasses)

    def forward(self, x):
        B, L, C, W, H = x['data'].size()
        input_data = x['data'].view((-1, C, W, H))
        features = self.backbone(input_data)
        features = features.view((B, L, -1))
        features = self.pooling(features)
        features = features.squeeze()
        output = self.classifier(features)
        if isinstance(self.loss, nn.modules.loss.CrossEntropyLoss):
            x['target'] = x['target'].squeeze()
        output_dict = {'output': output.detach().cpu().numpy(),
                       'target': x['target'].detach().cpu().numpy()}
        loss = self.loss(output, x['target'])
        return output_dict, loss

    def predict(self, x):
        features = []
        for batch_img in x['data']:
            features.append(self.backbone(batch_img))
        features = torch.stack(features).permute(1, 0, 2)
        features = self.pooling(features)
        features = features.squeeze()
        output = self.classifier(features)
        if isinstance(self.loss, (nn.BCELoss, nn.BCEWithLogitsLoss)):
            output = torch.sigmoid(output)
        elif isinstance(self.loss, nn.CrossEntropyLoss):
            output = nn.functional.softmax(output, dim=0)[:, 1]
        output_dict = {'output': output.detach().cpu()}
        return output_dict
