import torch.nn as nn
import torch
from at_learner_core.models.wrappers.losses import get_loss
from at_learner_core.models.wrappers.simple_classifier_wrapper import SimpleClassifierWrapper
from at_learner_core.models.architectures import get_backbone
from ..architectures.transformer import TransformerEncoder


class MultiModalWrapper(SimpleClassifierWrapper):
    def __init__(self, wrapper_config):
        super().__init__(wrapper_config)

    def _init_modules(self, wrapper_config):
        self.input_modalities = wrapper_config.input_modalities
        for modal_key in self.input_modalities:
            if (modal_key == 'optical_flow') or (modal_key == 'optical_flow_start'):
                backbone, feature_size = get_backbone(wrapper_config.backbone,
                                                      pretrained=wrapper_config.pretrained,
                                                      get_feature_size=True,
                                                      in_channels=2)
            else:
                backbone, feature_size = get_backbone(wrapper_config.backbone,
                                                      pretrained=wrapper_config.pretrained,
                                                      get_feature_size=True)

            setattr(self, 'backbone_' + modal_key, backbone)

        self.backbone_feature_size = feature_size
        self.pooling = nn.AdaptiveAvgPool2d((1, feature_size))
        self.pooling2 = nn.AdaptiveMaxPool2d((1, feature_size))
        self.pooling3 = nn.AdaptiveMaxPool2d((1, feature_size))

        self.classifier = nn.Sequential(
            nn.Linear(3 * feature_size, feature_size),
            nn.ReLU(inplace=True),
            nn.Linear(feature_size, wrapper_config.nclasses)
        )
        self.classifier = nn.Linear(3 * feature_size, wrapper_config.nclasses)

    def forward(self, x):
        B, C, W, H = x[self.input_modalities[0]].size()
        device = x[self.input_modalities[0]].device
        M = len(self.input_modalities)
        features = torch.empty((B, M, self.backbone_feature_size)).to(device)

        for idx, key in enumerate(self.input_modalities):
            features[:, idx, :] = getattr(self, 'backbone_' + key)(x[key])
        features = features.view((B, M, -1))

        features1 = self.pooling(features)
        features2 = self.pooling2(features)
        features3 = self.pooling3(-features)
        features = torch.cat([features1, features2, features3], axis=2)
        features = features.squeeze()
        output = self.classifier(features)
        sigmoid_output = torch.sigmoid(output)
        if isinstance(self.loss, nn.modules.loss.CrossEntropyLoss):
            x['target'] = x['target'].squeeze()

        output_dict = {'output': sigmoid_output.detach().cpu().numpy(),
                       'target': x['target'].detach().cpu().numpy()}
        for k, v in x.items():
            if k not in ['data', 'target'] + self.input_modalities:
                output_dict[k] = v

        loss = self.loss(output, x['target'])
        return output_dict, loss
