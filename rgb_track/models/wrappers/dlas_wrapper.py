import torch.nn as nn
import torch
from at_learner_core.models.wrappers.losses import get_loss
from at_learner_core.models.wrappers.simple_classifier_wrapper import SimpleClassifierWrapper
from at_learner_core.models.architectures import get_backbone_block
from ..architectures.transformer import TransformerEncoder
from collections import OrderedDict


class DLASWrapper(SimpleClassifierWrapper):
    def __init__(self, wrapper_config):
        super().__init__(wrapper_config)

    def _init_modules(self, wrapper_config):
        self.input_modalities = wrapper_config.input_modalities
        for modal_key in self.input_modalities:
            for idx in range(0, 4):
                if 'optical_flow' in modal_key and idx == 0:
                    backbone, feature_size = get_backbone_block(wrapper_config.backbone, idx, get_feature_size=True,
                                                                in_size=2)
                else:
                    backbone, feature_size = get_backbone_block(wrapper_config.backbone, idx, get_feature_size=True)
                setattr(self, f'{modal_key}_block{idx}', backbone)

        feature_sizes = []
        for idx in range(1, 4):
            backbone, feature_size = get_backbone_block(wrapper_config.backbone, idx, get_feature_size=True)
            feature_sizes.append(feature_size)
            setattr(self, f'agg_block{idx}', backbone)

        for idx in range(1, 4):
            planes = feature_sizes[idx-1]
            adaptive_block = nn.Sequential(nn.Conv2d(planes, planes, 1), nn.ReLU(inplace=True))
            setattr(self, f'agg_adaptive_block{idx}', adaptive_block)

        self.backbone_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.backbone_feature_size = feature_size

        self.pooling = nn.AdaptiveAvgPool2d((1, feature_size))
        self.pooling2 = nn.AdaptiveMaxPool2d((1, feature_size))
        self.pooling3 = nn.AdaptiveMaxPool2d((1, feature_size))

        self.classifier = nn.Linear(3*feature_size, wrapper_config.nclasses)
    
    def forward(self, x):
        B, C, W, H = x[self.input_modalities[0]].size()
        device = x[self.input_modalities[0]].device
        features_dict = OrderedDict()
        for modal_key in self.input_modalities:
            features_dict[modal_key] = getattr(self, f'{modal_key}_block0')(x[modal_key])

        features_agg = features_dict[self.input_modalities[0]]
        for modal_key in self.input_modalities[1:]:
            features_agg = features_agg + features_dict[modal_key]
        features_dict['agg'] = features_agg

        for idx in range(1, 4):
            for modal_key in self.input_modalities + ['agg']:
                features_dict[modal_key] = getattr(self, f'{modal_key}_block{idx}')(features_dict[modal_key])
            features_agg = features_dict[self.input_modalities[0]]
            for modal_key in self.input_modalities[1:]:
                features_agg = features_agg + features_dict[modal_key]
            features_dict['agg'] = features_dict['agg'] + getattr(self, f'agg_adaptive_block{idx}')(features_agg)

        for modal_key in self.input_modalities + ['agg']:
            features_dict[modal_key] = self.backbone_pooling(features_dict[modal_key]).squeeze()

        M = len(self.input_modalities) + 1
        features = torch.empty((B, M, self.backbone_feature_size)).to(device)
        for idx, key in enumerate(self.input_modalities + ['agg']):
            features[:, idx, :] = features_dict[key]
        features = features.view((B, M, -1))

        """
        results_dict = OrderedDict()   
        for modal_key in self.input_modalities + ['agg']:
            results_dict[modal_key] = getattr(self, f'{modal_key}_clf')(features_dict[modal_key])
        """
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
