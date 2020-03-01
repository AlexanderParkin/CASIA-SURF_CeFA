import torch.nn as nn
import torch
from at_learner_core.models.wrappers.losses import get_loss
from at_learner_core.models.wrappers.simple_classifier_wrapper import SimpleClassifierWrapper
from at_learner_core.models.architectures import get_backbone_block
from ..architectures.transformer import TransformerEncoder
from collections import OrderedDict


class SDNetWrapper(SimpleClassifierWrapper):
    def __init__(self, wrapper_config):
        super().__init__(wrapper_config)

    def _init_modules(self, wrapper_config):
        self.input_modalities = wrapper_config.input_modalities
        for modal_key in self.input_modalities:
            for idx in range(0, 4):
                backbone, feature_size = get_backbone_block(wrapper_config.backbone, idx, get_feature_size=True)
                setattr(self, f'{modal_key}_block{idx}', backbone)

        for idx in range(1, 4):
            backbone, feature_size = get_backbone_block(wrapper_config.backbone, idx, get_feature_size=True)
            setattr(self, f'fusion_block{idx}', backbone)

        for modal_key in self.input_modalities + ['fusion', 'sdf']:
            clf_layer = nn.Linear(feature_size, wrapper_config.nclasses)
            setattr(self, f'{modal_key}_clf', clf_layer)

        self.backbone_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.backbone_feature_size = feature_size
        #self.pooling = nn.AdaptiveAvgPool2d((1, feature_size))
        #self.transformer = TransformerEncoder(num_layers=6, hidden_size=feature_size)

        #self.classifier = nn.Sequential(
        #    nn.Linear(feature_size, feature_size//2),
        #    nn.ReLU(inplace=True),
        #    nn.Linear(feature_size//2, wrapper_config.nclasses)
        #)
        #self.classifier = nn.Linear(feature_size, wrapper_config.nclasses)
    
    def forward(self, x):
        features1 = OrderedDict()
        for modal_key in self.input_modalities:
            features1[modal_key] = getattr(self, f'{modal_key}_block0')(x[modal_key])
        features1['fusion'] = features1[self.input_modalities[0]] + features1[self.input_modalities[1]]

        features_end = OrderedDict()
        for modal_key in self.input_modalities + ['fusion']:
            features = getattr(self, f'{modal_key}_block1')(features1[modal_key])
            features = getattr(self, f'{modal_key}_block2')(features)
            features = getattr(self, f'{modal_key}_block3')(features)
            features_end[modal_key] = self.backbone_pooling(features).squeeze()
        features_end['sdf'] = features_end[self.input_modalities[0]] + \
                                           features_end[self.input_modalities[1]] + \
                                           features_end['fusion']

        results_dict = OrderedDict()
        for modal_key in self.input_modalities + ['fusion', 'sdf']:
            results_dict[modal_key] = getattr(self, f'{modal_key}_clf')(features_end[modal_key])

        loss_dict = OrderedDict()
        for modal_key in self.input_modalities + ['fusion', 'sdf']:
            loss_dict[modal_key] = self.loss(results_dict[modal_key], x['target'])

        #loss = loss_dict[self.input_modalities[0]] + loss_dict[self.input_modalities[1]]
        #loss = loss + loss_dict['fusion']
        loss = loss_dict['sdf']

        output_dict = {'target': x['target'].detach().cpu().numpy()}
        for modal_key in self.input_modalities + ['fusion', 'sdf']:
            output_dict[f'output_{modal_key}'] = torch.sigmoid(results_dict[modal_key]).detach().cpu().numpy().tolist()

        output_dict['output'] = torch.sigmoid(results_dict['sdf']).detach().cpu().numpy()
        for k, v in x.items():
            if k not in ['data', 'target'] + self.input_modalities:
                output_dict[k] = v

        return output_dict, loss

    def predict(self, x):
        B, L, C, W, H = x['data'].size()
        input_data = x['data'].view((-1, C, W, H))
        features = self.backbone(input_data)
        features = features.view((B, L, -1))
        features = self.transformer(features)    
        features = self.pooling(features)
        features = features.squeeze()
        output = self.classifier(features)
        if isinstance(self.loss, (nn.BCELoss, nn.BCEWithLogitsLoss)):
            output = torch.sigmoid(output)
        elif isinstance(self.loss, nn.CrossEntropyLoss):
            output = nn.functional.softmax(output, dim=0)[:, 1]
        output_dict = {'output': output.detach().cpu()}
        return output_dict
