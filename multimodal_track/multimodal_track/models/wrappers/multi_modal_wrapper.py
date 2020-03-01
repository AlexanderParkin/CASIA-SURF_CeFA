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
            if 'optical_flow' in modal_key:
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
        #self.transformer = TransformerEncoder(num_layers=6, hidden_size=feature_size)


        self.classifier = nn.Sequential(
            nn.Linear(3*feature_size, feature_size),
            nn.ReLU(inplace=True),
            nn.Linear(feature_size, wrapper_config.nclasses)
        )
        self.classifier = nn.Linear(3*feature_size, wrapper_config.nclasses)
    
    def forward(self, x):
        B, C, W, H = x[self.input_modalities[0]].size()
        device = x[self.input_modalities[0]].device
        M = len(self.input_modalities)#+1
        features = torch.empty((B, M, self.backbone_feature_size)).to(device)
        
        #image_features = torch.cat([x[self.input_image_modalities[0]],x[self.input_image_modalities[1]]],axis=1)
        #features[:,M-1,:] = self.image_backbone(image_features)
        
        for idx, key in enumerate(self.input_modalities):
            features[:, idx, :] = getattr(self, 'backbone_' + key)(x[key])
        features = features.view((B, M, -1))

        #features = self.transformer(features)
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
            if k not in ['rgb_data', 'ir_data', 'depth_data', 'target'] + self.input_modalities:
                output_dict[k] = v

        loss = self.loss(output, x['target'])
        return output_dict, loss

    def predict(self, x):
        B, L, C, W, H = x['data'].size()
        input_data = x['data'].view((-1, C, W, H))
        features = self.backbone(input_data)
        features = features.view((B, L, -1))
        #features = self.transformer(features)    
        features = self.pooling(features)
        features = features.squeeze()
        output = self.classifier(features)
        if isinstance(self.loss, (nn.BCELoss, nn.BCEWithLogitsLoss)):
            output = torch.sigmoid(output)
        elif isinstance(self.loss, nn.CrossEntropyLoss):
            output = nn.functional.softmax(output, dim=0)[:, 1]
        output_dict = {'output': output.detach().cpu()}
        return output_dict
