def get_backbone(architecture_name, 
                 pretrained=None, 
                 get_feature_size=False):
    feature_size = None 
    if architecture_name == 'MobilenetV2':
        from .mobilenetv2 import MobileNetV2
        model = MobileNetV2(pretrained=pretrained)
        feature_size = 1280
    elif architecture_name == 'se_mobilenet_v2':
        from .se_mobilenetv2 import SEMobileNetV2
        model = SEMobileNetV2(pretrained=pretrained)
        feature_size = 1280
    elif architecture_name == 'mobilenet_v2b':
        from .mobilenetv2b import MobileNetV2
        model = MobileNetV2(pretrained=pretrained)
        feature_size = 1792
    elif architecture_name.startswith('efficientnet-b'):
        from .efficientnet import EfficientNet
        if pretrained == 'ImageNet':
            model = EfficientNet.from_pretrained(architecture_name, num_classes=1)
        else:
            model = EfficientNet.from_name(architecture_name, num_classes=1)

        if architecture_name == 'efficientnet-b0' or architecture_name == 'efficientnet-b1':
            feature_size = 1280
        elif architecture_name == 'efficientnet-b2':
            feature_size = 1408
        elif architecture_name == 'efficientnet-b3':
            feature_size = 1536
        else:
            raise Exception('Unknown efficientnet backbone architecture type')
    elif architecture_name == 'resnext50':
        from .resnext import ResNeXt
        model = ResNeXt(cardinality=32, 
                        depth=50, 
                        widen_factor=4, 
                        version=2, 
                        pretrained=pretrained)
        feature_size = 2048
    else:
        raise Exception('Unknown backbone architecture type')

    if get_feature_size:
        return model, feature_size
    else:
        return model
