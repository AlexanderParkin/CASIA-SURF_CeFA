def get_backbone(architecture_name, 
                 pretrained=None, 
                 get_feature_size=False,
                 in_channels=3):
    feature_size = None 
    if architecture_name == 'MobilenetV2':
        from .mobilenetv2 import MobileNetV2
        model = MobileNetV2(pretrained=pretrained)
        feature_size = 1280
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
    elif architecture_name == 'simplenet112':
        from .simplenet import SimpleNet112
        model = SimpleNet112(pretrained=pretrained, in_channels=in_channels)
        feature_size = 256
    else:
        raise Exception('Unknown backbone architecture type')

    if get_feature_size:
        return model, feature_size
    else:
        return model


def get_backbone_block(architecture_name,
                       block_number,
                       in_size=None,
                       out_size=None,
                       get_feature_size=False):
    feature_size = None
    if architecture_name == 'resnet':
        from .resnet import get_resnet_block
        block = get_resnet_block(block_number, in_size, out_size)
        feature_sizes = [64, 64, 128, 256, 512]
        feature_size = feature_sizes[block_number]
    if architecture_name == 'simplenet':
        from .simplenet import get_simplenet_block
        block = get_simplenet_block(block_number, in_size, out_size)
        feature_sizes = [16, 32, 64, 128]
        feature_size = feature_sizes[block_number]
    else:
        raise Exception('Unknown backbone architecture type')

    if get_feature_size:
        return block, feature_size
    else:
        return block
