

def get_backbone(architecture_name,
                 pretrained=None,
                 get_feature_size=False):
    feature_size = None
    if architecture_name == 'LiteMobileNet':
        from .lite_mobilenet import LiteMobileNet
        model = LiteMobileNet(pretrained=pretrained,
                              input_channels=1)
        feature_size = 256
    else:
        raise Exception('Unknown backbone architecture type')

    if get_feature_size:
        return model, feature_size
    else:
        return model
