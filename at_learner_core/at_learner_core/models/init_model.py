from .wrappers import SimpleClassifierWrapper


def get_wrapper(config, wrapper_func=None):
    if wrapper_func is not None:
        wrapper = wrapper_func(config)
    elif config.wrapper_config.wrapper_name == 'SimpleClassifierWrapper':
        wrapper = SimpleClassifierWrapper(config.wrapper_config)
    else:
        raise Exception('Unknown wrapper architecture type')
    return wrapper
