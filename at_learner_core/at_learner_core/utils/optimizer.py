import torch.optim as optim


def get_lr_scheduler(scheduler_config, optimizer):
    if scheduler_config.lr_type == 'StepLR':
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                 step_size=scheduler_config.lr_decay_period,
                                                 gamma=scheduler_config.lr_decay_lvl)
        return lr_scheduler
    elif scheduler_config.lr_type == 'CosineLR':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=scheduler_config.t_max,
                                                            eta_min=scheduler_config.eta_min)
        return lr_scheduler
    else:
        raise Exception('Unknown lr_type')


def get_optimizer(parameters, optimizer_config):
    if optimizer_config.name == 'SGD':
        weight_decay = getattr(optimizer_config, 'weight_decay', 5e-4)
        momentum = getattr(optimizer_config, 'momentum', 0.9)
        optimizer = optim.SGD(parameters,
                              lr=optimizer_config.lr_config.lr,
                              momentum=momentum,
                              weight_decay=weight_decay)
    elif optimizer_config.name == 'Adam':
        weight_decay = getattr(optimizer_config, 'weight_decay', 5e-4)
        betas = getattr(optimizer_config, 'betas', (0.9, 0.999))
        optimizer = optim.Adam(parameters,
                               lr=optimizer_config.lr_config.lr,
                               betas=betas,
                               weight_decay=weight_decay)
    elif optimizer_config.name == 'AdamW':
        weight_decay = getattr(optimizer_config, 'weight_decay', 0.01)
        betas = getattr(optimizer_config, 'betas', (0.9, 0.999))
        optimizer = optim.AdamW(parameters,
                                lr=optimizer_config.lr_config.lr,
                                betas=betas,
                                weight_decay=weight_decay)
    return optimizer
