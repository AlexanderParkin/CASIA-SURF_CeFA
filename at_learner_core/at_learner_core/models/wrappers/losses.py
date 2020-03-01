import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import combinations

import numpy as np


def get_loss(loss_name, loss_config=None):
    if loss_name == 'BCE':
        loss = nn.BCEWithLogitsLoss()
    elif loss_name == 'CCE':
        loss = nn.CrossEntropyLoss()
    elif loss_name == 'focal_loss':
        gamma = getattr(loss_config, 'gamma', 2.0)
        alpha = getattr(loss_config, 'alpha', 0.25)
        loss = FocalLoss(gamma=gamma,
                         alpha=alpha)
    else:
        raise Exception('Unknown loss type')
    return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, outputs, targets):
        targets = targets.type(outputs.type())

        logpt = -F.binary_cross_entropy_with_logits(
            outputs, targets, reduction="none"
        )
        pt = torch.exp(logpt)

        # compute the loss
        loss = -((1 - pt).pow(self.gamma)) * logpt

        if self.alpha is not None:
            loss = loss * (self.alpha * targets + (1 - self.alpha) * (1 - targets))

        if self.reduction == "mean":
            loss = loss.mean()
        if self.reduction == "sum":
            loss = loss.sum()

        return loss
