# src/training/optimizers.py
"""优化器配置"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler


class PolyLR(_LRScheduler):
    """多项式学习率衰减"""

    def __init__(self, optimizer, max_epochs, power=0.9, last_epoch=-1):
        self.max_epochs = max_epochs
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (1 - self.last_epoch / self.max_epochs) ** self.power
                for base_lr in self.base_lrs]


def create_optimizer(model, lr_encoder, lr_decoder, momentum=0.9, weight_decay=1e-4):
    """
    为两阶段模型创建优化器（分别设置encoder和decoder的学习率）

    Args:
        model: 分割模型
        lr_encoder: encoder学习率
        lr_decoder: decoder学习率
    """
    encoder_params = []
    decoder_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if 'encoder' in name:
            encoder_params.append(param)
        else:
            decoder_params.append(param)

    optimizer = optim.SGD([
        {'params': encoder_params, 'lr': lr_encoder, 'name': 'encoder'},
        {'params': decoder_params, 'lr': lr_decoder, 'name': 'decoder'}
    ], momentum=momentum, weight_decay=weight_decay)

    return optimizer


def create_scheduler(optimizer, scheduler_type: str, **kwargs):
    """
    创建学习率调度器

    Args:
        scheduler_type: 'poly', 'step', 'cosine', 'plateau'
    """
    if scheduler_type == 'poly':
        return PolyLR(optimizer, kwargs['max_epochs'], kwargs.get('power', 0.9))
    elif scheduler_type == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 30),
            gamma=kwargs.get('gamma', 0.1)
        )
    elif scheduler_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs['max_epochs']
        )
    elif scheduler_type == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=kwargs.get('factor', 0.1),
            patience=kwargs.get('patience', 10)
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")