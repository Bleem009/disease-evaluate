# src/training/losses.py
"""损失函数"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    """Dice Loss + BCE Loss"""

    def __init__(self, dice_weight=1.0, bce_weight=1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        # pred: [B, 1, H, W], target: [B, H, W]
        target = target.unsqueeze(1).float()

        # BCE Loss
        bce_loss = self.bce(pred, target)

        # Dice Loss
        pred_prob = torch.sigmoid(pred)
        intersection = (pred_prob * target).sum(dim=(2, 3))
        union = pred_prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + 1e-6) / (union + 1e-6)
        dice_loss = 1 - dice.mean()

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FocalLoss(nn.Module):
    """Focal Loss - 用于处理类别不平衡"""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        target = target.unsqueeze(1).float()
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pred_prob = torch.sigmoid(pred)
        p_t = (pred_prob * target) + ((1 - pred_prob) * (1 - target))
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)

        return (alpha_t * (1. - p_t) ** self.gamma * bce).mean()