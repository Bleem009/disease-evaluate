# src/training/metrics.py
"""评估指标"""

import torch


def iou_score(pred, target, threshold=0.5):
    """计算IoU"""
    pred = (torch.sigmoid(pred) > threshold).float()
    target = target.unsqueeze(1).float()

    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection

    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


def pixel_accuracy(pred, target, threshold=0.5):
    """像素准确率"""
    pred = (torch.sigmoid(pred) > threshold).float()
    target = target.unsqueeze(1).float()

    correct = (pred == target).float().sum()
    total = target.numel()

    return (correct / total).item()


def dice_score(pred, target, threshold=0.5):
    """Dice系数"""
    pred = (torch.sigmoid(pred) > threshold).float()
    target = target.unsqueeze(1).float()

    intersection = (pred * target).sum(dim=(2, 3))
    return (2. * intersection / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + 1e-6)).mean().item()