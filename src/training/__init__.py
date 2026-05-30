# src/training/__init__.py
"""训练模块"""

from .trainer import SegmentationTrainer
from .losses import DiceBCELoss, FocalLoss
from .metrics import iou_score, pixel_accuracy, dice_score

__all__ = [
    'SegmentationTrainer',
    'DiceBCELoss',
    'FocalLoss',
    'iou_score',
    'pixel_accuracy',
    'dice_score'
]