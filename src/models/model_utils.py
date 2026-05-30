# src/models/model_utils.py
"""模型工具函数"""

import torch
import torch.nn as nn
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_pretrained_weights(model: nn.Module, pretrained_path: Path, strict: bool = False):
    """
    加载预训练权重

    Args:
        model: 目标模型
        pretrained_path: 预训练权重路径（支持.ptl和.pth格式）
        strict: 是否严格匹配
    """
    if not pretrained_path.exists():
        logger.warning(f"Pretrained weights not found: {pretrained_path}")
        return model

    try:
        # 尝试加载
        checkpoint = torch.load(pretrained_path, map_location='cpu')

        # 处理不同格式的checkpoint
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # 移除'module.'前缀（如果是多卡训练保存的）
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # 加载权重
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)

        if missing_keys:
            logger.info(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            logger.info(f"Unexpected keys: {unexpected_keys}")

        logger.info(f"Loaded pretrained weights from {pretrained_path}")

    except Exception as e:
        logger.error(f"Failed to load pretrained weights: {e}")
        logger.info("Using ImageNet pretrained weights instead")

    return model


def freeze_encoder(model: nn.Module):
    """冻结编码器（用于微调时固定底层特征）"""
    for name, param in model.named_parameters():
        if 'encoder' in name:
            param.requires_grad = False
    logger.info("Encoder frozen")


def unfreeze_encoder(model: nn.Module):
    """解冻编码器"""
    for param in model.parameters():
        param.requires_grad = True
    logger.info("Encoder unfrozen")