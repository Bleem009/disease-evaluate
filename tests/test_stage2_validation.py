#!/usr/bin/env python3
"""
独立测试脚本：使用训练好的多分类模型（如 latest_model.pth）评估测试集
"""

import os
import sys
from pathlib import Path

# 将项目根目录添加到 sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from configs.stage2_lesion_config import Stage2Config
from src.data.datasets import Stage2Dataset
from src.data.transforms import get_validation_transforms


def mean_iou_score(outputs, masks, num_classes, smooth=1e-6):
    """计算平均 IoU（mIoU）"""
    outputs = torch.softmax(outputs, dim=1)
    preds = torch.argmax(outputs, dim=1)
    ious = []
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        mask_cls = (masks == cls)
        intersection = (pred_cls & mask_cls).float().sum((1, 2))
        union = (pred_cls | mask_cls).float().sum((1, 2))
        iou = (intersection + smooth) / (union + smooth)
        ious.append(iou.mean().item())
    return sum(ious) / num_classes


def mean_dice_score(outputs, masks, num_classes, smooth=1e-6):
    """计算平均 Dice（mDice）"""
    outputs = torch.softmax(outputs, dim=1)
    preds = torch.argmax(outputs, dim=1)
    dices = []
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        mask_cls = (masks == cls)
        intersection = (pred_cls & mask_cls).float().sum((1, 2))
        denom = pred_cls.float().sum((1, 2)) + mask_cls.float().sum((1, 2))
        dice = (2 * intersection + smooth) / (denom + smooth)
        dices.append(dice.mean().item())
    return sum(dices) / num_classes


def load_model(config, checkpoint_path, device):
    """加载多分类模型"""
    model = smp.DeepLabV3Plus(
        encoder_name=config.encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=config.num_classes,
        activation=None
    )
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    # 处理不同 checkpoint 格式
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate(model, test_loader, num_classes, device):
    """在测试集上评估模型"""
    model.eval()
    total_loss = 0.0
    total_miou = 0.0
    total_mdice = 0.0
    num_batches = len(test_loader)

    for batch in test_loader:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        outputs = model(images)

        # 计算损失（可选）
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(outputs, masks)
        total_loss += loss.item()

        # 计算指标
        miou = mean_iou_score(outputs, masks, num_classes)
        mdice = mean_dice_score(outputs, masks, num_classes)
        total_miou += miou
        total_mdice += mdice

    avg_loss = total_loss / num_batches
    avg_miou = total_miou / num_batches
    avg_mdice = total_mdice / num_batches

    return {
        'loss': avg_loss,
        'miou': avg_miou,
        'mdice': avg_mdice
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Stage2 Multiclass Model on Test Set')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (e.g., checkpoints/stage2/latest_model.pth)')
    parser.add_argument('--config', type=str, default='stage2_lesion_config',
                        help='Config module name (default: stage2_lesion_config)')
    args = parser.parse_args()

    # 加载配置
    config = Stage2Config()
    if not hasattr(config, 'num_classes'):
        raise AttributeError("Stage2Config must have 'num_classes' attribute")
    num_classes = config.num_classes
    device = torch.device(config.device)

    print("=" * 60)
    print("Stage 2 Multiclass Model Evaluation")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Number of classes: {num_classes}")
    print(f"Device: {device}")
    print("=" * 60)

    # 检查测试集路径
    test_img_dir = getattr(config, 'test_img_dir', None)
    test_label_dir = Path(r"C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\stage2_fullmask\test\labels")
    if not (test_img_dir and test_img_dir.exists() and test_label_dir and test_label_dir.exists()):
        raise FileNotFoundError(f"Test set not found: img={test_img_dir}, label={test_label_dir}")

    # 加载测试集
    print("\nLoading test dataset...")
    test_dataset = Stage2Dataset(
        image_dir=test_img_dir,
        label_dir=test_label_dir,
        transform=get_validation_transforms(config.img_size)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    print(f"Test samples: {len(test_dataset)}")

    # 加载模型
    print("\nLoading model...")
    model = load_model(config, args.checkpoint, device)

    # 评估
    print("\nEvaluating on test set...")
    results = evaluate(model, test_loader, num_classes, device)

    # 输出结果
    print("\n" + "=" * 60)
    print("Test Results:")
    print(f"  Loss: {results['loss']:.4f}")
    print(f"  mIoU: {results['miou']:.4f}")
    print(f"  mDice: {results['mdice']:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()