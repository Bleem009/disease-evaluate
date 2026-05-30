#!/usr/bin/env python3
"""
训练阶段2（多类别版本）
- 支持多类别病灶分割（背景 + K 类病灶）
- 使用独立的训练集和验证集（已预先划分）
- 支持测试集评估
- 使用交叉熵损失，计算 mIoU 和 mDice 指标
"""

import os
# 强制 smp 使用本地缓存，不查询 Hugging Face
os.environ['SMP_ENCODER_WEIGHTS_URL'] = 'None'
os.environ['HF_HUB_OFFLINE'] = '1'

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import numpy as np
import random

from configs.stage2_lesion_config import Stage2Config
from src.data.datasets import Stage2Dataset
from src.data.transforms import get_training_transforms_scratch, get_validation_transforms
from src.training.trainer import SegmentationTrainer


# ===== 修改点 1：删除旧的损失函数导入（不再使用 FocalLoss / DiceBCELoss）=====
# from src.training.losses import DiceBCELoss, FocalLoss
# ===== 修改点 2：新增多分类指标函数定义 =====
def mean_iou_score(outputs, masks, num_classes, smooth=1e-6):
    """
    计算平均 IoU（mIoU）
    outputs: (N, C, H, W) logits
    masks: (N, H, W) 整数标签（0~C-1）
    """
    outputs = torch.softmax(outputs, dim=1)
    preds = torch.argmax(outputs, dim=1)  # (N, H, W)
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
    """
    计算平均 Dice（mDice）
    """
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    config = Stage2Config()
    set_seed(config.seed)

    # ===== 修改点 3：确保配置中有 num_classes =====
    # 请在 Stage2Config 中添加 num_classes 字段，例如 num_classes = 5（背景+4类病灶）
    if not hasattr(config, 'num_classes'):
        raise AttributeError("请在 Stage2Config 中定义 num_classes（总类别数）")
    num_classes = config.num_classes
    print("=" * 60)
    print("Stage 2 Training (Multiclass) FROM SCRATCH")
    print(f"Encoder: {config.encoder_name}")
    print(f"Number of classes: {num_classes}")
    print("=" * 60)

    # ========== 检查数据路径 ==========
    required_dirs = [
        ('train_img_dir', config.train_img_dir),
        ('train_label_dir', config.train_label_dir),
        ('val_img_dir', config.val_img_dir),
        ('val_label_dir', config.val_label_dir),
    ]
    for name, path in required_dirs:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}\nPlease run data preparation scripts first.")

    # 可选测试集路径
    test_img_dir = getattr(config, 'test_img_dir', None)
    test_label_dir = getattr(config, 'test_label_dir', None)
    use_test = test_img_dir and test_img_dir.exists() and test_label_dir and test_label_dir.exists()

    # ========== 创建数据集 ==========
    print("\nLoading datasets...")

    train_dataset = Stage2Dataset(
        image_dir=config.train_img_dir,
        label_dir=config.train_label_dir,
        transform=get_training_transforms_scratch(
            config.img_size,
            config.aug_rotation,
            config.aug_scale
        )
    )

    val_dataset = Stage2Dataset(
        image_dir=config.val_img_dir,
        label_dir=config.val_label_dir,
        transform=get_validation_transforms(config.img_size)
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # ========== DataLoader ==========
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    # ========== 创建模型 ==========
    print(f"\nCreating model: {config.encoder_name}-DeepLabV3+")

    # ===== 修改点 4：模型输出通道数改为 num_classes =====
    model = smp.DeepLabV3Plus(
        encoder_name=config.encoder_name,
        encoder_weights='imagenet',
        in_channels=3,
        classes=num_classes,          # 修改为多类别
        activation=None
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # ========== 优化器 ==========
    encoder_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'encoder' in name:
                encoder_params.append(param)
            else:
                decoder_params.append(param)

    optimizer = optim.SGD([
        {'params': encoder_params, 'lr': config.lr_encoder},
        {'params': decoder_params, 'lr': config.lr_decoder}
    ], momentum=config.momentum, weight_decay=config.weight_decay)

    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: (1 - epoch / config.num_epochs) ** config.lr_power
    )

    # ===== 损失函数为 CrossEntropyLoss =====
    # 如果类别不平衡，可设置权重，例如：
    # class_weights = torch.tensor([0.1, 1.0, 1.0, ...]).to(config.device)
    # criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    criterion = torch.nn.CrossEntropyLoss()
    print("\nUsing CrossEntropyLoss")

    # ===== 模型指标为 mIoU 和 mDice =====
    metrics = {
        'miou': lambda outputs, masks: mean_iou_score(outputs, masks, num_classes),
        'mdice': lambda outputs, masks: mean_dice_score(outputs, masks, num_classes)
    }

    # ========== 训练器 ==========
    trainer = SegmentationTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=metrics,
        best_metric_name='miou'
    )

    # ========== 开始训练 ==========
    print("\n" + "=" * 60)
    print("Starting Stage 2 Multiclass Training")
    print("=" * 60)
    trainer.train()

    # ========== 测试集评估（可选）==========
    if use_test:
        print("\n" + "=" * 60)
        print("Evaluating on test set...")
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
        # 加载最佳模型
        best_checkpoint = torch.load(trainer.best_model_path)
        trainer.model.load_state_dict(best_checkpoint['model_state_dict'])
        if hasattr(trainer, 'evaluate'):
            test_metrics = trainer.evaluate(test_loader)
            print("\nTest results:")
            for name, value in test_metrics.items():
                print(f"  {name}: {value:.4f}")
        else:
            print("Warning: trainer.evaluate() not found, skipping test evaluation.")
            print("You can manually compute test metrics using a separate script.")

    print("\nTraining completed!")
    print(f"Best model: {config.output_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()