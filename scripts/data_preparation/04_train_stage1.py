# scripts/data_preparation/04_train_stage1_scratch.py
# !/usr/bin/env python3
"""
训练阶段1（从头训练版本）
- 仅使用ImageNet预训练
- 不使用论文的StripeRustNet权重
- 更强的数据增强
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import os
# 强制 smp 使用本地缓存，不查询 Hugging Face
os.environ['SMP_ENCODER_WEIGHTS_URL'] = 'None'
os.environ['HF_HUB_OFFLINE'] = '1'

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import numpy as np
import random

from configs.stage1_leaf_config import Stage1Config
from src.data.datasets import Stage1Dataset
from src.data.transforms import get_training_transforms_scratch, get_validation_transforms
from src.training.trainer import SegmentationTrainer
from src.training.losses import DiceBCELoss, FocalLoss  # 可以尝试Focal Loss
from src.training.metrics import iou_score, pixel_accuracy, dice_score


def set_seed(seed):
    """设置随机种子确保可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    config = Stage1Config()
    set_seed(config.seed)

    print("=" * 60)
    print("Stage 1 Training FROM SCRATCH (ImageNet pretrain only)")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Encoder: {config.encoder_name}")
    print(f"Epochs: {config.num_epochs}")
    print("=" * 60)

    # ========== 定义已划分好的数据集路径 ==========
    # 您可以根据实际路径修改，或将这些路径添加到 Stage1Config 中
    train_img_dir = Path(r"C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\stage1\train\images")
    train_label_dir = Path(r"C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\stage1\train\labels")
    val_img_dir = Path(r"C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\stage1\val\images")
    val_label_dir = Path(r"C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\stage1\val\labels")
    test_img_dir = Path(r"C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\stage1\test\images")
    test_label_dir = Path(r"C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\stage1\test\labels")

    # 检查路径是否存在
    for d in [train_img_dir, train_label_dir, val_img_dir, val_label_dir, test_img_dir, test_label_dir]:
        if not d.exists():
            raise FileNotFoundError(f"Directory not found: {d}")

    # ========== 创建数据集（训练集用强增强，验证/测试集用基础变换）==========
    print("\nLoading datasets...")

    train_dataset = Stage1Dataset(
        image_dir=train_img_dir,
        label_dir=train_label_dir,
        transform=get_training_transforms_scratch(
            config.img_size,
            config.aug_rotation,
            config.aug_scale
        ),
        cache_images=False
    )

    val_dataset = Stage1Dataset(
        image_dir=val_img_dir,
        label_dir=val_label_dir,
        transform=get_validation_transforms(config.img_size),
        cache_images=False
    )

    test_dataset = Stage1Dataset(
        image_dir=test_img_dir,
        label_dir=test_label_dir,
        transform=get_validation_transforms(config.img_size),
        cache_images=False
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    if len(train_dataset) < 50:
        print("WARNING: Very small training set! Consider using heavier augmentation.")

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

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    # ========== 创建模型（与原来相同）==========
    print(f"\nCreating model: {config.encoder_name}-DeepLabV3+")
    model = smp.DeepLabV3Plus(
        encoder_name=config.encoder_name,
        encoder_weights='imagenet' if config.use_imagenet_pretrained else None,
        in_channels=3,
        classes=1,
        activation=None
    )
    # 统计参数量等（略）

    # ========== 优化器设置（与原来相同）==========
    encoder_params, decoder_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
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

    # ========== 损失函数（与原来相同）==========
    use_focal = False
    if use_focal:
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        print("\nUsing Focal Loss")
    else:
        criterion = DiceBCELoss(dice_weight=1.0, bce_weight=1.0)
        print("\nUsing Dice + BCE Loss")

    metrics = {
        'iou': iou_score,
        'dice': dice_score,
        'pixel_acc': pixel_accuracy
    }

    # ========== 创建训练器 ==========
    trainer = SegmentationTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=metrics
    )

    # ========== 开始训练 ==========
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        torch.save({
            'epoch': trainer.current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, config.output_dir / "interrupted.pth")
        print(f"Saved interrupted checkpoint to {config.output_dir / 'interrupted.pth'}")

    # ========== 训练完成后，加载最佳模型并评估测试集 ==========
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    # 加载最佳模型权重
    best_checkpoint = torch.load(trainer.best_model_path)
    trainer.model.load_state_dict(best_checkpoint['model_state_dict'])
    # 评估测试集
    test_metrics = trainer.evaluate(test_loader)  # 需要trainer有evaluate方法
    print("Test results:")
    for name, value in test_metrics.items():
        print(f"  {name}: {value:.4f}")
    print("=" * 60)

    # 打印训练中的最佳验证结果（可选）
    if trainer.history:
        best_epoch = max(trainer.history, key=lambda x: x['val']['iou'])
        print(f"\nBest validation result (Epoch {best_epoch['epoch']}):")
        print(f"  Val IoU: {best_epoch['val']['iou']:.4f}")
        print(f"  Val Loss: {best_epoch['val']['loss']:.4f}")


if __name__ == "__main__":
    main()
