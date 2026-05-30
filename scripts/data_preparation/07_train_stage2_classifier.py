#!/usr/bin/env python3
"""
训练阶段2（多类别 + 全局分类约束）
- 强制模型学习单张图片只含一种病害
- 分割损失 + 分类损失联合训练
"""

import os
os.environ['SMP_ENCODER_WEIGHTS_URL'] = 'None'
os.environ['HF_HUB_OFFLINE'] = '1'

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
import numpy as np
import random
from tqdm import tqdm
import time
import json
from torch.utils.tensorboard import SummaryWriter

from configs.stage2_lesion_config import Stage2Config
from src.data.datasets import Stage2Dataset
from src.data.transforms import get_training_transforms_scratch, get_validation_transforms
from src.utils.logger import setup_logger


# ==================== 带分类头的模型 ====================
class DeepLabV3PlusWithClassifier(nn.Module):
    """DeepLabV3+ 分割模型 + 全局分类头"""
    def __init__(self, encoder_name, encoder_weights, num_classes, in_channels=3):
        super().__init__()
        self.seg_model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=None
        )
        # 获取编码器输出通道数（不同 encoder 不同，这里以 resnet50 为例，输出 2048）
        # 动态获取：先跑一次 dummy 输入
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 224, 224)
            encoder_features = self.seg_model.encoder(dummy)
            # encoder 返回的是一个列表，取最后一层特征
            if isinstance(encoder_features, (list, tuple)):
                feat = encoder_features[-1]
            else:
                feat = encoder_features
            c = feat.shape[1]
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c, num_classes)

    def forward(self, x):
        # 分割输出
        seg_out = self.seg_model(x)
        # 提取编码器最后一层特征用于分类
        encoder_features = self.seg_model.encoder(x)
        if isinstance(encoder_features, (list, tuple)):
            feat = encoder_features[-1]
        else:
            feat = encoder_features
        pooled = self.global_avg_pool(feat).flatten(1)
        cls_out = self.classifier(pooled)
        return seg_out, cls_out


# ==================== 指标函数 ====================
def mean_iou_score(outputs, masks, num_classes, smooth=1e-6):
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


# ==================== 数据集包装，返回整图标签 ====================
class Stage2DatasetWithLabel(Stage2Dataset):
    """在原有数据集基础上，返回整图的病害类别标签（从掩膜中统计）"""
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        mask = data['mask']  # (H, W) long tensor
        # 统计非背景像素最多的类别
        values, counts = torch.unique(mask, return_counts=True)
        # 排除背景 0
        non_bg_mask = values != 0
        if non_bg_mask.any():
            non_bg_values = values[non_bg_mask]
            non_bg_counts = counts[non_bg_mask]
            cls_label = non_bg_values[torch.argmax(non_bg_counts)].item()
        else:
            cls_label = 0  # 无病灶
        data['cls_label'] = torch.tensor(cls_label, dtype=torch.long)
        return data


# ==================== 训练器 ====================
class Trainer:
    def __init__(self, model, config, train_loader, val_loader, criterion_seg, criterion_cls, optimizer, scheduler, device):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion_seg = criterion_seg
        self.criterion_cls = criterion_cls
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.logger = setup_logger("trainer", config.log_dir / "train.log")
        self.writer = SummaryWriter(config.log_dir / "tensorboard")
        self.current_epoch = 0
        self.best_metric = 0.0
        self.patience_counter = 0
        self.history = []
        self.best_model_path = config.output_dir / "best_model.pth"
        self.latest_model_path = config.output_dir / "latest_model.pth"

    def train_epoch(self):
        self.model.train()
        total_seg_loss = 0.0
        total_cls_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        for batch in pbar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            cls_labels = batch['cls_label'].to(self.device)

            self.optimizer.zero_grad()
            seg_out, cls_out = self.model(images)
            loss_seg = self.criterion_seg(seg_out, masks)
            loss_cls = self.criterion_cls(cls_out, cls_labels)
            loss = loss_seg + 0.5 * loss_cls  # 分类损失权重可调
            loss.backward()
            self.optimizer.step()

            total_seg_loss += loss_seg.item()
            total_cls_loss += loss_cls.item()
            pbar.set_postfix({'seg_loss': loss_seg.item(), 'cls_loss': loss_cls.item()})

        num_batches = len(self.train_loader)
        return {'seg_loss': total_seg_loss / num_batches, 'cls_loss': total_cls_loss / num_batches}

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_seg_loss = 0.0
        total_cls_loss = 0.0
        total_miou = 0.0
        total_mdice = 0.0
        total_cls_acc = 0.0
        num_batches = len(self.val_loader)

        for batch in tqdm(self.val_loader, desc="Validation"):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            cls_labels = batch['cls_label'].to(self.device)

            seg_out, cls_out = self.model(images)
            loss_seg = self.criterion_seg(seg_out, masks)
            loss_cls = self.criterion_cls(cls_out, cls_labels)
            total_seg_loss += loss_seg.item()
            total_cls_loss += loss_cls.item()

            # 分割指标
            miou = mean_iou_score(seg_out, masks, self.config.num_classes)
            mdice = mean_dice_score(seg_out, masks, self.config.num_classes)
            total_miou += miou
            total_mdice += mdice

            # 分类准确率
            pred_cls = torch.argmax(cls_out, dim=1)
            acc = (pred_cls == cls_labels).float().mean().item()
            total_cls_acc += acc

        return {
            'seg_loss': total_seg_loss / num_batches,
            'cls_loss': total_cls_loss / num_batches,
            'miou': total_miou / num_batches,
            'mdice': total_mdice / num_batches,
            'cls_acc': total_cls_acc / num_batches
        }

    def _save_checkpoint(self, val_results):
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_results': val_results,
        }
        torch.save(checkpoint, self.latest_model_path)

        current_metric = val_results.get('miou', val_results['seg_loss'])
        # 注意：这里以 miou 为最佳指标，越大越好
        is_better = current_metric > self.best_metric
        if is_better:
            self.best_metric = current_metric
            torch.save(checkpoint, self.best_model_path)
            self.logger.info(f"  -> Saved best model (miou: {current_metric:.4f})")
            self.patience_counter = 0
        else:
            self.patience_counter += 1

    def train(self):
        self.logger.info(f"Starting training for {self.config.num_epochs} epochs")
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch + 1
            start_time = time.time()
            train_res = self.train_epoch()
            val_res = self.validate()
            epoch_time = time.time() - start_time

            # 记录
            self.logger.info(
                f"Epoch {self.current_epoch}/{self.config.num_epochs} | Time: {epoch_time:.1f}s | "
                f"Train seg_loss: {train_res['seg_loss']:.4f} cls_loss: {train_res['cls_loss']:.4f} | "
                f"Val seg_loss: {val_res['seg_loss']:.4f} cls_loss: {val_res['cls_loss']:.4f} | "
                f"miou: {val_res['miou']:.4f} mdice: {val_res['mdice']:.4f} cls_acc: {val_res['cls_acc']:.4f}"
            )
            self.writer.add_scalar('Loss/train_seg', train_res['seg_loss'], self.current_epoch)
            self.writer.add_scalar('Loss/train_cls', train_res['cls_loss'], self.current_epoch)
            self.writer.add_scalar('Loss/val_seg', val_res['seg_loss'], self.current_epoch)
            self.writer.add_scalar('Loss/val_cls', val_res['cls_loss'], self.current_epoch)
            self.writer.add_scalar('Metrics/miou', val_res['miou'], self.current_epoch)
            self.writer.add_scalar('Metrics/mdice', val_res['mdice'], self.current_epoch)
            self.writer.add_scalar('Metrics/cls_acc', val_res['cls_acc'], self.current_epoch)

            self._save_checkpoint(val_res)
            if self.scheduler:
                self.scheduler.step()
            if self.patience_counter >= self.config.early_stop_patience:
                self.logger.info("Early stopping triggered")
                break

        self.writer.close()
        self.logger.info("Training completed!")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    config = Stage2Config()
    set_seed(config.seed)

    if not hasattr(config, 'num_classes'):
        raise AttributeError("请在 Stage2Config 中定义 num_classes")
    num_classes = config.num_classes
    device = torch.device(config.device)

    print("=" * 60)
    print("Stage 2 Training (Multiclass + Global Classification)")
    print(f"Encoder: {config.encoder_name}")
    print(f"Number of classes: {num_classes}")
    print("=" * 60)

    # 数据路径检查
    required_dirs = [
        ('train_img_dir', config.train_img_dir),
        ('train_label_dir', config.train_label_dir),
        ('val_img_dir', config.val_img_dir),
        ('val_label_dir', config.val_label_dir),
    ]
    for name, path in required_dirs:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")

    # 数据集（使用带标签的版本）
    print("\nLoading datasets...")
    train_dataset = Stage2DatasetWithLabel(
        image_dir=config.train_img_dir,
        label_dir=config.train_label_dir,
        transform=get_training_transforms_scratch(
            config.img_size, config.aug_rotation, config.aug_scale
        )
    )
    val_dataset = Stage2DatasetWithLabel(
        image_dir=config.val_img_dir,
        label_dir=config.val_label_dir,
        transform=get_validation_transforms(config.img_size)
    )
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                              num_workers=config.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False,
                            num_workers=config.num_workers, pin_memory=True)

    # 模型
    print(f"\nCreating model: {config.encoder_name}-DeepLabV3+ with classifier")
    model = DeepLabV3PlusWithClassifier(
        encoder_name=config.encoder_name,
        encoder_weights='imagenet',
        num_classes=num_classes,
        in_channels=3
    )
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # 优化器（可沿用之前的策略）
    encoder_params = [p for n, p in model.named_parameters() if 'seg_model.encoder' in n]
    other_params = [p for n, p in model.named_parameters() if 'seg_model.encoder' not in n]
    optimizer = optim.SGD([
        {'params': encoder_params, 'lr': config.lr_encoder},
        {'params': other_params, 'lr': config.lr_decoder}
    ], momentum=config.momentum, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: (1 - epoch / config.num_epochs) ** config.lr_power
    )

    criterion_seg = nn.CrossEntropyLoss()
    criterion_cls = nn.CrossEntropyLoss()
    print("\nUsing CrossEntropyLoss for both segmentation and classification")

    # 训练
    trainer = Trainer(model, config, train_loader, val_loader,
                      criterion_seg, criterion_cls, optimizer, scheduler, device)
    trainer.train()

    print(f"\nTraining completed! Best model saved to {config.output_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()