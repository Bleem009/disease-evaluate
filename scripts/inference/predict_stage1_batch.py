#!/usr/bin/env python3
"""
评估阶段1（叶片分割）模型在测试集上的性能
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from configs.stage1_leaf_config import Stage1Config
from src.data.datasets import Stage1Dataset
from src.data.transforms import get_validation_transforms
from src.training.metrics import iou_score, dice_score, pixel_accuracy
import segmentation_models_pytorch as smp


def load_model(config, checkpoint_path, device):
    """加载阶段1模型"""
    model = smp.DeepLabV3Plus(
        encoder_name=config.encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    )
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate(model, dataloader, device):
    """在测试集上评估模型，返回各项指标的平均值"""
    model.eval()
    total_iou = 0.0
    total_dice = 0.0
    total_pixel_acc = 0.0
    num_batches = len(dataloader)

    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device).float()  # 形状 [B, H, W]，值 0/1

        # 前向传播得到 logits
        logits = model(images)  # 形状 [B, 1, H, W]

        # 计算指标（直接传入 logits 和 masks）
        iou = iou_score(logits, masks)          # 返回 Python float
        dice = dice_score(logits, masks)         # 返回 Python float
        pixel_acc = pixel_accuracy(logits, masks) # 返回 Python float

        total_iou += iou
        total_dice += dice
        total_pixel_acc += pixel_acc

    avg_iou = total_iou / num_batches
    avg_dice = total_dice / num_batches
    avg_pixel_acc = total_pixel_acc / num_batches

    return {
        'iou': avg_iou,
        'dice': avg_dice,
        'pixel_acc': avg_pixel_acc,
        'num_batches': num_batches
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Stage1 (Leaf Segmentation) on test set')
    parser.add_argument('--checkpoint', type=str,
                        default=r'C:\Users\86159\PycharmProjects\disease_evaluation\checkpoints\stage1\best_model.pth',
                        help='Path to stage1 model checkpoint')
    parser.add_argument('--test_img_dir', type=str,
                        default=r'C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\stage1\test\images',
                        help='Test images directory')
    parser.add_argument('--test_label_dir', type=str,
                        default=r'C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\stage1\test\labels',
                        help='Test labels directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 加载配置
    config = Stage1Config()

    # 加载模型
    print("Loading model...")
    model = load_model(config, args.checkpoint, device)

    # 创建测试数据集和 DataLoader
    test_dataset = Stage1Dataset(
        image_dir=args.test_img_dir,
        label_dir=args.test_label_dir,
        transform=get_validation_transforms(config.img_size),
        cache_images=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Test set size: {len(test_dataset)} images")
    print(f"Number of batches: {len(test_loader)}")

    # 评估
    results = evaluate(model, test_loader, device)

    # 打印结果
    print("\n" + "=" * 50)
    print("Test Set Evaluation Results")
    print("=" * 50)
    print(f"IoU         : {results['iou']:.4f}")
    print(f"Dice        : {results['dice']:.4f}")
    print(f"Pixel Acc   : {results['pixel_acc']:.4f}")
    print(f"Num batches : {results['num_batches']}")
    print("=" * 50)


if __name__ == '__main__':
    main()