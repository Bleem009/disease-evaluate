#!/usr/bin/env python3
"""
单阶段推理：仅使用叶片分割模型预测叶片掩膜
支持加载真值掩码并计算IoU
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

from configs.stage1_leaf_config import Stage1Config
from src.data.transforms import get_validation_transforms
import segmentation_models_pytorch as smp


def load_stage1_model(config, checkpoint_path, device):
    """加载阶段1模型"""
    model = smp.DeepLabV3Plus(
        encoder_name=config.encoder_name,
        encoder_weights=None,  # 不加载预训练，因为我们要加载自己的权重
        in_channels=3,
        classes=1,
        activation=None
    )
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    # 如果checkpoint包含'model_state_dict'键，则取它；否则直接取state_dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def predict_stage1(model, image_path, transform, device, threshold=0.5):
    """对单张图像进行叶片分割预测"""
    # 读取原始图像（用于最终叠加和保存尺寸）
    orig_image = Image.open(image_path).convert('RGB')
    orig_np = np.array(orig_image)
    h, w = orig_np.shape[:2]

    # 应用验证集变换（resize + 归一化）
    transformed = transform(image=orig_np, mask=np.zeros((h, w), dtype=np.uint8))  # mask占位
    input_tensor = transformed['image'].unsqueeze(0).to(device)  # [1,3,H,W]

    with torch.no_grad():
        logits = model(input_tensor)  # [1,1,H,W]
        probs = torch.sigmoid(logits)  # [0,1]

    # 获取预测掩膜（在变换后的尺寸上）
    mask_resized = (probs[0, 0].cpu().numpy() > threshold).astype(np.uint8)

    # 将掩膜恢复到原始图像尺寸（因为变换可能改变了尺寸）
    from skimage.transform import resize
    mask = resize(mask_resized, (h, w), preserve_range=True, order=0).astype(np.uint8)

    return mask, orig_np


def compute_iou(pred_mask, gt_mask):
    """
    计算IoU（Intersection over Union）
    Args:
        pred_mask: 预测掩码，二值图像（0或1）
        gt_mask: 真值掩码，二值图像（0或1）
    Returns:
        iou: IoU值，范围[0, 1]
    """
    # 确保掩码是二值的
    pred_binary = (pred_mask > 0).astype(np.uint8)
    gt_binary = (gt_mask > 0).astype(np.uint8)

    # 计算交集和并集
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()

    # 避免除零
    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    iou = intersection / union
    return iou


def load_ground_truth_mask(mask_path, target_shape):
    """
    加载真值掩码并调整尺寸
    Args:
        mask_path: 真值掩码路径
        target_shape: 目标尺寸 (h, w)
    Returns:
        gt_mask: 调整后的真值掩码
    """
    if mask_path is None or not Path(mask_path).exists():
        return None

    # 加载真值掩码
    gt_mask = Image.open(mask_path).convert('L')  # 转为灰度图
    gt_mask = np.array(gt_mask)

    # 二值化（假设真值掩码中大于0的值为前景）
    gt_mask = (gt_mask > 0).astype(np.uint8)

    # 调整尺寸以匹配预测掩码
    if gt_mask.shape[:2] != target_shape:
        from skimage.transform import resize
        gt_mask = resize(gt_mask, target_shape, preserve_range=True, order=0).astype(np.uint8)

    return gt_mask


def visualize_result(image, pred_mask, gt_mask=None, severity=None, save_path=None, show=True):
    """
    可视化原始图像与叶片掩膜叠加
    Args:
        image: 原始图像
        pred_mask: 预测掩码
        gt_mask: 真值掩码（可选）
        severity: 病害严重程度（可选）
        save_path: 保存路径
        show: 是否显示
    """
    # 计算IoU（如果提供了真值掩码）
    iou = None
    gt_pixels = None
    if gt_mask is not None:
        iou = compute_iou(pred_mask, gt_mask)
        gt_pixels = np.sum(gt_mask > 0)

    # 创建叠加图
    overlay = image.copy()
    alpha = 0.5
    green = np.array([0, 255, 0])  # RGB绿色

    # 确保 image 是 RGB 格式
    if image.shape[-1] == 3:
        # 创建布尔掩膜
        leaf_area = pred_mask > 0
        # 叠加绿色
        overlay[leaf_area] = overlay[leaf_area] * (1 - alpha) + green * alpha

    # 根据是否有真值掩码决定布局
    if gt_mask is not None:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. 原始图像
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # 2. 预测掩码
    axes[1].imshow(pred_mask, cmap='Greens')
    pred_pixels = np.sum(pred_mask > 0)
    title_pred = f'Predicted Leaf Mask\n{pred_pixels:,} pixels'
    if gt_pixels is not None:
        diff = pred_pixels - gt_pixels
        diff_percent = (diff / gt_pixels * 100) if gt_pixels > 0 else 0
        title_pred += f'\nDiff: {diff:+,} ({diff_percent:+.1f}%)'
    axes[1].set_title(title_pred)
    axes[1].axis('off')

    # 3. 叠加图
    axes[2].imshow(overlay.astype(np.uint8))
    title_overlay = 'Prediction Overlay'
    if severity is not None:
        title_overlay += f' (severity: {severity:.2f}%)'
    if iou is not None:
        title_overlay += f'\nIoU: {iou:.4f}'
    axes[2].set_title(title_overlay)
    axes[2].axis('off')

    # 如果有真值掩码，显示更多对比信息
    if gt_mask is not None:
        # 4. 真值掩码
        axes[3].imshow(gt_mask, cmap='Blues')
        axes[3].set_title(f'Ground Truth Mask\n{gt_pixels:,} pixels')
        axes[3].axis('off')

        # 5. 真值叠加图
        overlay_gt = image.copy()
        gt_area = gt_mask > 0
        blue = np.array([0, 0, 255])  # RGB蓝色
        overlay_gt[gt_area] = overlay_gt[gt_area] * (1 - alpha) + blue * alpha
        axes[4].imshow(overlay_gt.astype(np.uint8))
        axes[4].set_title('Ground Truth Overlay')
        axes[4].axis('off')

        # 6. 差异图（预测 vs 真值）
        # TP: 预测正确（绿色），FP: 假阳性（红色），FN: 假阴性（蓝色）
        diff_vis = image.copy()
        tp = np.logical_and(pred_mask > 0, gt_mask > 0)  # True Positive
        fp = np.logical_and(pred_mask > 0, gt_mask == 0)  # False Positive
        fn = np.logical_and(pred_mask == 0, gt_mask > 0)  # False Negative

        diff_vis[tp] = diff_vis[tp] * 0.5 + np.array([0, 255, 0]) * 0.5  # 绿色
        diff_vis[fp] = diff_vis[fp] * 0.5 + np.array([255, 0, 0]) * 0.5  # 红色
        diff_vis[fn] = diff_vis[fn] * 0.5 + np.array([0, 0, 255]) * 0.5  # 蓝色

        axes[5].imshow(diff_vis.astype(np.uint8))
        tp_pixels = np.sum(tp)
        fp_pixels = np.sum(fp)
        fn_pixels = np.sum(fn)
        axes[5].set_title(f'Difference Map\nTP:{tp_pixels} FP:{fp_pixels} FN:{fn_pixels}\nIoU:{iou:.4f}')
        axes[5].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    if show:
        plt.show()
    else:
        plt.close()

    return fig, iou


def main():
    parser = argparse.ArgumentParser(description='Stage1 Inference: Leaf Segmentation')
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--mask', type=str, default=None,
                        help='Ground truth mask path (optional, for IoU calculation)')
    parser.add_argument('--checkpoint', type=str,
                        default=r'C:\Users\86159\PycharmProjects\disease_evaluation\checkpoints\stage1\best_model.pth',
                        help='Path to stage1 model checkpoint')
    parser.add_argument('--output', type=str, default='outputs/stage1_prediction.png',
                        help='Output visualization path')
    parser.add_argument('--no-display', action='store_true', help='Do not display result')
    parser.add_argument('--threshold', type=float, default=0.5, help='Mask threshold')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 加载配置
    config = Stage1Config()
    # 获取验证集变换（与训练时验证集使用的相同）
    transform = get_validation_transforms(config.img_size)

    # 加载模型
    print("Loading stage1 model...")
    model = load_stage1_model(config, args.checkpoint, device)

    # 预测
    print(f"Processing image: {args.image}")
    pred_mask, image_np = predict_stage1(model, args.image, transform, device, args.threshold)

    # 计算叶片像素数
    pred_pixels = np.sum(pred_mask > 0)
    print(f"Predicted leaf pixels: {pred_pixels:,}")

    # 加载真值掩码（如果提供）
    gt_mask = None
    if args.mask:
        print(f"Loading ground truth mask: {args.mask}")
        gt_mask = load_ground_truth_mask(args.mask, pred_mask.shape[:2])
        if gt_mask is not None:
            gt_pixels = np.sum(gt_mask > 0)
            print(f"Ground truth leaf pixels: {gt_pixels:,}")
            iou = compute_iou(pred_mask, gt_mask)
            print(f"IoU: {iou:.4f}")
        else:
            print("Warning: Could not load ground truth mask")

    # 可视化
    fig, iou = visualize_result(
        image_np,
        pred_mask,
        gt_mask=gt_mask,
        save_path=args.output,
        show=not args.no_display
    )

    print(f"Done. Visualization saved to {args.output}")
    if iou is not None:
        print(f"Final IoU: {iou:.4f}")


if __name__ == '__main__':
    main()