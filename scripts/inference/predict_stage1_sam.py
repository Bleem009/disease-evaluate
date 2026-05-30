#!/usr/bin/env python3
"""
评估 SAM 模型在叶片分割测试集上的性能
支持多种掩膜选择策略：
- 'largest': 面积最大（默认）
- 'quality': 自定义质量评估（边界梯度、内部一致性等）
- 'mcp_score': 使用 mcp_tools.py 中的颜色方差+位置得分策略
"""

import sys
from pathlib import Path

import torch

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2
from skimage.measure import regionprops


# ---------- 自定义质量评估函数（可选）----------
def compute_boundary_gradient(mask, image):
    if mask.sum() == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel)
    boundary = (dilated - mask).astype(bool)
    if boundary.sum() == 0:
        return 0.0
    return float(grad_mag[boundary].mean())

def compute_internal_consistency(mask, image):
    if mask.sum() < 2:
        return 1e6
    pixels = image[mask]
    var = np.var(pixels, axis=0).sum() / (255*255*3)
    return float(var)

def compute_edge_contrast(mask, image):
    if mask.sum() == 0 or mask.sum() == mask.size:
        return 0.0
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel)
    eroded = cv2.erode(mask.astype(np.uint8), kernel)
    boundary_outer = (dilated - mask).astype(bool)
    boundary_inner = (mask - eroded).astype(bool)
    if boundary_inner.sum() == 0 or boundary_outer.sum() == 0:
        return 0.0
    mean_inner = image[boundary_inner].mean(axis=0)
    mean_outer = image[boundary_outer].mean(axis=0)
    return float(np.linalg.norm(mean_inner - mean_outer))

def compute_shape_compactness(mask):
    if mask.sum() == 0:
        return 0.0
    props = regionprops(mask.astype(np.uint8))[0]
    perimeter = props.perimeter
    area = props.area
    if area == 0:
        return 0.0
    compactness = (perimeter * perimeter) / (4 * np.pi * area)
    return 1.0 / (compactness + 1e-6)

def evaluate_mask_quality(mask, image):
    grad = compute_boundary_gradient(mask, image)
    internal_var = compute_internal_consistency(mask, image)
    contrast = compute_edge_contrast(mask, image)
    compact = compute_shape_compactness(mask)
    norm_grad = min(grad / 50.0, 1.0) if grad > 0 else 0.0
    norm_contrast = min(contrast / 50.0, 1.0) if contrast > 0 else 0.0
    norm_internal = 1.0 - min(internal_var, 1.0)
    norm_compact = min(compact, 1.0)
    area_ratio = mask.sum() / (image.shape[0] * image.shape[1])
    total = (0.1 * norm_grad + 0.1 * norm_contrast + 0.1 * norm_internal + 0.1 * norm_compact + 0.6 * area_ratio)
    return total


# ---------- MCP 风格评分（颜色方差 + 位置得分）----------
def compute_mcp_score(mask, image):
    """
    完全复刻 mcp_tools.py 中的评分逻辑
    返回总分，越高越好
    """
    # 颜色方差
    masked_pixels = image[mask]
    if len(masked_pixels) < 100:
        color_var = 0
    else:
        color_var = np.var(masked_pixels, axis=0).sum()
    color_var_norm = min(color_var / (255 * 255 * 3), 1.0)

    # 位置得分（越靠近图像中心越高）
    h, w = mask.shape
    center_y, center_x = h // 2, w // 2
    y_indices, x_indices = np.where(mask)
    if len(y_indices) == 0:
        center_dist = 1e6
    else:
        centroid_y = np.mean(y_indices)
        centroid_x = np.mean(x_indices)
        center_dist = np.sqrt((centroid_y - center_y) ** 2 + (centroid_x - center_x) ** 2)
    max_dist = np.sqrt(h ** 2 + w ** 2) / 2
    position_score = 1.0 - min(center_dist / max_dist, 1.0)

    total_score = 0.5 * color_var_norm + 0.5 * position_score
    return total_score


# ---------- SAM 模型加载 ----------
def load_sam_model(checkpoint_path, model_type="vit_b", device="cuda"):
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator


# ---------- 掩膜预测（支持不同选择策略）----------
def predict_leaf_mask_sam(image_np, mask_generator, strategy='largest'):
    """
    参数:
        image_np: RGB图像
        mask_generator: SAM mask generator
        strategy: 'largest', 'quality', 'mcp_score'
    返回:
        二值掩膜 (H, W) bool
    """
    masks = mask_generator.generate(image_np)
    if not masks:
        return np.zeros(image_np.shape[:2], dtype=bool)

    if strategy == 'largest':
        # 选择面积最大的掩膜
        largest = max(masks, key=lambda x: x['area'])
        return largest['segmentation']

    elif strategy == 'quality':
        # 使用自定义质量评估（基于边界梯度、内部一致性、边缘对比度、紧凑度、面积占比）
        best_mask = None
        best_score = -1.0
        for ann in masks:
            mask = ann['segmentation']
            area_ratio = ann['area'] / (image_np.shape[0] * image_np.shape[1])
            if area_ratio < 0.01:
                continue
            score = evaluate_mask_quality(mask, image_np)
            if score > best_score:
                best_score = score
                best_mask = mask
        if best_mask is None:
            # 如果所有掩膜都太小，退回面积最大
            largest = max(masks, key=lambda x: x['area'])
            best_mask = largest['segmentation']
        return best_mask

    elif strategy == 'mcp_score':
        # 完全复刻 mcp_tools.py 的评分逻辑
        best_mask = None
        best_score = -1.0
        for ann in masks:
            mask = ann['segmentation']
            # 可选：过滤面积过小的掩膜（与原始 mcp_tools 一致，不过滤）
            score = compute_mcp_score(mask, image_np)
            if score > best_score:
                best_score = score
                best_mask = mask
        # 如果所有得分都是0（理论上不会），回退到面积最大
        if best_mask is None:
            largest = max(masks, key=lambda x: x['area'])
            best_mask = largest['segmentation']
        return best_mask

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ---------- 指标计算 ----------
def compute_iou(pred, true):
    inter = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    return inter / (union + 1e-6)

def compute_dice(pred, true):
    inter = np.logical_and(pred, true).sum()
    return 2 * inter / (pred.sum() + true.sum() + 1e-6)

def compute_pixel_accuracy(pred, true):
    correct = (pred == true).sum()
    return correct / pred.size


# ---------- 主函数 ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sam_checkpoint', type=str, default=r'D:\edge_download\sam_vit_b_01ec64.pth')
    parser.add_argument('--test_img_dir', type=str,
                        default=r'C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\stage1\test\images')
    parser.add_argument('--test_label_dir', type=str,
                        default=r'C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\stage1\test\labels')
    parser.add_argument('--model_type', type=str, default='vit_b')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--strategy', type=str, default='largest',
                        choices=['largest', 'quality', 'mcp_score'],
                        help='Mask selection strategy: largest area, custom quality, or MCP color+position score')
    args = parser.parse_args()

    print(f"Using device: {args.device}")
    print(f"Mask selection strategy: {args.strategy}")
    mask_generator = load_sam_model(args.sam_checkpoint, args.model_type, args.device)

    img_dir = Path(args.test_img_dir)
    label_dir = Path(args.test_label_dir)
    img_files = sorted([f for f in img_dir.glob("*") if f.suffix.lower() in ['.jpg','.jpeg','.png']])
    valid_pairs = []
    for f in img_files:
        label_f = label_dir / f"{f.stem}.png"
        if label_f.exists():
            valid_pairs.append((f, label_f))
    print(f"Found {len(valid_pairs)} test images")

    total_iou = 0.0
    total_dice = 0.0
    total_pa = 0.0

    for img_path, label_path in tqdm(valid_pairs, desc="Evaluating SAM"):
        image_np = np.array(Image.open(img_path).convert('RGB'))
        true_mask = np.array(Image.open(label_path).convert('L')) > 127
        pred_mask = predict_leaf_mask_sam(image_np, mask_generator, args.strategy)

        iou = compute_iou(pred_mask, true_mask)
        dice = compute_dice(pred_mask, true_mask)
        pa = compute_pixel_accuracy(pred_mask, true_mask)

        total_iou += iou
        total_dice += dice
        total_pa += pa

    n = len(valid_pairs)
    print("\n" + "="*50)
    print(f"SAM Evaluation (strategy: {args.strategy})")
    print(f"IoU       : {total_iou/n:.4f}")
    print(f"Dice      : {total_dice/n:.4f}")
    print(f"Pixel Acc : {total_pa/n:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()