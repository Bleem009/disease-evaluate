#!/usr/bin/env python3
"""
端到端推理脚本：叶片分割 + 病灶分割（标准DeepLabV3+，无分类头）
输入：单张原始叶片图像
输出：原始图像、叶片掩膜、病灶掩膜（在裁剪区域内）、叠加图
"""

import os
import sys
from pathlib import Path

# 设置项目根目录（根据实际情况修改）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.transform import resize
import segmentation_models_pytorch as smp

from src.data.transforms import get_validation_transforms
from configs.stage1_leaf_config import Stage1Config
from configs.stage2_lesion_config import Stage2Config


# ==================== 用户配置（请修改为实际路径）====================
IMG_PATH = r"C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\stage1\test\images\rice_blast_google_0062.jpg"
STAGE1_CKPT = r"C:\Users\86159\PycharmProjects\disease_evaluation\checkpoints\stage1\best_model.pth"
STAGE2_CKPT = r"C:\Users\86159\PycharmProjects\disease_evaluation\checkpoints\stage2\可能输出多类病斑\best_model.pth"  # 标准DeepLabV3+权重
PADDING = 10                    # 裁剪时的边距，与数据准备脚本一致
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ===================================================================

# 类别颜色映射（与训练时一致）
CLASS_NAMES = {
    0: 'Background',
    1: 'Powdery Mildew',
    2: 'Rust',
    3: 'Spot',
    4: 'Bacterial Blight'
}
CLASS_COLORS = {
    0: [0, 0, 0],
    1: [255, 0, 0],
    2: [0, 255, 0],
    3: [0, 0, 255],
    4: [255, 255, 0]
}


# ==================== 模型加载 ====================
def load_stage1_model(checkpoint_path, device):
    config = Stage1Config()
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

def load_stage2_model(checkpoint_path, device):
    """加载标准DeepLabV3+模型（不带分类头）"""
    config = Stage2Config()
    model = smp.DeepLabV3Plus(
        encoder_name=config.encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=config.num_classes,   # 多类别输出
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


# ==================== 图像预处理 ====================
def get_transform(img_size):
    """获取验证集变换（与训练时一致）"""
    return get_validation_transforms(img_size)


# ==================== 推理主函数 ====================
@torch.no_grad()
def inference_single_image(img_path, stage1_model, stage2_model,
                           transform1, transform2, device, padding=10):
    # 读取原始图像
    img_pil = Image.open(img_path).convert('RGB')
    img_orig = np.array(img_pil)
    h, w = img_orig.shape[:2]

    # ---- 阶段1：叶片分割 ----
    transformed1 = transform1(image=img_orig, mask=np.zeros((h, w), dtype=np.uint8))
    input_tensor1 = transformed1['image'].unsqueeze(0).to(device)
    logits1 = stage1_model(input_tensor1)
    prob1 = torch.sigmoid(logits1)[0, 0].cpu().numpy()
    leaf_prob = resize(prob1, (h, w), preserve_range=True, order=1)
    leaf_mask = leaf_prob > 0.5

    # 根据叶片掩膜计算裁剪区域
    y_indices, x_indices = np.where(leaf_mask)
    if len(y_indices) == 0:
        print("警告：未检测到叶片区域，将使用整图")
        y1, y2 = 0, h-1
        x1, x2 = 0, w-1
    else:
        y1, y2 = y_indices.min(), y_indices.max()
        x1, x2 = x_indices.min(), x_indices.max()
    y1 = max(0, y1 - padding)
    y2 = min(h-1, y2 + padding)
    x1 = max(0, x1 - padding)
    x2 = min(w-1, x2 + padding)
    img_cropped = img_orig[y1:y2+1, x1:x2+1].copy()
    crop_h, crop_w = img_cropped.shape[:2]

    # ---- 阶段2：病灶分割（标准模型） ----
    transformed2 = transform2(image=img_cropped, mask=np.zeros((crop_h, crop_w), dtype=np.uint8))
    input_tensor2 = transformed2['image'].unsqueeze(0).to(device)
    logits2 = stage2_model(input_tensor2)          # [1, num_classes, H2, W2]
    prob2 = torch.softmax(logits2, dim=1)[0].cpu().numpy()  # (C, H2, W2)
    pred2 = np.argmax(prob2, axis=0)               # (H2, W2)，正方形，尺寸为 config2.img_size
    # 将预测结果 resize 回裁剪区域的尺寸
    lesion_mask_cropped = resize(pred2, (crop_h, crop_w), preserve_range=True, order=0).astype(np.uint8)

    # 将裁剪区域内的病灶掩膜映射回原图坐标
    lesion_mask_full = np.zeros((h, w), dtype=np.uint8)
    lesion_mask_full[y1:y2+1, x1:x2+1] = lesion_mask_cropped

    return img_orig, leaf_mask, lesion_mask_full, (x1, y1, x2, y2)


# ==================== 可视化 ====================
def visualize_results(img_orig, leaf_mask, lesion_mask, bbox):
    """显示原始图像、叶片掩膜、病灶掩膜、叠加图"""
    overlay = img_orig.copy().astype(np.float32)
    alpha_leaf = 0.3
    alpha_lesion = 0.5

    # 叶片叠加（绿色）
    leaf_overlay = np.zeros_like(overlay)
    leaf_overlay[leaf_mask] = [0, 255, 0]
    overlay = overlay * (1 - alpha_leaf) + leaf_overlay * alpha_leaf

    # 病灶叠加（按类别颜色）
    for cls, color in CLASS_COLORS.items():
        if cls == 0:
            continue
        mask_cls = (lesion_mask == cls)
        if np.any(mask_cls):
            overlay[mask_cls] = overlay[mask_cls] * (1 - alpha_lesion) + np.array(color) * alpha_lesion
    overlay = overlay.astype(np.uint8)

    # 生成彩色病灶掩膜（用于显示）
    colored_lesion = np.zeros((*lesion_mask.shape, 3), dtype=np.uint8)
    for cls, color in CLASS_COLORS.items():
        colored_lesion[lesion_mask == cls] = color

    # 绘制裁剪框
    x1, y1, x2, y2 = bbox
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes[0,0].imshow(img_orig)
    axes[0,0].set_title("Original Image")
    axes[0,0].axis('off')
    # 在原图上绘制裁剪框
    axes[0,0].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2))

    axes[0,1].imshow(leaf_mask, cmap='gray')
    axes[0,1].set_title("Leaf Mask")
    axes[0,1].axis('off')

    axes[1,0].imshow(colored_lesion)
    axes[1,0].set_title("Lesion Mask (Multiclass)")
    axes[1,0].axis('off')

    axes[1,1].imshow(overlay)
    axes[1,1].set_title("Overlay (Leaf+Lesion)")
    axes[1,1].axis('off')

    plt.tight_layout()
    plt.show()


# ==================== 主函数 ====================
def main():
    print(f"Using device: {DEVICE}")

    # 加载配置和变换
    config1 = Stage1Config()
    config2 = Stage2Config()
    transform1 = get_transform(config1.img_size)
    transform2 = get_transform(config2.img_size)

    # 加载模型
    print("Loading stage1 model...")
    model1 = load_stage1_model(STAGE1_CKPT, DEVICE)
    print("Loading stage2 model (standard DeepLabV3+)...")
    model2 = load_stage2_model(STAGE2_CKPT, DEVICE)

    # 推理
    print(f"Processing image: {IMG_PATH}")
    img_orig, leaf_mask, lesion_mask, bbox = inference_single_image(
        IMG_PATH, model1, model2, transform1, transform2, DEVICE, PADDING
    )

    # 可视化
    visualize_results(img_orig, leaf_mask, lesion_mask, bbox)

    # 统计病灶像素
    unique, counts = np.unique(lesion_mask, return_counts=True)
    print("Lesion statistics (pixels):")
    for cls, count in zip(unique, counts):
        if cls != 0:
            print(f"  {CLASS_NAMES[cls]}: {count} pixels")

if __name__ == "__main__":
    main()