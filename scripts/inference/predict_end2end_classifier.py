#!/usr/bin/env python3
"""
端到端推理脚本：叶片分割 + 病灶分割
使用全图病灶掩膜（与原图同尺寸）进行正确对齐的可视化和评估
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.transform import resize
import segmentation_models_pytorch as smp
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.data.transforms import get_validation_transforms
from configs.stage1_leaf_config import Stage1Config
from configs.stage2_lesion_config import Stage2Config

# ==================== 用户配置 ====================
IMG_PATH = r"C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\stage1\test\images\bell_pepper_frogeye_leaf_spot_Google_0049.jpg"
STAGE1_CKPT = r"C:\Users\86159\PycharmProjects\disease_evaluation\checkpoints\stage1\best_model.pth"
STAGE2_CKPT = r"C:\Users\86159\PycharmProjects\disease_evaluation\checkpoints\stage2\best_model.pth"
PADDING = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 全图病灶掩膜目录（请修改为实际路径，存放与原图同尺寸的PNG）
LESION_FULL_MASK_DIR = Path(r"C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\stage2_fullmask\test\labels")

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


# ==================== 阶段2模型定义 ====================
class DeepLabV3PlusWithClassifier(nn.Module):
    def __init__(self, encoder_name, encoder_weights, num_classes, in_channels=3):
        super().__init__()
        self.seg_model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=None
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 224, 224)
            encoder_features = self.seg_model.encoder(dummy)
            if isinstance(encoder_features, (list, tuple)):
                feat = encoder_features[-1]
            else:
                feat = encoder_features
            c = feat.shape[1]
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c, num_classes)

    def forward(self, x):
        seg_out = self.seg_model(x)
        return seg_out


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
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def load_stage2_model(checkpoint_path, device):
    config = Stage2Config()
    model = DeepLabV3PlusWithClassifier(
        encoder_name=config.encoder_name,
        encoder_weights=None,
        num_classes=config.num_classes,
        in_channels=3
    )
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('encoder.') or k.startswith('decoder.') or k.startswith('segmentation_head.'):
            new_state_dict['seg_model.' + k] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def get_transform(img_size):
    return get_validation_transforms(img_size)


# ==================== 推理函数 ====================
@torch.no_grad()
def inference_single_image(img_path, stage1_model, stage2_model,
                           transform1, device, padding=10):
    img_pil = Image.open(img_path).convert('RGB')
    img_orig = np.array(img_pil)
    h, w = img_orig.shape[:2]

    # 阶段1：叶片分割
    transformed1 = transform1(image=img_orig, mask=np.zeros((h, w), dtype=np.uint8))
    input_tensor1 = transformed1['image'].unsqueeze(0).to(device)
    logits1 = stage1_model(input_tensor1)
    prob1 = torch.sigmoid(logits1)[0, 0].cpu().numpy()
    leaf_prob = resize(prob1, (h, w), preserve_range=True, order=1)
    leaf_mask = leaf_prob > 0.5

    # 涂黑背景
    masked = img_orig.copy()
    masked[~leaf_mask] = [0, 0, 0]

    # 裁剪区域
    y_indices, x_indices = np.where(leaf_mask)
    if len(y_indices) == 0:
        print("警告：未检测到叶片区域，将使用整图")
        y1, y2 = 0, h - 1
        x1, x2 = 0, w - 1
    else:
        y1, y2 = y_indices.min(), y_indices.max()
        x1, x2 = x_indices.min(), x_indices.max()
    y1 = max(0, y1 - padding)
    y2 = min(h - 1, y2 + padding)
    x1 = max(0, x1 - padding)
    x2 = min(w - 1, x2 + padding)
    img_cropped = masked[y1:y2 + 1, x1:x2 + 1].copy()
    crop_h, crop_w = img_cropped.shape[:2]
    leaf_mask_cropped = leaf_mask[y1:y2 + 1, x1:x2 + 1].copy()

    print(f"裁剪区域: ({x1}, {y1}) to ({x2}, {y2}), 尺寸: {crop_w}x{crop_h}")

    # 阶段2：病灶分割
    target_size = Stage2Config().img_size
    transform2 = A.Compose([
        A.Resize(target_size, target_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    transformed2 = transform2(image=img_cropped)
    input_tensor2 = transformed2['image'].unsqueeze(0).to(device)
    logits2 = stage2_model(input_tensor2)
    prob2 = torch.softmax(logits2, dim=1)[0].cpu().numpy()
    pred_resized = np.argmax(prob2, axis=0)
    lesion_pred_cropped = cv2.resize(pred_resized.astype(np.uint8), (crop_w, crop_h),
                                     interpolation=cv2.INTER_NEAREST)
    lesion_mask_full = np.zeros((h, w), dtype=np.uint8)
    lesion_mask_full[y1:y2 + 1, x1:x2 + 1] = lesion_pred_cropped

    return (img_orig, leaf_mask, lesion_mask_full,
            lesion_pred_cropped, (x1, y1, x2, y2), (crop_h, crop_w), leaf_mask_cropped)


# ==================== 指标计算 ====================
def compute_pa_and_iou(pred_mask, true_mask, num_classes=2):
    pred_flat = pred_mask.flatten()
    true_flat = true_mask.flatten()
    correct = np.sum(pred_flat == true_flat)
    pa = correct / len(pred_flat)
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred_flat == cls)
        true_cls = (true_flat == cls)
        intersection = np.sum(pred_cls & true_cls)
        union = np.sum(pred_cls | true_cls)
        iou = intersection / (union + 1e-6)
        ious.append(iou)
    return pa, ious


def compute_binary_iou_masked(pred_binary, true_binary, mask):
    """只在mask区域内计算二值IoU"""
    if np.sum(mask) == 0:
        return 0.0
    pred_flat = pred_binary[mask].flatten()
    true_flat = true_binary[mask].flatten()
    intersection = np.sum(pred_flat & true_flat)
    union = np.sum(pred_flat | true_flat)
    return intersection / (union + 1e-6) if union > 0 else 0.0


# ==================== 可视化 + 评估（使用全图真值掩膜）====================
def visualize_and_evaluate(img_orig, leaf_pred, lesion_pred_full,
                           lesion_pred_cropped, bbox, crop_shape, leaf_mask_cropped,
                           leaf_gt_path, lesion_true_full):
    """
    12张图可视化布局（阶段1 + 阶段2）:
    第1行: [原图, 叶片预测, 叶片叠加, 病灶预测, 病灶叠加, 差异图]
    第2行: [原图, 叶片真值, 叶片真值叠加, 病灶真值, 病灶真值叠加, 差异图]
    """
    h, w = img_orig.shape[:2]
    x1, y1, x2, y2 = bbox
    crop_h, crop_w = crop_shape

    # ========== 读取叶片真值 ==========
    leaf_true_pil = Image.open(leaf_gt_path).convert('L')
    if leaf_true_pil.size != (w, h):
        leaf_true_pil = leaf_true_pil.resize((w, h), Image.NEAREST)
    leaf_true = np.array(leaf_true_pil) > 0
    leaf_true_cropped = leaf_true[y1:y2 + 1, x1:x2 + 1]

    # ========== 病灶真值处理 ==========
    if lesion_true_full.shape[:2] != (h, w):
        lesion_true_full = resize(lesion_true_full, (h, w), preserve_range=True, order=0).astype(np.uint8)
    lesion_true_cropped = lesion_true_full[y1:y2 + 1, x1:x2 + 1]
    lesion_true_cropped_binary = lesion_true_cropped > 0

    # ========== 计算指标 ==========
    # 叶片分割指标
    leaf_pred_uint8 = leaf_pred.astype(np.uint8)
    leaf_true_uint8 = leaf_true.astype(np.uint8)
    pa, ious = compute_pa_and_iou(leaf_pred_uint8, leaf_true_uint8, num_classes=2)
    iou_bg, iou_leaf = ious[0], ious[1]

    # 病灶IoU（仅在叶片区域内计算）
    leaf_region_mask = leaf_pred & leaf_true
    pred_bin = lesion_pred_full > 0
    true_bin = lesion_true_full > 0
    iou_lesion = compute_binary_iou_masked(pred_bin, true_bin, leaf_region_mask)
    miou = (iou_bg + iou_leaf + iou_lesion) / 3.0

    # 统计TP/FP/FN（裁剪区域内，仅叶片区域）
    leaf_region_cropped = leaf_mask_cropped & leaf_true_cropped
    pred_bin_crop = lesion_pred_cropped > 0
    true_bin_crop = lesion_true_cropped_binary

    tp_crop = pred_bin_crop & true_bin_crop & leaf_region_cropped
    fp_crop = pred_bin_crop & ~true_bin_crop & leaf_region_cropped
    fn_crop = ~pred_bin_crop & true_bin_crop & leaf_region_cropped

    tp_pixels = np.sum(tp_crop)
    fp_pixels = np.sum(fp_crop)
    fn_pixels = np.sum(fn_crop)

    print("\n" + "=" * 60)
    print("Evaluation Metrics")
    print(f"PA (Pixel Accuracy):           {pa * 100:.2f}%")
    print(f"IoU-Background:                {iou_bg * 100:.2f}%")
    print(f"IoU-Leaf:                      {iou_leaf * 100:.2f}%")
    print(f"IoU-Lesion (in leaf region):   {iou_lesion * 100:.2f}%")
    print(f"mIoU:                          {miou * 100:.2f}%")
    print(f"TP: {tp_pixels}, FP: {fp_pixels}, FN: {fn_pixels}")
    print("=" * 60)

    # ========== 创建可视化元素 ==========
    alpha = 0.5

    # --- 阶段1: 叶片分割可视化 ---

    # 叶片预测叠加图（绿色）
    overlay_leaf_pred = img_orig.copy().astype(np.float32)
    leaf_green = np.array([0, 255, 0])
    overlay_leaf_pred[leaf_pred] = overlay_leaf_pred[leaf_pred] * (1 - alpha) + leaf_green * alpha
    overlay_leaf_pred = overlay_leaf_pred.astype(np.uint8)

    # 叶片真值叠加图（蓝色）
    overlay_leaf_true = img_orig.copy().astype(np.float32)
    leaf_blue = np.array([0, 0, 255])
    overlay_leaf_true[leaf_true] = overlay_leaf_true[leaf_true] * (1 - alpha) + leaf_blue * alpha
    overlay_leaf_true = overlay_leaf_true.astype(np.uint8)

    # 叶片差异图（TP绿/FP红/FN蓝）- 全图尺寸
    leaf_diff_full = np.zeros_like(img_orig)
    leaf_tp = leaf_pred & leaf_true  # 预测正确的叶片（绿色）
    leaf_fp = leaf_pred & ~leaf_true  # 假阳性叶片（红色）
    leaf_fn = ~leaf_pred & leaf_true  # 假阴性叶片（蓝色）
    leaf_diff_full[leaf_tp] = [0, 255, 0]
    leaf_diff_full[leaf_fp] = [255, 0, 0]
    leaf_diff_full[leaf_fn] = [0, 0, 255]

    # --- 阶段2: 病灶分割可视化 ---

    # 病灶预测叠加图（多色）
    overlay_lesion_pred = img_orig.copy().astype(np.float32)
    for cls, color in CLASS_COLORS.items():
        if cls == 0:
            continue
        mask_cls = (lesion_pred_full == cls)
        if np.any(mask_cls):
            overlay_lesion_pred[mask_cls] = overlay_lesion_pred[mask_cls] * (1 - alpha) + np.array(color) * alpha
    overlay_lesion_pred = overlay_lesion_pred.astype(np.uint8)

    # 病灶真值叠加图（多色）
    overlay_lesion_true = img_orig.copy().astype(np.float32)
    for cls, color in CLASS_COLORS.items():
        if cls == 0:
            continue
        mask_cls = (lesion_true_full == cls)
        if np.any(mask_cls):
            overlay_lesion_true[mask_cls] = overlay_lesion_true[mask_cls] * (1 - alpha) + np.array(color) * alpha
    overlay_lesion_true = overlay_lesion_true.astype(np.uint8)

    # 裁剪区域预测彩色图
    colored_pred_cropped = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
    for cls, color in CLASS_COLORS.items():
        colored_pred_cropped[lesion_pred_cropped == cls] = color

    # 裁剪区域真值彩色图
    colored_true_cropped = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
    for cls, color in CLASS_COLORS.items():
        colored_true_cropped[lesion_true_cropped == cls] = color

    # 病灶差异图（裁剪区域）
    lesion_diff_cropped = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
    lesion_diff_cropped[tp_crop] = [0, 255, 0]  # TP: 绿色
    lesion_diff_cropped[fp_crop] = [255, 0, 0]  # FP: 红色
    lesion_diff_cropped[fn_crop] = [0, 0, 255]  # FN: 蓝色

    # ========== 12张图布局 (2行 x 6列) ==========
    fig, axes = plt.subplots(2, 6, figsize=(24, 8))

    # --- 第1行: 预测结果 ---

    # (0,0) 原始图像 + 裁剪框
    axes[0, 0].imshow(img_orig)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    axes[0, 0].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       fill=False, edgecolor='red', linewidth=2))

    # (0,1) 叶片预测掩码
    axes[0, 1].imshow(leaf_pred, cmap='Greens')
    pred_leaf_pixels = np.sum(leaf_pred)
    true_leaf_pixels = np.sum(leaf_true)
    diff_leaf = pred_leaf_pixels - true_leaf_pixels
    diff_leaf_percent = (diff_leaf / true_leaf_pixels * 100) if true_leaf_pixels > 0 else 0
    axes[0, 1].set_title(f'Leaf Prediction\n'
                         f'{pred_leaf_pixels:,} px\n'
                         f'Diff: {diff_leaf:+,} ({diff_leaf_percent:+.1f}%)')
    axes[0, 1].axis('off')

    # (0,2) 叶片预测叠加图
    axes[0, 2].imshow(overlay_leaf_pred)
    axes[0, 2].set_title(f'Leaf Overlay\nIoU: {iou_leaf:.4f}')
    axes[0, 2].axis('off')

    # (0,3) 病灶预测掩码（裁剪区域，彩色）
    axes[0, 3].imshow(colored_pred_cropped)
    pred_lesion_pixels = np.sum(lesion_pred_cropped > 0)
    true_lesion_pixels = np.sum(lesion_true_cropped > 0)
    diff_lesion = pred_lesion_pixels - true_lesion_pixels
    diff_lesion_percent = (diff_lesion / true_lesion_pixels * 100) if true_lesion_pixels > 0 else 0
    axes[0, 3].set_title(f'Lesion Prediction\n'
                         f'{pred_lesion_pixels:,} px\n'
                         f'Diff: {diff_lesion:+,} ({diff_lesion_percent:+.1f}%)')
    axes[0, 3].axis('off')

    # (0,4) 病灶预测叠加图（全图）
    axes[0, 4].imshow(overlay_lesion_pred)
    axes[0, 4].set_title(f'Lesion Overlay\nIoU: {iou_lesion:.4f}')
    axes[0, 4].axis('off')

    # (0,5) 叶片差异图（全图）
    axes[0, 5].imshow(leaf_diff_full)
    leaf_tp_pixels = np.sum(leaf_tp)
    leaf_fp_pixels = np.sum(leaf_fp)
    leaf_fn_pixels = np.sum(leaf_fn)
    axes[0, 5].set_title(f'Leaf Diff\n'
                         f'TP:{leaf_tp_pixels} FP:{leaf_fp_pixels} FN:{leaf_fn_pixels}\n'
                         f'IoU:{iou_leaf:.4f}')
    axes[0, 5].axis('off')

    # --- 第2行: 真值对比 ---

    # (1,0) 原始图像（重复，保持对齐）
    axes[1, 0].imshow(img_orig)
    axes[1, 0].set_title("Original Image")
    axes[1, 0].axis('off')
    axes[1, 0].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       fill=False, edgecolor='red', linewidth=2))

    # (1,1) 叶片真值掩码
    axes[1, 1].imshow(leaf_true, cmap='Blues')
    axes[1, 1].set_title(f'Leaf Ground Truth\n{true_leaf_pixels:,} px')
    axes[1, 1].axis('off')

    # (1,2) 叶片真值叠加图
    axes[1, 2].imshow(overlay_leaf_true)
    axes[1, 2].set_title('Leaf GT Overlay')
    axes[1, 2].axis('off')

    # (1,3) 病灶真值掩码（裁剪区域，彩色）
    axes[1, 3].imshow(colored_true_cropped)
    axes[1, 3].set_title(f'Lesion Ground Truth\n{true_lesion_pixels:,} px')
    axes[1, 3].axis('off')

    # (1,4) 病灶真值叠加图（全图）
    axes[1, 4].imshow(overlay_lesion_true)
    axes[1, 4].set_title('Lesion GT Overlay')
    axes[1, 4].axis('off')

    # (1,5) 病灶差异图（裁剪区域）
    axes[1, 5].imshow(lesion_diff_cropped)
    axes[1, 5].set_title(f'Lesion Diff (cropped)\n'
                         f'TP:{tp_pixels} FP:{fp_pixels} FN:{fn_pixels}\n'
                         f'IoU:{iou_lesion:.4f}')
    axes[1, 5].axis('off')

    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.3)
    plt.show()

    return fig


# ==================== 主函数 ====================
def main():
    print(f"Using device: {DEVICE}")
    config1 = Stage1Config()
    config2 = Stage2Config()
    transform1 = get_transform(config1.img_size)

    print("Loading models...")
    model1 = load_stage1_model(STAGE1_CKPT, DEVICE)
    model2 = load_stage2_model(STAGE2_CKPT, DEVICE)

    print(f"\nProcessing: {IMG_PATH}")
    (img_orig, leaf_mask, lesion_mask_full,
     lesion_pred_cropped, bbox, crop_shape, leaf_mask_cropped) = inference_single_image(
        IMG_PATH, model1, model2, transform1, DEVICE, PADDING
    )

    img_stem = Path(IMG_PATH).stem
    leaf_gt_path = config1.test_label_dir / f"{img_stem}.png"

    # 加载全图病灶真值掩膜（由用户生成）
    lesion_gt_full_path = LESION_FULL_MASK_DIR / f"{img_stem}.png"
    if not lesion_gt_full_path.exists():
        print(f"Error: Full lesion ground truth not found at {lesion_gt_full_path}")
        print("Please generate full-size lesion masks first.")
        return

    lesion_true_full = np.array(Image.open(lesion_gt_full_path).convert('L'))

    if leaf_gt_path.exists() and lesion_gt_full_path.exists():
        visualize_and_evaluate(img_orig, leaf_mask, lesion_mask_full,
                               lesion_pred_cropped, bbox, crop_shape, leaf_mask_cropped,
                               leaf_gt_path, lesion_true_full)
    else:
        print("Ground truth not found, skipping evaluation.")

    unique, counts = np.unique(lesion_pred_cropped, return_counts=True)
    print("\nLesion statistics (cropped region):")
    for cls, count in zip(unique, counts):
        if cls != 0:
            print(f"  {CLASS_NAMES[cls]}: {count} pixels ({100 * count / lesion_pred_cropped.size:.2f}%)")


if __name__ == "__main__":
    main()