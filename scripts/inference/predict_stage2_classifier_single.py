#!/usr/bin/env python3
"""
阶段2单模型推理脚本：
- 输入：根据真值叶片掩膜裁剪后的图片
- 输出：阶段2病灶分割结果 + 6张图可视化
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
import segmentation_models_pytorch as smp
import cv2

from src.data.transforms import get_validation_transforms
from configs.stage2_lesion_config import Stage2Config

# ==================== 用户配置 ====================
# 裁剪后的图像路径（已根据叶片真值掩膜裁剪）
CROPPED_IMG_PATH = r"C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\stage2\train\images\rice_blast_2.png"
# 对应的病灶真值掩膜（与裁剪图同尺寸）
LESION_GT_PATH = r"C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\stage2\train\labels\rice_blast_2.png"
# 阶段2模型检查点
STAGE2_CKPT = r"C:\Users\86159\PycharmProjects\disease_evaluation\checkpoints\stage2\best_model.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = {
    0: 'Background',
    1: 'Powdery Mildew',
    2: 'Rust',
    3: 'Spot',
    4: 'Bacterial Blight'
}
CLASS_COLORS = {
    0: [0, 0, 0],
    1: [255, 0, 0],  # 红色 - Powdery Mildew
    2: [0, 255, 0],  # 绿色 - Rust
    3: [0, 0, 255],  # 蓝色 - Spot
    4: [255, 255, 0]  # 黄色 - Bacterial Blight
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
def load_stage2_model(checkpoint_path, device):
    config = Stage2Config()
    model = DeepLabV3PlusWithClassifier(
        encoder_name=config.encoder_name,
        encoder_weights=None,
        num_classes=config.num_classes,
        in_channels=3
    )
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('encoder.') or k.startswith('decoder.') or k.startswith('segmentation_head.'):
            new_state_dict['seg_model.' + k] = v
        elif k.startswith('classifier.'):
            new_state_dict[k] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


# ==================== 辅助函数 ====================
def resize_with_pad(image, target_size, pad_value=0):
    """保持长宽比resize+padding"""
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(image, (new_w, new_h),
                         interpolation=cv2.INTER_LINEAR if len(image.shape) == 3 else cv2.INTER_NEAREST)

    pad_top = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left

    if len(image.shape) == 3:
        padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                    cv2.BORDER_CONSTANT, value=(pad_value, pad_value, pad_value))
    else:
        padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                    cv2.BORDER_CONSTANT, value=pad_value)
    return padded, (scale, pad_top, pad_left, (h, w))


def remove_pad_and_resize(padded_mask, pad_info, original_size):
    """去除padding并resize回原始尺寸"""
    scale, pad_top, pad_left, (orig_h, orig_w) = pad_info
    h_crop = int(orig_h * scale)
    w_crop = int(orig_w * scale)
    cropped = padded_mask[pad_top:pad_top + h_crop, pad_left:pad_left + w_crop]
    resized = cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return resized


# ==================== 阶段2推理 ====================
@torch.no_grad()
def inference_stage2(cropped_img_path, stage2_model, device):
    """
    对裁剪后的叶片图像进行病灶分割推理
    """
    # 读取裁剪后的图像
    img_pil = Image.open(cropped_img_path).convert('RGB')
    img_orig = np.array(img_pil)
    h, w = img_orig.shape[:2]

    print(f"Input image size: {w}x{h}")

    # 获取阶段2配置
    config = Stage2Config()
    target_size = config.img_size

    # Resize + Padding（保持长宽比）
    img_padded, pad_info = resize_with_pad(img_orig, target_size, pad_value=0)

    # 归一化
    temp_transform = get_validation_transforms(target_size)
    normalize = None
    for t in temp_transform.transforms:
        if hasattr(t, 'mean'):
            normalize = t
            break

    if normalize is None:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = normalize.mean
        std = normalize.std

    img_normalized = (img_padded / 255.0 - np.array(mean).reshape(1, 1, 3)) / np.array(std).reshape(1, 1, 3)
    input_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).float().to(device)

    # 模型推理
    logits = stage2_model(input_tensor)
    prob = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_padded = np.argmax(prob, axis=0)

    # 去除padding并resize回原始尺寸
    lesion_pred = remove_pad_and_resize(pred_padded, pad_info, (h, w))

    return img_orig, lesion_pred


# ==================== 指标计算 ====================
def compute_pa_and_iou(pred_mask, true_mask, num_classes=5):
    """计算像素准确率和各类别IoU"""
    pred_flat = pred_mask.flatten()
    true_flat = true_mask.flatten()

    # 像素准确率
    correct = np.sum(pred_flat == true_flat)
    pa = correct / len(pred_flat)

    # 各类别IoU
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred_flat == cls)
        true_cls = (true_flat == cls)
        intersection = np.sum(pred_cls & true_cls)
        union = np.sum(pred_cls | true_cls)
        iou = intersection / (union + 1e-6)
        ious.append(iou)

    return pa, ious


def compute_binary_iou(pred_binary, true_binary):
    """计算二值IoU（前景vs背景）"""
    pred_flat = pred_binary.flatten()
    true_flat = true_binary.flatten()
    intersection = np.sum(pred_flat & true_flat)
    union = np.sum(pred_flat | true_flat)
    return intersection / (union + 1e-6) if union > 0 else 0.0


# ==================== 6张图可视化 ====================
def visualize_stage2(img_orig, lesion_pred, lesion_gt_path):
    """
    6张图可视化布局（阶段2）:
    [原图, 预测掩码, 预测叠加]
    [真值掩码, 真值叠加, 差异图]
    """
    h, w = img_orig.shape[:2]

    # 读取病灶真值
    lesion_gt_pil = Image.open(lesion_gt_path).convert('L')
    if lesion_gt_pil.size != (w, h):
        lesion_gt_pil = lesion_gt_pil.resize((w, h), Image.NEAREST)
    lesion_gt = np.array(lesion_gt_pil)

    # ========== 计算指标 ==========
    pa, ious = compute_pa_and_iou(lesion_pred, lesion_gt, num_classes=5)
    miou = np.mean(ious)

    # 二值IoU（病灶vs背景）
    pred_binary = lesion_pred > 0
    gt_binary = lesion_gt > 0
    iou_binary = compute_binary_iou(pred_binary, gt_binary)

    # 统计TP/FP/FN
    tp = np.sum(pred_binary & gt_binary)
    fp = np.sum(pred_binary & ~gt_binary)
    fn = np.sum(~pred_binary & gt_binary)

    # 各类别像素统计
    print("\n" + "=" * 60)
    print("Evaluation Metrics")
    print(f"PA (Pixel Accuracy):  {pa * 100:.2f}%")
    print(f"mIoU (5 classes):     {miou * 100:.2f}%")
    print(f"IoU (Lesion vs BG):   {iou_binary * 100:.2f}%")
    print("-" * 40)
    print("Per-class IoU:")
    for cls in range(5):
        print(f"  {CLASS_NAMES[cls]:20s}: {ious[cls] * 100:6.2f}%")
    print("-" * 40)
    print(f"TP: {tp}, FP: {fp}, FN: {fn}")
    print("=" * 60)

    # ========== 创建可视化 ==========
    alpha = 0.5

    # --- 预测掩码彩色图 ---
    colored_pred = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in CLASS_COLORS.items():
        colored_pred[lesion_pred == cls] = color

    # --- 真值掩码彩色图 ---
    colored_gt = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in CLASS_COLORS.items():
        colored_gt[lesion_gt == cls] = color

    # --- 预测叠加图 ---
    overlay_pred = img_orig.copy().astype(np.float32)
    for cls, color in CLASS_COLORS.items():
        if cls == 0:
            continue
        mask_cls = (lesion_pred == cls)
        if np.any(mask_cls):
            overlay_pred[mask_cls] = overlay_pred[mask_cls] * (1 - alpha) + np.array(color) * alpha
    overlay_pred = overlay_pred.astype(np.uint8)

    # --- 真值叠加图 ---
    overlay_gt = img_orig.copy().astype(np.float32)
    for cls, color in CLASS_COLORS.items():
        if cls == 0:
            continue
        mask_cls = (lesion_gt == cls)
        if np.any(mask_cls):
            overlay_gt[mask_cls] = overlay_gt[mask_cls] * (1 - alpha) + np.array(color) * alpha
    overlay_gt = overlay_gt.astype(np.uint8)

    # --- 差异图（TP/FP/FN）---
    diff_img = np.zeros((h, w, 3), dtype=np.uint8)
    tp_mask = pred_binary & gt_binary  # 正确检测的病灶（绿色）
    fp_mask = pred_binary & ~gt_binary  # 假阳性（红色）
    fn_mask = ~pred_binary & gt_binary  # 假阴性（蓝色）
    diff_img[tp_mask] = [0, 255, 0]  # TP: 绿色
    diff_img[fp_mask] = [255, 0, 0]  # FP: 红色
    diff_img[fn_mask] = [0, 0, 255]  # FN: 蓝色

    # ========== 6张图布局 (2行 x 3列) ==========
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 第1行: 预测结果
    axes[0, 0].imshow(img_orig)
    axes[0, 0].set_title("Original Image (Cropped)")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(colored_pred)
    pred_pixels = np.sum(lesion_pred > 0)
    gt_pixels = np.sum(lesion_gt > 0)
    diff_pixels = pred_pixels - gt_pixels
    diff_percent = (diff_pixels / gt_pixels * 100) if gt_pixels > 0 else 0
    axes[0, 1].set_title(f'Predicted Lesion Mask\n'
                         f'{pred_pixels:,} pixels\n'
                         f'Diff: {diff_pixels:+,} ({diff_percent:+.1f}%)')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(overlay_pred)
    axes[0, 2].set_title(f'Prediction Overlay\nmIoU: {miou:.4f}')
    axes[0, 2].axis('off')

    # 第2行: 真值对比
    axes[1, 0].imshow(colored_gt)
    axes[1, 0].set_title(f'Ground Truth Mask\n{gt_pixels:,} pixels')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(overlay_gt)
    axes[1, 1].set_title('Ground Truth Overlay')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(diff_img)
    axes[1, 2].set_title(f'Difference Map\n'
                         f'TP:{tp} FP:{fp} FN:{fn}\n'
                         f'IoU:{iou_binary:.4f}')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.3)
    plt.show()

    return fig


# ==================== 主函数 ====================
def main():
    print(f"Using device: {DEVICE}")

    # 检查输入文件
    if not Path(CROPPED_IMG_PATH).exists():
        print(f"Error: Input image not found at {CROPPED_IMG_PATH}")
        return
    if not Path(LESION_GT_PATH).exists():
        print(f"Error: Ground truth not found at {LESION_GT_PATH}")
        return

    # 加载模型
    print("Loading stage2 model...")
    model = load_stage2_model(STAGE2_CKPT, DEVICE)

    # 阶段2推理
    print(f"\nProcessing: {CROPPED_IMG_PATH}")
    img_orig, lesion_pred = inference_stage2(CROPPED_IMG_PATH, model, DEVICE)

    # 可视化和评估
    if Path(LESION_GT_PATH).exists():
        visualize_stage2(img_orig, lesion_pred, LESION_GT_PATH)
    else:
        print("Ground truth not found, skipping evaluation.")

    # 打印病灶统计
    unique, counts = np.unique(lesion_pred, return_counts=True)
    print("\nLesion prediction statistics:")
    for cls, count in zip(unique, counts):
        percentage = 100 * count / lesion_pred.size
        print(f"  {CLASS_NAMES[cls]:20s}: {count:8,} pixels ({percentage:5.2f}%)")

    print(f"\nDone!")


if __name__ == "__main__":
    main()