#!/usr/bin/env python3
"""
端到端批量推理脚本：叶片分割 + 病灶分割
对目录下所有图片进行处理，生成官网风格可视化（紫色/绿色发光边界）
并输出平均指标（IoU, PA, mIoU 等）
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from skimage.transform import resize
import segmentation_models_pytorch as smp
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from src.data.transforms import get_validation_transforms
from configs.stage1_leaf_config import Stage1Config
from configs.stage2_lesion_config import Stage2Config

# ==================== 用户配置 ====================
INPUT_DIR = r"C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\用于论文\images"  # 输入图片目录
OUTPUT_DIR = r"C:\Users\86159\Desktop\lesion_visualizations"  # 输出可视化目录
STAGE1_CKPT = r"C:\Users\86159\PycharmProjects\disease_evaluation\checkpoints\stage1\best_model.pth"
STAGE2_CKPT = r"C:\Users\86159\PycharmProjects\disease_evaluation\checkpoints\stage2\best_model.pth"
PADDING = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 全图病灶掩膜目录（存放与原图同尺寸的PNG）
LESION_FULL_MASK_DIR = Path(r"C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\用于论文\lesion")
# 叶片真值目录（用于评估）
LEAF_GT_DIR = Stage1Config().test_label_dir  # 根据你的配置自动获取

CLASS_NAMES = {
    0: 'Background',
    1: 'Powdery Mildew',
    2: 'Rust',
    3: 'Spot',
    4: 'Bacterial Blight'
}

# ==================== 可视化配置（官网风格）====================
VIS_CONFIG_MODEL = {
    'mask_alpha': 0.55,
    'mask_color': [128, 0, 128],          # 紫色
    'boundary_color': [255, 105, 180],
    'boundary_thickness': 3,
    'glow_thickness': 6,
    'glow_color': [255, 20, 147],
}

VIS_CONFIG_GT = {
    'mask_alpha': 0.55,
    'mask_color': [128, 0, 128],          # 紫色
    'boundary_color': [255, 105, 180],
    'boundary_thickness': 3,
    'glow_thickness': 6,
    'glow_color': [255, 20, 147],
}


# ==================== 官网风格可视化函数 ====================
def visualize_mask_on_image(image_np, mask, save_path, color_config):
    """生成官网风格掩膜可视化并保存"""
    mask = mask.astype(bool)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    mask_overlay = np.zeros_like(image_bgr)
    mask_overlay[mask] = color_config['mask_color']
    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    glow_layer = np.zeros_like(image_bgr)
    if len(contours) > 0:
        cv2.drawContours(glow_layer, contours, -1,
                         color_config['glow_color'],
                         color_config['glow_thickness'])
        cv2.drawContours(glow_layer, contours, -1,
                         color_config['boundary_color'],
                         color_config['boundary_thickness'])
        cv2.drawContours(glow_layer, contours, -1, [255, 255, 255], 1)
    result = cv2.addWeighted(image_bgr, 1.0, mask_overlay, color_config['mask_alpha'], 0)
    result = cv2.addWeighted(result, 1.0, glow_layer, 0.85, 0)
    original_pixels = image_bgr[mask].astype(np.float32)
    tint_color = np.array(color_config['mask_color'], dtype=np.float32)
    tinted = original_pixels * 0.7 + tint_color * 0.3
    result[mask] = tinted.astype(np.uint8)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), result)


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
        return self.seg_model(x)


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


@torch.no_grad()
def inference_single_image(img_path, stage1_model, stage2_model, transform1, device, padding=10):
    """对单张图片进行推理，返回所需结果"""
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

    return img_orig, leaf_mask, lesion_mask_full, lesion_pred_cropped, leaf_mask_cropped, (x1, y1, x2, y2), (crop_h, crop_w)


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
    if np.sum(mask) == 0:
        return 0.0
    pred_flat = pred_binary[mask].flatten()
    true_flat = true_binary[mask].flatten()
    intersection = np.sum(pred_flat & true_flat)
    union = np.sum(pred_flat | true_flat)
    return intersection / (union + 1e-6) if union > 0 else 0.0


# ==================== 批量处理 ====================
def main():
    print(f"Using device: {DEVICE}")
    config1 = Stage1Config()
    transform1 = get_transform(config1.img_size)

    print("Loading models...")
    model1 = load_stage1_model(STAGE1_CKPT, DEVICE)
    model2 = load_stage2_model(STAGE2_CKPT, DEVICE)

    # 获取所有图片文件
    input_path = Path(INPUT_DIR)
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
    print(f"Found {len(image_files)} images.")

    # 累积指标
    total_pa = 0.0
    total_iou_bg = 0.0
    total_iou_leaf = 0.0
    total_iou_lesion = 0.0
    total_miou = 0.0
    num_valid = 0

    # 输出目录
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(image_files, desc="Processing"):
        stem = img_path.stem
        leaf_gt_path = LEAF_GT_DIR / f"{stem}.png"
        lesion_gt_path = LESION_FULL_MASK_DIR / f"{stem}.png"

        if not leaf_gt_path.exists() or not lesion_gt_path.exists():
            print(f"Skipping {stem}: GT not found")
            continue

        # 推理
        (img_orig, leaf_pred, lesion_pred_full, lesion_pred_cropped,
         leaf_mask_cropped, bbox, crop_shape) = inference_single_image(
            str(img_path), model1, model2, transform1, DEVICE, PADDING)

        # 读取真值
        leaf_true = np.array(Image.open(leaf_gt_path).convert('L')) > 127
        lesion_true_full = np.array(Image.open(lesion_gt_path).convert('L'))
        if lesion_true_full.shape[:2] != img_orig.shape[:2]:
            lesion_true_full = resize(lesion_true_full, img_orig.shape[:2], preserve_range=True, order=0).astype(np.uint8)
        lesion_true_binary = lesion_true_full > 0

        # 叶片指标
        leaf_pred_uint8 = leaf_pred.astype(np.uint8)
        leaf_true_uint8 = leaf_true.astype(np.uint8)
        pa, ious = compute_pa_and_iou(leaf_pred_uint8, leaf_true_uint8, num_classes=2)
        iou_bg, iou_leaf = ious[0], ious[1]

        # 病灶 IoU（仅在叶片区域内）
        leaf_region_mask = leaf_pred & leaf_true
        pred_bin = lesion_pred_full > 0
        iou_lesion = compute_binary_iou_masked(pred_bin, lesion_true_binary, leaf_region_mask)
        miou = (iou_bg + iou_leaf + iou_lesion) / 3.0

        # 累加
        total_pa += pa
        total_iou_bg += iou_bg
        total_iou_leaf += iou_leaf
        total_iou_lesion += iou_lesion
        total_miou += miou
        num_valid += 1

        # 保存可视化
        # 叶片预测（紫色）
        vis_leaf = out_dir / f"{stem}_leaf_pred.png"
        visualize_mask_on_image(img_orig, leaf_pred, vis_leaf, VIS_CONFIG_MODEL)
        # 病灶预测（紫色）
        vis_lesion_pred = out_dir / f"{stem}_lesion_pred.png"
        visualize_mask_on_image(img_orig, pred_bin, vis_lesion_pred, VIS_CONFIG_MODEL)
        # 病灶真值（绿色）
        vis_lesion_gt = out_dir / f"{stem}_lesion_gt.png"
        visualize_mask_on_image(img_orig, lesion_true_binary, vis_lesion_gt, VIS_CONFIG_GT)

    if num_valid == 0:
        print("No valid images found.")
        return

    # 输出平均指标
    print("\n" + "=" * 60)
    print(f"Average metrics over {num_valid} images:")
    print(f"PA (Pixel Accuracy):           {total_pa / num_valid * 100:.2f}%")
    print(f"IoU-Background:                {total_iou_bg / num_valid * 100:.2f}%")
    print(f"IoU-Leaf:                      {total_iou_leaf / num_valid * 100:.2f}%")
    print(f"IoU-Lesion (in leaf region):   {total_iou_lesion / num_valid * 100:.2f}%")
    print(f"mIoU (Bg+Leaf+Lesion):         {total_miou / num_valid * 100:.2f}%")
    print("=" * 60)
    print(f"All visualizations saved to: {out_dir}")


if __name__ == "__main__":
    main()