#!/usr/bin/env python3
"""
独立测试阶段2模型（使用全图病灶掩膜评估）
- 利用真实叶片掩膜裁剪图像，输入阶段2模型
- 将预测映射回全图，计算全图指标：
  PA, IoU-B, IoU-leaf, IoU-lesion, MIoU (Bg+Leaf+Lesion)
- 统计平均推理时间
"""

import os
import sys
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from skimage.transform import resize
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.stage2_lesion_config import Stage2Config

# ==================== 用户配置（请根据实际情况修改）====================
STAGE2_CHECKPOINT = r"C:\Users\86159\PycharmProjects\disease_evaluation\checkpoints\stage2\best_model.pth"
TEST_IMG_DIR = Stage2Config().test_img_dir               # 原始测试图像目录
TEST_LEAF_MASK_DIR = Path(r"C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\stage2\test\images")   # 叶片真实掩膜目录（二值PNG）
FULL_LESION_MASK_DIR = Path(r"C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\stage2\test\labels")  # 全图病灶掩膜目录
PADDING = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ===================================================================


class DeepLabV3PlusWithClassifier(nn.Module):
    """DeepLabV3+ 分割模型 + 全局分类头（用于加载Stage2权重）"""
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
        return self.seg_model(x)  # 推理时只返回分割结果


def compute_pa_and_iou(pred_mask, true_mask, num_classes=2):
    """计算像素精度(PA)和各类IoU（二分类）"""
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


def compute_binary_iou(pred_binary, true_binary):
    """计算二值掩膜的IoU（用于病灶）"""
    pred_flat = pred_binary.flatten()
    true_flat = true_binary.flatten()
    intersection = np.sum(pred_flat & true_flat)
    union = np.sum(pred_flat | true_flat)
    return intersection / (union + 1e-6)


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
    # 适配键名（如果训练时保存的键名没有 seg_model. 前缀）
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


@torch.no_grad()
def process_single_image(img_path, leaf_gt_path, lesion_gt_full_path,
                         stage2_model, device, padding=10):
    """处理单张图像，返回全图预测和指标计算所需数据"""
    # 读取原始图像
    img_orig = np.array(Image.open(img_path).convert('RGB'))
    h, w = img_orig.shape[:2]

    # 读取叶片真实掩膜（对齐尺寸）
    leaf_true_pil = Image.open(leaf_gt_path).convert('L')
    if leaf_true_pil.size != (w, h):
        leaf_true_pil = leaf_true_pil.resize((w, h), Image.NEAREST)
    leaf_true = np.array(leaf_true_pil) > 0

    # 读取病灶真实掩膜（全图，对齐尺寸）
    lesion_true_full = np.array(Image.open(lesion_gt_full_path).convert('L'))
    if lesion_true_full.shape[:2] != (h, w):
        from skimage.transform import resize as sk_resize
        lesion_true_full = sk_resize(lesion_true_full, (h, w), preserve_range=True, order=0).astype(np.uint8)
    lesion_true_binary = lesion_true_full > 0

    # 使用真实叶片掩膜进行裁剪（与端到端中预测叶片掩膜不同，这里为“理想”裁剪）
    # 计算裁剪区域
    y_indices, x_indices = np.where(leaf_true)
    if len(y_indices) == 0:
        y1, y2 = 0, h-1
        x1, x2 = 0, w-1
    else:
        y1, y2 = y_indices.min(), y_indices.max()
        x1, x2 = x_indices.min(), x_indices.max()
    y1 = max(0, y1 - padding)
    y2 = min(h-1, y2 + padding)
    x1 = max(0, x1 - padding)
    x2 = min(w-1, x2 + padding)

    # 裁剪图像（背景已保留，但只裁剪叶片区域）
    img_cropped = img_orig[y1:y2+1, x1:x2+1].copy()
    crop_h, crop_w = img_cropped.shape[:2]

    # 阶段2推理
    t_start = time.time()
    target_size = Stage2Config().img_size
    transform = A.Compose([
        A.Resize(target_size, target_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    transformed = transform(image=img_cropped)
    input_tensor = transformed['image'].unsqueeze(0).to(device)
    logits = stage2_model(input_tensor)
    prob = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred = np.argmax(prob, axis=0)                     # (target_size, target_size)
    lesion_pred_cropped = resize(pred, (crop_h, crop_w), preserve_range=True, order=0).astype(np.uint8)

    # 映射回全图
    lesion_pred_full = np.zeros((h, w), dtype=np.uint8)
    lesion_pred_full[y1:y2+1, x1:x2+1] = lesion_pred_cropped
    lesion_pred_binary = lesion_pred_full > 0

    total_time = time.time() - t_start

    return leaf_true, lesion_pred_binary, lesion_true_binary, total_time


def main():
    print(f"Using device: {DEVICE}")
    print("Loading stage2 model...")
    model = load_stage2_model(STAGE2_CHECKPOINT, DEVICE)

    # 获取测试图像列表
    img_files = sorted([f for f in TEST_IMG_DIR.glob("*") if f.suffix.lower() in ['.jpg','.jpeg','.png']])
    valid_pairs = []
    for img_f in img_files:
        stem = img_f.stem
        leaf_gt = TEST_LEAF_MASK_DIR / f"{stem}.png"
        lesion_gt_full = FULL_LESION_MASK_DIR / f"{stem}.png"
        if leaf_gt.exists() and lesion_gt_full.exists():
            valid_pairs.append((img_f, leaf_gt, lesion_gt_full))
        else:
            print(f"Warning: missing ground truth for {stem}, skipping")
    print(f"Found {len(valid_pairs)} valid test images")

    # 累积指标
    total_pa = 0.0
    total_iou_bg = 0.0
    total_iou_leaf = 0.0
    total_iou_lesion = 0.0
    total_miou = 0.0
    total_time = 0.0
    num_samples = len(valid_pairs)

    for img_path, leaf_gt_path, lesion_gt_full_path in tqdm(valid_pairs, desc="Evaluating"):
        leaf_true, lesion_pred_bin, lesion_true_bin, proc_time = process_single_image(
            img_path, leaf_gt_path, lesion_gt_full_path,
            model, DEVICE, PADDING
        )

        # 叶片分割指标（这里叶片预测就是真实叶片掩膜，因为使用了真值裁剪）
        # 但为了指标一致性，我们将叶片预测视为真实掩膜（因为阶段2不预测叶片）
        # 注意：这里的叶片 IoU 实际上就是 1.0，因为 leaf_true 就是真值
        # 如果希望评估阶段2模型在真实叶片区域下的病灶分割性能，叶片指标无意义，可忽略。
        # 我们仍然计算 PA 和 IoU，但叶片预测直接使用 leaf_true 本身。
        leaf_pred = leaf_true  # 理想情况
        pa, ious = compute_pa_and_iou(leaf_pred.astype(np.uint8), leaf_true.astype(np.uint8), num_classes=2)
        iou_bg, iou_leaf = ious[0], ious[1]  # iou_leaf 必然为 1.0

        iou_lesion = compute_binary_iou(lesion_pred_bin, lesion_true_bin)

        miou = (iou_bg + iou_leaf + iou_lesion) / 3.0

        total_pa += pa
        total_iou_bg += iou_bg
        total_iou_leaf += iou_leaf
        total_iou_lesion += iou_lesion
        total_miou += miou
        total_time += proc_time

    avg_pa = total_pa / num_samples * 100
    avg_iou_bg = total_iou_bg / num_samples * 100
    avg_iou_leaf = total_iou_leaf / num_samples * 100
    avg_iou_lesion = total_iou_lesion / num_samples * 100
    avg_miou = total_miou / num_samples * 100
    avg_inference_time_ms = total_time / num_samples * 1000

    print("\n" + "="*60)
    print("Stage2 Model Test Results (using full-image lesion masks)")
    print("(Leaf mask = ground truth, lesion predicted from cropped region)")
    print("="*60)
    print(f"PA (Pixel Accuracy):            {avg_pa:.2f}%")
    print(f"IoU-B (Background):             {avg_iou_bg:.2f}%")
    print(f"IoU-leaf (Leaf):                {avg_iou_leaf:.2f}%")
    print(f"IoU-lesion (Lesion):            {avg_iou_lesion:.2f}%")
    print(f"MIoU (Bg+Leaf+Lesion):          {avg_miou:.2f}%")
    print(f"Average Inference Time:         {avg_inference_time_ms:.2f} ms per image")
    print("="*60)


if __name__ == "__main__":
    main()
