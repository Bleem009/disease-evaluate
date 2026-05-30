#!/usr/bin/env python3
"""
测试脚本：联合阶段1（叶片分割）和阶段2（病灶分割）模型在测试集上的性能
- 阶段1单独评估：PA, IoU-B, IoU-leaf, mIoU, 推理时间
- 端到端三分类评估（背景、健康叶片、病灶）：PA, IoU-B, IoU-H, IoU-Lesion, mIoU
- 分别统计阶段1和阶段2的平均推理时间（毫秒/张）
"""

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

from configs.stage1_leaf_config import Stage1Config
from configs.stage2_lesion_config import Stage2Config

# ==================== 用户配置 ====================
STAGE1_CHECKPOINT = r"C:\Users\86159\PycharmProjects\disease_evaluation\checkpoints\stage1\best_model.pth"
STAGE2_CHECKPOINT = r"C:\Users\86159\PycharmProjects\disease_evaluation\checkpoints\stage2\best_model.pth"
TEST_IMG_DIR = Stage1Config().test_img_dir
TEST_LEAF_MASK_DIR = Stage1Config().test_label_dir
FULL_LESION_MASK_DIR = Path(r"C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\stage2_fullmask\test\labels")
PADDING = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WARMUP_ITERATIONS = 50
# ===================================================================


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


def compute_binary_metrics(pred_binary, true_binary):
    """
    计算二分类（背景/前景）的PA和每个类别的IoU
    pred_binary, true_binary: bool or 0/1 array, shape (H,W)
    返回: pa, iou_bg, iou_fg
    """
    pred_flat = pred_binary.flatten().astype(np.uint8)
    true_flat = true_binary.flatten().astype(np.uint8)
    correct = np.sum(pred_flat == true_flat)
    pa = correct / len(pred_flat)
    # 背景 (0)
    pred_bg = (pred_flat == 0)
    true_bg = (true_flat == 0)
    inter_bg = np.sum(pred_bg & true_bg)
    union_bg = np.sum(pred_bg | true_bg)
    iou_bg = inter_bg / (union_bg + 1e-6)
    # 前景 (1)
    pred_fg = (pred_flat == 1)
    true_fg = (true_flat == 1)
    inter_fg = np.sum(pred_fg & true_fg)
    union_fg = np.sum(pred_fg | true_fg)
    iou_fg = inter_fg / (union_fg + 1e-6)
    return pa, iou_bg, iou_fg


def compute_multiclass_metrics(pred, true, num_classes=3):
    """
    计算多分类的像素精度(PA)和每个类别的IoU
    pred, true: shape (H, W), 整数值0..num_classes-1
    返回: (pa, iou_list)
    """
    pred_flat = pred.flatten()
    true_flat = true.flatten()
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
        elif k.startswith('classifier.'):
            new_state_dict[k] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def warmup_models(model1, model2, device, img_size1, img_size2, warmup_iters=50):
    print(f"Warming up models with {warmup_iters} iterations...")
    dummy1 = torch.randn(1, 3, img_size1, img_size1).to(device)
    dummy2 = torch.randn(1, 3, img_size2, img_size2).to(device)
    for _ in range(warmup_iters):
        with torch.no_grad():
            _ = model1(dummy1)
            _ = model2(dummy2)
    torch.cuda.synchronize()
    print("Warmup finished.")


@torch.no_grad()
def process_single_image(img_path, leaf_gt_path, lesion_gt_full_path,
                         stage1_model, stage2_model, device, padding=10):
    """
    返回：
    - 叶片预测二值掩膜 (bool)
    - 叶片真实二值掩膜 (bool)
    - 病灶预测二值掩膜（全图） (bool)
    - 病灶真实二值掩膜（全图） (bool)
    - stage1 推理时间（秒）
    - stage2 推理时间（秒）
    """
    img_orig = np.array(Image.open(img_path).convert('RGB'))
    h, w = img_orig.shape[:2]

    leaf_true_pil = Image.open(leaf_gt_path).convert('L')
    if leaf_true_pil.size != (w, h):
        leaf_true_pil = leaf_true_pil.resize((w, h), Image.NEAREST)
    leaf_true = np.array(leaf_true_pil) > 0

    lesion_true_full = np.array(Image.open(lesion_gt_full_path).convert('L'))
    if lesion_true_full.shape[:2] != (h, w):
        from skimage.transform import resize as sk_resize
        lesion_true_full = sk_resize(lesion_true_full, (h, w), preserve_range=True, order=0).astype(np.uint8)
    lesion_true_binary = lesion_true_full > 0

    # ========== 阶段1：叶片分割 ==========
    torch.cuda.synchronize()
    t1_start = time.time()

    transform1 = A.Compose([
        A.Resize(Stage1Config().img_size, Stage1Config().img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    transformed1 = transform1(image=img_orig)
    input_tensor1 = transformed1['image'].unsqueeze(0).to(device)
    logits1 = stage1_model(input_tensor1)
    prob1 = torch.sigmoid(logits1)[0, 0].cpu().numpy()
    leaf_prob = resize(prob1, (h, w), preserve_range=True, order=1)
    leaf_pred_binary = leaf_prob > 0.5

    torch.cuda.synchronize()
    t1_end = time.time()
    time_stage1 = t1_end - t1_start

    # ---------- 准备裁剪区域 ----------
    masked = img_orig.copy()
    masked[~leaf_pred_binary] = [0, 0, 0]
    y_indices, x_indices = np.where(leaf_pred_binary)
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
    img_cropped = masked[y1:y2+1, x1:x2+1].copy()
    crop_h, crop_w = img_cropped.shape[:2]

    # ========== 阶段2：病灶分割 ==========
    torch.cuda.synchronize()
    t2_start = time.time()

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
    pred2 = np.argmax(prob2, axis=0)
    lesion_pred_cropped = resize(pred2, (crop_h, crop_w), preserve_range=True, order=0).astype(np.uint8)

    torch.cuda.synchronize()
    t2_end = time.time()
    time_stage2 = t2_end - t2_start

    # 映射回全图
    lesion_pred_full = np.zeros((h, w), dtype=np.uint8)
    lesion_pred_full[y1:y2+1, x1:x2+1] = lesion_pred_cropped
    lesion_pred_binary = lesion_pred_full > 0

    return (leaf_pred_binary, leaf_true,
            lesion_pred_binary, lesion_true_binary,
            time_stage1, time_stage2)


def main():
    print(f"Using device: {DEVICE}")
    print("Loading models...")
    model1 = load_stage1_model(STAGE1_CHECKPOINT, DEVICE)
    model2 = load_stage2_model(STAGE2_CHECKPOINT, DEVICE)

    config1 = Stage1Config()
    config2 = Stage2Config()
    warmup_models(model1, model2, DEVICE, config1.img_size, config2.img_size, WARMUP_ITERATIONS)

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

    # 累加变量 - 阶段1（叶片分割）
    total_pa_stage1 = 0.0
    total_iou_bg_stage1 = 0.0
    total_iou_leaf = 0.0
    total_miou_stage1 = 0.0
    total_time_stage1 = 0.0

    # 累加变量 - 端到端三分类
    total_pa_e2e = 0.0
    total_iou_bg_e2e = 0.0
    total_iou_health = 0.0
    total_iou_lesion = 0.0
    total_time_stage2 = 0.0

    num_samples = len(valid_pairs)

    for img_path, leaf_gt_path, lesion_gt_full_path in tqdm(valid_pairs, desc="Evaluating"):
        (leaf_pred, leaf_true,
         lesion_pred_bin, lesion_true_bin,
         t1, t2) = process_single_image(
            img_path, leaf_gt_path, lesion_gt_full_path,
            model1, model2, DEVICE, PADDING
        )

        # ========== 第一阶段单独评估（二分类：背景/叶片） ==========
        # 将布尔掩码转为 0/1 整数数组（0=背景，1=叶片）
        pred_binary_stage1 = leaf_pred.astype(np.uint8)
        true_binary_stage1 = leaf_true.astype(np.uint8)
        pa1, iou_bg1, iou_leaf1 = compute_binary_metrics(pred_binary_stage1, true_binary_stage1)
        miou1 = (iou_bg1 + iou_leaf1) / 2.0

        total_pa_stage1 += pa1
        total_iou_bg_stage1 += iou_bg1
        total_iou_leaf += iou_leaf1
        total_miou_stage1 += miou1
        total_time_stage1 += t1

        # ========== 端到端三分类评估（背景、健康叶片、病灶） ==========
        # 真实标签
        true_background = ~leaf_true
        true_healthy = leaf_true & (~lesion_true_bin)
        true_lesion = lesion_true_bin
        true_multiclass = np.zeros_like(leaf_true, dtype=np.uint8)
        true_multiclass[true_background] = 0
        true_multiclass[true_healthy] = 1
        true_multiclass[true_lesion] = 2

        # 预测标签
        pred_background = ~leaf_pred
        pred_healthy = leaf_pred & (~lesion_pred_bin)
        pred_lesion = lesion_pred_bin
        pred_multiclass = np.zeros_like(leaf_pred, dtype=np.uint8)
        pred_multiclass[pred_background] = 0
        pred_multiclass[pred_healthy] = 1
        pred_multiclass[pred_lesion] = 2

        pa2, ious2 = compute_multiclass_metrics(pred_multiclass, true_multiclass, num_classes=3)
        iou_bg2, iou_health, iou_lesion2 = ious2

        total_pa_e2e += pa2
        total_iou_bg_e2e += iou_bg2
        total_iou_health += iou_health
        total_iou_lesion += iou_lesion2
        total_time_stage2 += t2

    # 计算平均值并转换为百分比
    avg_pa_stage1 = total_pa_stage1 / num_samples * 100
    avg_iou_bg_stage1 = total_iou_bg_stage1 / num_samples * 100
    avg_iou_leaf = total_iou_leaf / num_samples * 100
    avg_miou_stage1 = total_miou_stage1 / num_samples * 100
    avg_time_stage1_ms = total_time_stage1 / num_samples * 1000

    avg_pa_e2e = total_pa_e2e / num_samples * 100
    avg_iou_bg_e2e = total_iou_bg_e2e / num_samples * 100
    avg_iou_health = total_iou_health / num_samples * 100
    avg_iou_lesion = total_iou_lesion / num_samples * 100
    avg_miou_e2e = (avg_iou_bg_e2e + avg_iou_health + avg_iou_lesion) / 3.0
    avg_time_stage2_ms = total_time_stage2 / num_samples * 1000
    total_avg_time_ms = avg_time_stage1_ms + avg_time_stage2_ms

    # 输出结果
    print("\n" + "="*70)
    print("Stage 1 (Leaf Segmentation) - Binary Classification Results")
    print("="*70)
    print(f"PA (Pixel Accuracy):           {avg_pa_stage1:.2f}%")
    print(f"IoU-B (Background):            {avg_iou_bg_stage1:.2f}%")
    print(f"IoU-leaf (Leaf):               {avg_iou_leaf:.2f}%")
    print(f"mIoU (Bg+Leaf):                {avg_miou_stage1:.2f}%")
    print(f"Inference Time:                {avg_time_stage1_ms:.2f} ms per image")

    print("\n" + "="*70)
    print("End-to-End 3-Class Segmentation (Bg + Healthy Leaf + Lesion) Results")
    print("="*70)
    print(f"PA (Pixel Accuracy):           {avg_pa_e2e:.2f}%")
    print(f"IoU-B (Background):            {avg_iou_bg_e2e:.2f}%")
    print(f"IoU-H (Healthy Leaf):          {avg_iou_health:.2f}%")
    print(f"IoU-Lesion:                    {avg_iou_lesion:.2f}%")
    print(f"mIoU (Bg+Health+Lesion):       {avg_miou_e2e:.2f}%")

    print("\n" + "="*70)
    print("Inference Time Summary")
    print("="*70)
    print(f"Stage1 Inference Time:         {avg_time_stage1_ms:.2f} ms per image")
    print(f"Stage2 Inference Time:         {avg_time_stage2_ms:.2f} ms per image")
    print(f"Total Inference Time:          {total_avg_time_ms:.2f} ms per image")
    print("="*70)


if __name__ == "__main__":
    main()