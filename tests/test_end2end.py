#!/usr/bin/env python3
"""
联合测试脚本：阶段1（叶片分割）+ 阶段2（标准DeepLabV3+，无分类头）
使用全图病灶掩膜（与原图同尺寸）进行评估
在测试集上评估以下指标：
- PA (Pixel Accuracy)
- IoU-B (Background)
- IoU-leaf (Leaf)
- IoU-lesion (Lesion, binary)
- MIoU (mean of Background, Leaf, Lesion)
- Average inference time (ms per image)
"""

import os
import sys
from pathlib import Path
import time
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from skimage.transform import resize
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.stage1_leaf_config import Stage1Config
from configs.stage2_lesion_config import Stage2Config

# ==================== 用户配置（请根据实际情况修改）====================
STAGE1_CHECKPOINT = r"C:\Users\86159\PycharmProjects\disease_evaluation\checkpoints\stage1\best_model.pth"
STAGE2_CHECKPOINT = r"C:\Users\86159\PycharmProjects\disease_evaluation\checkpoints\stage2\可能输出多类病斑\best_model.pth"
TEST_IMG_DIR = Stage1Config().test_img_dir          # 原始测试图像目录
TEST_LEAF_MASK_DIR = Stage1Config().test_label_dir  # 叶片真实掩膜目录（二值PNG）
FULL_LESION_MASK_DIR = Path(r"C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\stage2_fullmask\test\labels")  # 全图病灶掩膜目录
PADDING = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ===================================================================


# ==================== 模型加载 ====================
def load_stage1_model(checkpoint_path, device):
    """加载阶段1模型（叶片分割，二分类）"""
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
    """加载阶段2模型（标准DeepLabV3+，多类别输出）"""
    config = Stage2Config()
    model = smp.DeepLabV3Plus(
        encoder_name=config.encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=config.num_classes,
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


# ==================== 指标计算函数 ====================
def compute_pa_and_iou_binary(pred_binary, true_binary):
    """计算二值掩膜的PA和各类IoU（背景和前景）"""
    pred_flat = pred_binary.flatten()
    true_flat = true_binary.flatten()
    # 像素精度
    correct = np.sum(pred_flat == true_flat)
    pa = correct / len(pred_flat)
    # 背景IoU (class 0)
    pred_bg = (pred_flat == 0)
    true_bg = (true_flat == 0)
    inter_bg = np.sum(pred_bg & true_bg)
    union_bg = np.sum(pred_bg | true_bg)
    iou_bg = inter_bg / (union_bg + 1e-6)
    # 前景IoU (class 1)
    pred_fg = (pred_flat == 1)
    true_fg = (true_flat == 1)
    inter_fg = np.sum(pred_fg & true_fg)
    union_fg = np.sum(pred_fg | true_fg)
    iou_fg = inter_fg / (union_fg + 1e-6)
    return pa, iou_bg, iou_fg

def compute_binary_iou(pred_binary, true_binary):
    """计算二值掩膜的IoU（前景）"""
    pred_flat = pred_binary.flatten()
    true_flat = true_binary.flatten()
    inter = np.sum(pred_flat & true_flat)
    union = np.sum(pred_flat | true_flat)
    return inter / (union + 1e-6)


# ==================== 联合处理单张图像 ====================
@torch.no_grad()
def process_single_image(img_path, leaf_mask_path, lesion_full_mask_path,
                         stage1_model, stage2_model, device, padding=10):
    """
    处理单张图像，返回：
    - 叶片预测二值掩膜
    - 叶片真实二值掩膜
    - 病灶预测二值掩膜（全图）
    - 病灶真实二值掩膜（全图）
    - 总推理时间
    """
    # 读取原始图像
    img_orig = np.array(Image.open(img_path).convert('RGB'))
    h, w = img_orig.shape[:2]

    # 读取叶片真实掩膜（二值），并确保尺寸一致
    leaf_true_pil = Image.open(leaf_mask_path).convert('L')
    if leaf_true_pil.size != (w, h):
        leaf_true_pil = leaf_true_pil.resize((w, h), Image.NEAREST)
    leaf_true = np.array(leaf_true_pil) > 0

    # 读取全图病灶真实掩膜，并确保尺寸一致
    lesion_true_full = np.array(Image.open(lesion_full_mask_path).convert('L'))
    if lesion_true_full.shape[:2] != (h, w):
        lesion_true_full = resize(lesion_true_full, (h, w), preserve_range=True, order=0).astype(np.uint8)
    lesion_true_binary = lesion_true_full > 0

    # ---- 阶段1：叶片分割 ----
    t_start = time.time()
    # 定义与训练时一致的变换（阶段1）
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

    # 根据预测叶片掩膜裁剪区域
    y_indices, x_indices = np.where(leaf_pred_binary)
    if len(y_indices) == 0:
        # 未检测到叶片，使用整图
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
    target_size = Stage2Config().img_size
    transform2 = A.Compose([
        A.Resize(target_size, target_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    transformed2 = transform2(image=img_cropped)
    input_tensor2 = transformed2['image'].unsqueeze(0).to(device)
    logits2 = stage2_model(input_tensor2)
    prob2 = torch.softmax(logits2, dim=1)[0].cpu().numpy()  # (C, H2, W2)
    pred2 = np.argmax(prob2, axis=0)  # (H2, W2)
    # 将预测结果resize回裁剪区域尺寸
    lesion_pred_cropped = resize(pred2, (crop_h, crop_w), preserve_range=True, order=0).astype(np.uint8)
    # 二值化（任何非0类别视为病灶）
    lesion_pred_binary_cropped = lesion_pred_cropped > 0

    # 将裁剪区域内的预测病灶掩膜映射回原图坐标
    lesion_pred_binary_full = np.zeros((h, w), dtype=bool)
    lesion_pred_binary_full[y1:y2+1, x1:x2+1] = lesion_pred_binary_cropped
    total_time = time.time() - t_start

    return leaf_pred_binary, leaf_true, lesion_pred_binary_full, lesion_true_binary, total_time


# ==================== 主测试函数 ====================
def main():
    print(f"Using device: {DEVICE}")
    print("Loading models...")
    model1 = load_stage1_model(STAGE1_CHECKPOINT, DEVICE)
    model2 = load_stage2_model(STAGE2_CHECKPOINT, DEVICE)

    # 获取测试图像列表（取原始图像目录和叶片掩膜目录的交集，且全图病灶掩膜必须存在）
    img_files = sorted([f for f in TEST_IMG_DIR.glob("*") if f.suffix.lower() in ['.jpg','.jpeg','.png']])
    valid_pairs = []
    for img_f in img_files:
        stem = img_f.stem
        leaf_gt = TEST_LEAF_MASK_DIR / f"{stem}.png"
        lesion_full_gt = FULL_LESION_MASK_DIR / f"{stem}.png"
        if leaf_gt.exists() and lesion_full_gt.exists():
            valid_pairs.append((img_f, leaf_gt, lesion_full_gt))
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

    # 逐张处理
    for img_path, leaf_path, lesion_full_path in tqdm(valid_pairs, desc="Evaluating"):
        (leaf_pred, leaf_true,
         lesion_pred_bin, lesion_true_bin,
         proc_time) = process_single_image(
            img_path, leaf_path, lesion_full_path,
            model1, model2, DEVICE, PADDING
        )

        # 计算叶片分割指标（二分类）
        pa, iou_bg, iou_leaf = compute_pa_and_iou_binary(leaf_pred, leaf_true)
        # 计算病灶分割指标（二值）
        iou_lesion = compute_binary_iou(lesion_pred_bin, lesion_true_bin)
        # 计算三类别平均IoU（背景、叶片、病灶）
        miou = (iou_bg + iou_leaf + iou_lesion) / 3.0

        total_pa += pa
        total_iou_bg += iou_bg
        total_iou_leaf += iou_leaf
        total_iou_lesion += iou_lesion
        total_miou += miou
        total_time += proc_time

    # 计算平均值
    avg_pa = total_pa / num_samples * 100
    avg_iou_bg = total_iou_bg / num_samples * 100
    avg_iou_leaf = total_iou_leaf / num_samples * 100
    avg_iou_lesion = total_iou_lesion / num_samples * 100
    avg_miou = total_miou / num_samples * 100
    avg_inference_time_ms = total_time / num_samples * 1000

    # 打印结果
    print("\n" + "="*60)
    print("Test Set Results (Stage1 + Standard Stage2, using full-image lesion masks)")
    print("="*60)
    print(f"PA (Pixel Accuracy):            {avg_pa:.2f}%")
    print(f"IoU-B (Background):             {avg_iou_bg:.2f}%")
    print(f"IoU-leaf (Leaf):                {avg_iou_leaf:.2f}%")
    print(f"IoU-lesion (Lesion, binary):    {avg_iou_lesion:.2f}%")
    print(f"MIoU (Bg+Leaf+Lesion):          {avg_miou:.2f}%")
    print(f"Average Inference Time:         {avg_inference_time_ms:.2f} ms per image")
    print("="*60)


if __name__ == "__main__":
    main()