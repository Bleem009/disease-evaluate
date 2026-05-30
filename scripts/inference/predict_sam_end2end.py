#!/usr/bin/env python3
"""
端到端测试脚本：SAM叶片分割（最大面积掩码） + 第二阶段病灶分割（带分类头）
计算指标：PA, IoU-B, IoU-leaf, IoU-lesion, MIoU, 平均推理时间
"""

import sys
from pathlib import Path
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from skimage.transform import resize
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.stage1_leaf_config import Stage1Config
from configs.stage2_lesion_config import Stage2Config


# ==================== 用户配置（可修改）====================
SAM_CHECKPOINT = r"D:\edge_download\sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"
STAGE2_CKPT = r"C:\Users\86159\PycharmProjects\disease_evaluation\checkpoints\stage2\best_model.pth"
TEST_IMG_DIR = Stage1Config().test_img_dir          # 原始测试图像目录
TEST_LEAF_MASK_DIR = Stage1Config().test_label_dir  # 叶片真实掩膜目录（二值PNG）
FULL_LESION_MASK_DIR = Path(r"C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\stage2_fullmask\test\labels")  # 全图病灶掩膜目录
PADDING = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =======================================================


# ------------------------- 第二阶段模型定义（带分类头）-------------------------
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
        return self.seg_model(x)  # 推理时只返回分割结果


# ------------------------- 模型加载 -------------------------
def load_sam_model(checkpoint_path, model_type="vit_b", device="cuda"):
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    # 低显存参数配置
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=16,          # 默认32，降低一半可减少显存
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        crop_n_layers=1,             # 禁用多尺度裁剪（最关键，大幅减少显存）
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )
    return mask_generator

def load_stage2_model(checkpoint_path, device):
    """加载第二阶段模型（带分类头）"""
    config = Stage2Config()
    model = DeepLabV3PlusWithClassifier(
        encoder_name=config.encoder_name,
        encoder_weights=None,
        num_classes=config.num_classes,
        in_channels=3
    )
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    # 适配键名
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


# ------------------------- SAM 叶片预测（最大面积掩码）-------------------------
def predict_leaf_mask_sam(image_np, mask_generator):
    """使用SAM预测叶片掩膜，选择面积最大的掩膜"""
    masks = mask_generator.generate(image_np)
    if not masks:
        return np.zeros(image_np.shape[:2], dtype=bool)
    largest = max(masks, key=lambda x: x['area'])
    return largest['segmentation']


# ------------------------- 指标计算 -------------------------
def compute_pa_and_iou_binary(pred_binary, true_binary):
    """计算二值掩膜的PA和各类IoU（背景和前景）"""
    pred_flat = pred_binary.flatten()
    true_flat = true_binary.flatten()
    correct = np.sum(pred_flat == true_flat)
    pa = correct / len(pred_flat)
    # 背景IoU
    pred_bg = (pred_flat == 0)
    true_bg = (true_flat == 0)
    inter_bg = np.sum(pred_bg & true_bg)
    union_bg = np.sum(pred_bg | true_bg)
    iou_bg = inter_bg / (union_bg + 1e-6)
    # 前景IoU
    pred_fg = (pred_flat == 1)
    true_fg = (true_flat == 1)
    inter_fg = np.sum(pred_fg & true_fg)
    union_fg = np.sum(pred_fg | true_fg)
    iou_fg = inter_fg / (union_fg + 1e-6)
    return pa, iou_bg, iou_fg

def compute_binary_iou(pred_binary, true_binary):
    pred_flat = pred_binary.flatten()
    true_flat = true_binary.flatten()
    inter = np.sum(pred_flat & true_flat)
    union = np.sum(pred_flat | true_flat)
    return inter / (union + 1e-6)


# ------------------------- 端到端处理单张图像 -------------------------
@torch.no_grad()
def process_single_image(img_path, leaf_gt_path, lesion_full_path,
                         mask_generator, stage2_model, device, padding=10):
    """
    使用SAM叶片分割 + 第二阶段病灶分割
    返回叶片预测、叶片真值、病灶预测二值、病灶真值二值、总时间
    """
    # 读取原始图像
    img_orig = np.array(Image.open(img_path).convert('RGB'))
    h, w = img_orig.shape[:2]

    # 读取真值
    leaf_true_pil = Image.open(leaf_gt_path).convert('L')
    if leaf_true_pil.size != (w, h):
        leaf_true_pil = leaf_true_pil.resize((w, h), Image.NEAREST)
    leaf_true = np.array(leaf_true_pil) > 0

    lesion_true_full = np.array(Image.open(lesion_full_path).convert('L'))
    if lesion_true_full.shape[:2] != (h, w):
        lesion_true_full = resize(lesion_true_full, (h, w), preserve_range=True, order=0).astype(np.uint8)
    lesion_true_binary = lesion_true_full > 0

    # ---- SAM 叶片分割 ----
    t_start = time.time()
    leaf_pred = predict_leaf_mask_sam(img_orig, mask_generator)

    # 根据预测叶片掩膜裁剪区域（涂黑背景）
    masked = img_orig.copy()
    masked[~leaf_pred] = [0, 0, 0]

    y_indices, x_indices = np.where(leaf_pred)
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

    # ---- 第二阶段病灶分割 ----
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
    lesion_pred_binary_cropped = lesion_pred_cropped > 0

    # 映射回全图
    lesion_pred_binary_full = np.zeros((h, w), dtype=bool)
    lesion_pred_binary_full[y1:y2+1, x1:x2+1] = lesion_pred_binary_cropped
    total_time = time.time() - t_start

    return leaf_pred, leaf_true, lesion_pred_binary_full, lesion_true_binary, total_time


# ------------------------- 主函数 -------------------------
def main():
    parser = argparse.ArgumentParser(description='End-to-end evaluation: SAM (largest mask) + Stage2 model')
    parser.add_argument('--sam_checkpoint', type=str, default=SAM_CHECKPOINT)
    parser.add_argument('--stage2_checkpoint', type=str, default=STAGE2_CKPT)
    parser.add_argument('--test_img_dir', type=str, default=str(TEST_IMG_DIR))
    parser.add_argument('--test_leaf_dir', type=str, default=str(TEST_LEAF_MASK_DIR))
    parser.add_argument('--test_lesion_dir', type=str, default=str(FULL_LESION_MASK_DIR))
    parser.add_argument('--padding', type=int, default=PADDING)
    parser.add_argument('--device', type=str, default=DEVICE)
    args = parser.parse_args()

    device = args.device
    print(f"Using device: {device}")

    # 加载SAM模型
    print("Loading SAM model...")
    mask_generator = load_sam_model(args.sam_checkpoint, SAM_MODEL_TYPE, device)

    # 加载第二阶段模型
    print("Loading stage2 model...")
    stage2_model = load_stage2_model(args.stage2_checkpoint, device)

    # 获取测试图像列表
    img_dir = Path(args.test_img_dir)
    leaf_dir = Path(args.test_leaf_dir)
    lesion_dir = Path(args.test_lesion_dir)
    img_files = sorted([f for f in img_dir.glob("*") if f.suffix.lower() in ['.jpg','.jpeg','.png']])
    valid_pairs = []
    for img_f in img_files:
        stem = img_f.stem
        leaf_f = leaf_dir / f"{stem}.png"
        lesion_f = lesion_dir / f"{stem}.png"
        if leaf_f.exists() and lesion_f.exists():
            valid_pairs.append((img_f, leaf_f, lesion_f))
    print(f"Found {len(valid_pairs)} valid test images")

    # 累积指标
    total_pa = 0.0
    total_iou_bg = 0.0
    total_iou_leaf = 0.0
    total_iou_lesion = 0.0
    total_miou = 0.0
    total_time = 0.0

    for img_path, leaf_path, lesion_path in tqdm(valid_pairs, desc="Evaluating"):
        leaf_pred, leaf_true, lesion_pred, lesion_true, proc_time = process_single_image(
            img_path, leaf_path, lesion_path, mask_generator, stage2_model, device, args.padding
        )
        # 叶片分割指标
        pa, iou_bg, iou_leaf = compute_pa_and_iou_binary(leaf_pred, leaf_true)
        # 病灶分割指标（全图二值）
        iou_lesion = compute_binary_iou(lesion_pred, lesion_true)
        miou = (iou_bg + iou_leaf + iou_lesion) / 3.0

        total_pa += pa
        total_iou_bg += iou_bg
        total_iou_leaf += iou_leaf
        total_iou_lesion += iou_lesion
        total_miou += miou
        total_time += proc_time
        del leaf_pred, leaf_true, lesion_pred, lesion_true
        torch.cuda.empty_cache()

    n = len(valid_pairs)
    avg_pa = total_pa / n * 100
    avg_iou_bg = total_iou_bg / n * 100
    avg_iou_leaf = total_iou_leaf / n * 100
    avg_iou_lesion = total_iou_lesion / n * 100
    avg_miou = total_miou / n * 100
    avg_time_ms = total_time / n * 1000

    print("\n" + "="*60)
    print("End-to-End Evaluation Results (SAM Leaf + Stage2 Lesion)")
    print("="*60)
    print(f"PA (Pixel Accuracy):            {avg_pa:.2f}%")
    print(f"IoU-B (Background):             {avg_iou_bg:.2f}%")
    print(f"IoU-leaf (Leaf):                {avg_iou_leaf:.2f}%")
    print(f"IoU-lesion (Lesion, binary):    {avg_iou_lesion:.2f}%")
    print(f"MIoU (Bg+Leaf+Lesion):          {avg_miou:.2f}%")
    print(f"Average Inference Time:         {avg_time_ms:.2f} ms per image")
    print("="*60)


if __name__ == "__main__":
    main()