#!/usr/bin/env python3
"""
端到端测试：Mask2Former 叶片分割（零样本） + 第二阶段病灶分割
增加可视化：保存原始图像、语义掩膜、提取的叶片掩膜、真实叶片掩膜、病灶预测等
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
from pathlib import Path
import time
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from skimage.transform import resize
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse
import matplotlib.pyplot as plt

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
from configs.stage1_leaf_config import Stage1Config
from configs.stage2_lesion_config import Stage2Config
import torch.nn as nn
import segmentation_models_pytorch as smp

# ==================== 用户配置 ====================
MASK2FORMER_MODEL_NAME = "facebook/mask2former-swin-tiny-ade-semantic"
STAGE2_CKPT = r"C:\Users\86159\PycharmProjects\disease_evaluation\checkpoints\stage2\best_model.pth"
TEST_IMG_DIR = Stage1Config().test_img_dir
TEST_LEAF_MASK_DIR = Stage1Config().test_label_dir      # 叶片真实掩膜目录
FULL_LESION_MASK_DIR = Path(r"C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\stage2_fullmask\test\labels")  # 全图病灶真值
PADDING = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VIS_DIR = Path("./mask2former_vis")   # 可视化保存目录
# =================================================

# 第二阶段模型定义（带分类头）
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


def load_mask2former(device):
    """加载 Mask2Former 模型和处理器"""
    print(f"Loading Mask2Former from {MASK2FORMER_MODEL_NAME}")
    processor = Mask2FormerImageProcessor.from_pretrained(MASK2FORMER_MODEL_NAME)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(MASK2FORMER_MODEL_NAME).to(device)
    model.eval()
    return model, processor


import cv2
import numpy as np


def extract_leaf_mask_from_semantic(pred_mask, leaf_class_ids=[5, 10,15,17, 18, 67, 73]):
    """
    从语义分割掩膜中提取叶片掩膜，只保留指定类别，然后取最大连通域。
    pred_mask: (H, W) numpy array, 类别索引
    leaf_class_ids: 与叶片相关的类别索引列表
    """
    # 只保留叶片类别的像素
    leaf_candidate = np.isin(pred_mask, leaf_class_ids)
    if not np.any(leaf_candidate):
        return np.zeros_like(pred_mask, dtype=bool)

    # 可选：形态学闭运算连接邻近区域（可选）
    kernel = np.ones((5, 5), np.uint8)
    leaf_candidate = cv2.morphologyEx(leaf_candidate.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)

    # 提取连通域
    num_labels, labels = cv2.connectedComponents(leaf_candidate.astype(np.uint8))
    if num_labels <= 1:
        return leaf_candidate

    # 计算每个连通域的面积（排除背景0）
    areas = [np.sum(labels == i) for i in range(1, num_labels)]
    largest_label = np.argmax(areas) + 1
    leaf_mask = (labels == largest_label)
    return leaf_mask


@torch.no_grad()
def predict_leaf_mask_mask2former(image_np, model, processor, device):
    """使用 Mask2Former 预测叶片掩膜，同时返回原始语义掩膜"""
    pil_img = Image.fromarray(image_np)
    inputs = processor(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    pred_mask = processor.post_process_semantic_segmentation(outputs, target_sizes=[image_np.shape[:2]])[0]
    pred_mask_np = pred_mask.cpu().numpy().astype(np.uint8)
    leaf_mask = extract_leaf_mask_from_semantic(pred_mask_np, image_np)
    return leaf_mask, pred_mask_np


def compute_pa_and_iou_binary(pred_binary, true_binary):
    pred_flat = pred_binary.flatten()
    true_flat = true_binary.flatten()
    correct = np.sum(pred_flat == true_flat)
    pa = correct / len(pred_flat)
    # 背景 IoU
    pred_bg = (pred_flat == 0)
    true_bg = (true_flat == 0)
    inter_bg = np.sum(pred_bg & true_bg)
    union_bg = np.sum(pred_bg | true_bg)
    iou_bg = inter_bg / (union_bg + 1e-6)
    # 前景 IoU
    pred_fg = (pred_flat == 1)
    true_fg = (true_flat == 1)
    inter_fg = np.sum(pred_fg & true_fg)
    union_fg = np.sum(pred_fg | true_fg)
    iou_fg = inter_fg / (union_fg + 1e-6)
    return pa, iou_bg, iou_fg


def compute_binary_iou(pred_binary, true_binary):
    inter = np.sum(pred_binary & true_binary)
    union = np.sum(pred_binary | true_binary)
    return inter / (union + 1e-6)


@torch.no_grad()
def process_single_image(img_path, leaf_gt_path, lesion_full_path,
                         mask2former_model, mask2former_processor,
                         stage2_model, device, padding=10, vis_idx=None):
    """单张图像端到端处理，可选保存可视化"""
    img_orig = np.array(Image.open(img_path).convert('RGB'))
    h, w = img_orig.shape[:2]

    # 真实掩膜
    leaf_true_pil = Image.open(leaf_gt_path).convert('L')
    if leaf_true_pil.size != (w, h):
        leaf_true_pil = leaf_true_pil.resize((w, h), Image.NEAREST)
    leaf_true = np.array(leaf_true_pil) > 0

    lesion_true_full = np.array(Image.open(lesion_full_path).convert('L'))
    if lesion_true_full.shape[:2] != (h, w):
        lesion_true_full = resize(lesion_true_full, (h, w), preserve_range=True, order=0).astype(np.uint8)
    lesion_true_binary = lesion_true_full > 0

    # ---- 阶段1：Mask2Former 叶片分割 ----
    t_start = time.time()
    leaf_pred, semantic_mask = predict_leaf_mask_mask2former(img_orig, mask2former_model, mask2former_processor, device)

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

    # ---- 阶段2：病灶分割 ----
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

    # 可视化保存（前10张或指定索引）
    if vis_idx is not None and vis_idx < 10:
        save_visualization(img_orig, semantic_mask, leaf_pred, leaf_true,
                           lesion_pred_binary_full, lesion_true_binary,
                           img_path.stem, vis_idx)

    return leaf_pred, leaf_true, lesion_pred_binary_full, lesion_true_binary, total_time


def save_visualization(img_orig, semantic_mask, leaf_pred, leaf_true,
                       lesion_pred, lesion_true, stem, idx):
    """保存多子图可视化"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes[0,0].imshow(img_orig)
    axes[0,0].set_title('Original Image')
    axes[0,0].axis('off')
    # 语义掩膜（类别索引）
    im = axes[0,1].imshow(semantic_mask, cmap='tab20', vmin=0, vmax=semantic_mask.max())
    axes[0,1].set_title('Mask2Former Semantic Mask')
    axes[0,1].axis('off')
    plt.colorbar(im, ax=axes[0,1], ticks=np.unique(semantic_mask))
    # 提取的叶片预测
    axes[0,2].imshow(leaf_pred, cmap='gray')
    axes[0,2].set_title('Extracted Leaf Mask (Pred)')
    axes[0,2].axis('off')
    # 叶片真值
    axes[1,0].imshow(leaf_true, cmap='gray')
    axes[1,0].set_title('Leaf Ground Truth')
    axes[1,0].axis('off')
    # 病灶预测
    axes[1,1].imshow(lesion_pred, cmap='gray')
    axes[1,1].set_title('Lesion Prediction')
    axes[1,1].axis('off')
    # 病灶真值
    axes[1,2].imshow(lesion_true, cmap='gray')
    axes[1,2].set_title('Lesion Ground Truth')
    axes[1,2].axis('off')
    plt.tight_layout()
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(VIS_DIR / f"{stem}_vis.png", bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {VIS_DIR / f'{stem}_vis.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage2_checkpoint', type=str, default=STAGE2_CKPT)
    parser.add_argument('--test_img_dir', type=str, default=str(TEST_IMG_DIR))
    parser.add_argument('--test_leaf_dir', type=str, default=str(TEST_LEAF_MASK_DIR))
    parser.add_argument('--test_lesion_dir', type=str, default=str(FULL_LESION_MASK_DIR))
    parser.add_argument('--padding', type=int, default=PADDING)
    parser.add_argument('--device', type=str, default=DEVICE)
    args = parser.parse_args()

    device = args.device
    print(f"Using device: {device}")

    # 加载模型
    print("Loading Mask2Former...")
    mask2former_model, mask2former_processor = load_mask2former(device)
    print("Loading stage2 model...")
    stage2_model = load_stage2_model(args.stage2_checkpoint, device)

    # 获取测试图像列表
    img_dir = Path(args.test_img_dir)
    leaf_dir = Path(args.test_leaf_dir)
    lesion_dir = Path(args.test_lesion_dir)
    img_files = sorted([f for f in img_dir.glob("*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
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

    for idx, (img_path, leaf_path, lesion_path) in enumerate(tqdm(valid_pairs, desc="Evaluating")):
        leaf_pred, leaf_true, lesion_pred, lesion_true, proc_time = process_single_image(
            img_path, leaf_path, lesion_path,
            mask2former_model, mask2former_processor,
            stage2_model, device, args.padding, vis_idx=idx
        )
        # 叶片分割指标
        pa, iou_bg, iou_leaf = compute_pa_and_iou_binary(leaf_pred, leaf_true)
        # 病灶分割指标
        iou_lesion = compute_binary_iou(lesion_pred, lesion_true)
        miou = (iou_bg + iou_leaf + iou_lesion) / 3.0

        total_pa += pa
        total_iou_bg += iou_bg
        total_iou_leaf += iou_leaf
        total_iou_lesion += iou_lesion
        total_miou += miou
        total_time += proc_time

    n = len(valid_pairs)
    avg_pa = total_pa / n * 100
    avg_iou_bg = total_iou_bg / n * 100
    avg_iou_leaf = total_iou_leaf / n * 100
    avg_iou_lesion = total_iou_lesion / n * 100
    avg_miou = total_miou / n * 100
    avg_time_ms = total_time / n * 1000

    print("\n" + "="*60)
    print("End-to-End Evaluation: Mask2Former (zero-shot leaf) + Stage2 (lesion)")
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