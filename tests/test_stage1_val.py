#!/usr/bin/env python3
"""
评估阶段一模型（叶片分割）在测试集上的性能
- 计算指标：PA, IoU(leaf), Dice(leaf), IoU(background)
- 测量平均推理时间（毫秒/张），包含 GPU 预热和 CUDA 同步
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import time
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.transform import resize

from configs.stage1_leaf_config import Stage1Config

# ==================== 用户配置 ====================
STAGE1_CHECKPOINT = r"C:\Users\86159\PycharmProjects\disease_evaluation\checkpoints\stage1\best_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WARMUP_ITERATIONS = 50   # GPU预热次数
# =================================================


def load_stage1_model(checkpoint_path, device):
    """从配置创建模型，并加载训练好的权重"""
    config = Stage1Config()
    print(f"Creating model with encoder: {config.encoder_name}")
    model = smp.DeepLabV3Plus(
        encoder_name=config.encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    )
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    # 移除可能的 'module.' 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k[7:] if k.startswith('module.') else k
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def warmup_model(model, device, img_size, warmup_iters=50):
    """使用随机输入预热模型，确保GPU进入稳定状态"""
    print(f"Warming up model with {warmup_iters} iterations...")
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
    for _ in range(warmup_iters):
        with torch.no_grad():
            _ = model(dummy_input)
    torch.cuda.synchronize()
    print("Warmup finished.")


def compute_metrics(pred_binary, true_binary):
    """计算二分类分割指标（针对叶片前景）"""
    pred_flat = pred_binary.flatten()
    true_flat = true_binary.flatten()

    correct = np.sum(pred_flat == true_flat)
    pa = correct / len(pred_flat)

    tp = np.sum((pred_flat == 1) & (true_flat == 1))
    fp = np.sum((pred_flat == 1) & (true_flat == 0))
    fn = np.sum((pred_flat == 0) & (true_flat == 1))

    iou_leaf = tp / (tp + fp + fn + 1e-6)
    dice_leaf = 2 * tp / (2 * tp + fp + fn + 1e-6)

    tn = np.sum((pred_flat == 0) & (true_flat == 0))
    iou_bg = tn / (tn + fp + fn + 1e-6)

    return pa, iou_leaf, dice_leaf, iou_bg


@torch.no_grad()
def evaluate_stage1(model, test_img_dir, test_mask_dir, img_size, device):
    """遍历测试集，计算平均指标和平均推理时间"""
    img_files = sorted([f for f in test_img_dir.glob("*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    valid_pairs = []
    for img_f in img_files:
        stem = img_f.stem
        mask_f = test_mask_dir / f"{stem}.png"
        if mask_f.exists():
            valid_pairs.append((img_f, mask_f))
        else:
            print(f"Warning: mask not found for {img_f.name}")

    if not valid_pairs:
        raise RuntimeError(f"No valid test pairs found in {test_img_dir} and {test_mask_dir}")
    print(f"Found {len(valid_pairs)} test images with masks.")

    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    total_pa = total_iou_leaf = total_dice_leaf = total_iou_bg = 0.0
    total_inference_time = 0.0

    for img_path, mask_path in tqdm(valid_pairs, desc="Evaluating Stage1"):
        # 读取图像（不计入推理时间）
        img = np.array(Image.open(img_path).convert('RGB'))
        h, w = img.shape[:2]

        # 读取真实掩膜（不计入推理时间）
        mask = np.array(Image.open(mask_path).convert('L'))
        mask_binary = (mask > 0).astype(np.uint8)
        if mask_binary.shape[:2] != (h, w):
            mask_binary = resize(mask_binary, (h, w), preserve_range=True, order=0).astype(np.uint8)

        # ---- 开始计时（包含预处理、模型推理、后处理） ----
        torch.cuda.synchronize()
        t_start = time.time()

        transformed = transform(image=img)
        input_tensor = transformed['image'].unsqueeze(0).to(device)
        logits = model(input_tensor)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        prob_resized = resize(prob, (h, w), preserve_range=True, order=1)
        pred_binary = (prob_resized > 0.5).astype(np.uint8)

        torch.cuda.synchronize()
        inference_time = time.time() - t_start
        # ---- 计时结束 ----

        pa, iou_leaf, dice_leaf, iou_bg = compute_metrics(pred_binary, mask_binary)

        total_pa += pa
        total_iou_leaf += iou_leaf
        total_dice_leaf += dice_leaf
        total_iou_bg += iou_bg
        total_inference_time += inference_time

    n = len(valid_pairs)
    avg_inference_time_ms = total_inference_time / n * 1000

    return {
        'PA': total_pa / n,
        'IoU_leaf': total_iou_leaf / n,
        'Dice_leaf': total_dice_leaf / n,
        'IoU_background': total_iou_bg / n,
        'Inference_time_ms': avg_inference_time_ms
    }


def main():
    config = Stage1Config()
    print(f"Using device: {DEVICE}")
    print(f"Encoder: {config.encoder_name}")
    print(f"Image size: {config.img_size}")
    print(f"Test image dir: {config.test_img_dir}")
    print(f"Test mask dir: {config.test_label_dir}")

    if not config.test_img_dir.exists():
        raise FileNotFoundError(f"Test image directory not found: {config.test_img_dir}")
    if not config.test_label_dir.exists():
        raise FileNotFoundError(f"Test mask directory not found: {config.test_label_dir}")

    model = load_stage1_model(STAGE1_CHECKPOINT, DEVICE)

    # ===== GPU 预热 =====
    warmup_model(model, DEVICE, config.img_size, WARMUP_ITERATIONS)

    # ===== 正式评估 =====
    results = evaluate_stage1(model, config.test_img_dir, config.test_label_dir, config.img_size, DEVICE)

    print("\n" + "=" * 50)
    print("Stage1 Model Test Results (Leaf Segmentation)")
    print("=" * 50)
    print(f"Pixel Accuracy (PA):           {results['PA'] * 100:.4f}%")
    print(f"IoU (Leaf):                    {results['IoU_leaf'] * 100:.4f}%")
    print(f"Dice (Leaf):                   {results['Dice_leaf'] * 100:.4f}%")
    print(f"IoU (Background):              {results['IoU_background'] * 100:.4f}%")
    print(f"Average Inference Time:        {results['Inference_time_ms']:.2f} ms per image")
    print("=" * 50)


if __name__ == "__main__":
    main()