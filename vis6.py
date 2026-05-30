#!/usr/bin/env python3
"""
多类别病灶分割推理脚本（支持多种病害）
输入：裁剪后的叶片图像 + 可选的真值掩码
输出：
    - 纯彩色预测掩码
    - 纯彩色真值掩码（如有提供）
    - 预测 overlay（原图 + 彩色掩码叠加）
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
import torch
from PIL import Image
from skimage.transform import resize

from configs.stage2_lesion_config import Stage2Config
from src.data.transforms import get_validation_transforms
import segmentation_models_pytorch as smp

# ================== 类别颜色映射 ==================
CLASS_COLORS = {
    0: [0, 0, 0],  # 背景：黑色
    1: [255, 0, 0],  # 白粉病：红色
    2: [0, 255, 0],  # 锈病：绿色
    3: [0, 0, 255],  # 斑点病：蓝色
    4: [255, 255, 0]  # 细菌性叶枯病：黄色
}


def load_multiclass_model(config, checkpoint_path, device):
    """加载多类别分割模型"""
    model = smp.DeepLabV3Plus(
        encoder_name=config.encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=config.num_classes,
        activation=None
    )
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def predict_multiclass(model, image_path, transform, device):
    """对单张图像进行多类别预测，返回掩码和原始图像"""
    orig_image = Image.open(image_path).convert('RGB')
    orig_np = np.array(orig_image)
    h, w = orig_np.shape[:2]

    transformed = transform(image=orig_np, mask=np.zeros((h, w), dtype=np.uint8))
    input_tensor = transformed['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)

    mask_resized = pred[0].cpu().numpy().astype(np.uint8)
    mask = resize(mask_resized, (h, w), preserve_range=True, order=0).astype(np.uint8)

    return mask, orig_np


def save_colored_mask(mask, save_path):
    """保存纯彩色掩码"""
    colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls, color in CLASS_COLORS.items():
        colored[mask == cls] = color

    Image.fromarray(colored).save(save_path)
    print(f"Saved colored mask to {save_path}")


def save_overlay(image, mask, save_path, alpha=0.5):
    """
    保存 overlay：原图 + 彩色掩码半透明叠加

    Args:
        image: 原始图像 (H, W, 3)
        mask: 类别索引掩码 (H, W)
        save_path: 保存路径
        alpha: 掩码透明度，默认 0.5
    """
    overlay = image.copy().astype(np.float32)

    for cls, color in CLASS_COLORS.items():
        if cls == 0:
            continue
        area = mask == cls
        if np.any(area):
            overlay[area] = overlay[area] * (1 - alpha) + np.array(color) * alpha

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(save_path)
    print(f"Saved overlay to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Multiclass Lesion Segmentation - Save Masks and Overlay')
    parser.add_argument('--image', type=str,
                        default=r"C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\stage2\test\images\soybean_frog_eye_leaf_spot_Bing_0015.png")
    parser.add_argument('--mask', type=str,
                        default=r"C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\stage2\test\labels\soybean_frog_eye_leaf_spot_Bing_0015.png")
    parser.add_argument('--checkpoint', type=str,
                        default=r'C:\Users\86159\PycharmProjects\disease_evaluation\checkpoints\stage2\可能输出多类病斑\best_model.pth')

    # 输出路径
    parser.add_argument('--output_pred', type=str, default='outputs/pred_mask.png',
                        help='Output path for colored prediction mask')
    parser.add_argument('--output_gt', type=str, default='outputs/gt_mask.png',
                        help='Output path for colored ground truth mask')
    parser.add_argument('--output_overlay', type=str, default='outputs/pred_overlay.png',
                        help='Output path for prediction overlay')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 加载配置
    config = Stage2Config()
    if not hasattr(config, 'num_classes'):
        raise AttributeError("Stage2Config must have 'num_classes' attribute")
    print(f"Number of classes: {config.num_classes}")

    # 获取验证集变换
    transform = get_validation_transforms(config.img_size)

    # 加载模型
    print("Loading model...")
    model = load_multiclass_model(config, args.checkpoint, device)

    # 预测（返回掩码和原图）
    print(f"Processing image: {args.image}")
    pred_mask, orig_image = predict_multiclass(model, args.image, transform, device)

    # 确保输出目录存在
    Path(args.output_pred).parent.mkdir(parents=True, exist_ok=True)

    # ========== 保存纯彩色预测掩码 ==========
    save_colored_mask(pred_mask, args.output_pred)

    # ========== 保存预测 overlay ==========
    save_overlay(orig_image, pred_mask, args.output_overlay)

    # ========== 如果提供了真值掩码，保存彩色真值掩码 ==========
    if args.mask:
        if args.mask.endswith('.npy'):
            gt_mask = np.load(args.mask).astype(np.uint8)
        else:
            gt_mask = np.array(Image.open(args.mask).convert('L')).astype(np.uint8)

        # 确保尺寸一致
        if gt_mask.shape != pred_mask.shape:
            gt_mask = resize(gt_mask, pred_mask.shape, preserve_range=True, order=0).astype(np.uint8)

        save_colored_mask(gt_mask, args.output_gt)

    print("Done.")


if __name__ == '__main__':
    main()