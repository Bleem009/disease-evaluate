#!/usr/bin/env python3
"""
评估 SAM 模型在叶片分割测试集上的性能
官网风格可视化：分别保存 SAM 掩膜、自定义模型掩膜、真值掩膜
（紫色半透明+发光边界；真值使用绿色）
"""

import sys
from pathlib import Path

import torch

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2
from skimage.measure import regionprops
from skimage.transform import resize

import segmentation_models_pytorch as smp
from configs.stage1_leaf_config import Stage1Config
from src.data.transforms import get_validation_transforms

# ==================== 可视化配置（官网风格）====================
# SAM 和自定义模型使用紫色
VIS_CONFIG_MODEL = {
    'mask_alpha': 0.55,
    'mask_color': [128, 0, 128],          # 紫色
    'boundary_color': [255, 105, 180],    # 热粉色
    'boundary_thickness': 3,
    'glow_thickness': 6,
    'glow_color': [255, 20, 147],         # 深粉色
}

# 真值掩膜使用绿色（风格一致，仅颜色不同）
VIS_CONFIG_GT = {
    'mask_alpha': 0.55,
    'mask_color': [128, 0, 128],          # 紫色
    'boundary_color': [255, 105, 180],    # 热粉色
    'boundary_thickness': 3,
    'glow_thickness': 6,
    'glow_color': [255, 20, 147],         # 深粉色
}

# ---------- 质量评估函数（SAM 策略用）----------
def compute_boundary_gradient(mask, image):
    if mask.sum() == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel)
    boundary = (dilated - mask).astype(bool)
    if boundary.sum() == 0:
        return 0.0
    return float(grad_mag[boundary].mean())

def compute_internal_consistency(mask, image):
    if mask.sum() < 2:
        return 1e6
    pixels = image[mask]
    var = np.var(pixels, axis=0).sum() / (255 * 255 * 3)
    return float(var)

def compute_edge_contrast(mask, image):
    if mask.sum() == 0 or mask.sum() == mask.size:
        return 0.0
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel)
    eroded = cv2.erode(mask.astype(np.uint8), kernel)
    boundary_outer = (dilated - mask).astype(bool)
    boundary_inner = (mask - eroded).astype(bool)
    if boundary_inner.sum() == 0 or boundary_outer.sum() == 0:
        return 0.0
    mean_inner = image[boundary_inner].mean(axis=0)
    mean_outer = image[boundary_outer].mean(axis=0)
    return float(np.linalg.norm(mean_inner - mean_outer))

def compute_shape_compactness(mask):
    if mask.sum() == 0:
        return 0.0
    props = regionprops(mask.astype(np.uint8))[0]
    perimeter = props.perimeter
    area = props.area
    if area == 0:
        return 0.0
    compactness = (perimeter * perimeter) / (4 * np.pi * area)
    return 1.0 / (compactness + 1e-6)

def evaluate_mask_quality(mask, image):
    grad = compute_boundary_gradient(mask, image)
    internal_var = compute_internal_consistency(mask, image)
    contrast = compute_edge_contrast(mask, image)
    compact = compute_shape_compactness(mask)
    norm_grad = min(grad / 50.0, 1.0) if grad > 0 else 0.0
    norm_contrast = min(contrast / 50.0, 1.0) if contrast > 0 else 0.0
    norm_internal = 1.0 - min(internal_var, 1.0)
    norm_compact = min(compact, 1.0)
    area_ratio = mask.sum() / (image.shape[0] * image.shape[1])
    total = (0.1 * norm_grad + 0.1 * norm_contrast + 0.1 * norm_internal + 0.1 * norm_compact + 0.6 * area_ratio)
    return total

def compute_mcp_score(mask, image):
    masked_pixels = image[mask]
    if len(masked_pixels) < 100:
        color_var = 0
    else:
        color_var = np.var(masked_pixels, axis=0).sum()
    color_var_norm = min(color_var / (255 * 255 * 3), 1.0)
    h, w = mask.shape
    center_y, center_x = h // 2, w // 2
    y_indices, x_indices = np.where(mask)
    if len(y_indices) == 0:
        center_dist = 1e6
    else:
        centroid_y = np.mean(y_indices)
        centroid_x = np.mean(x_indices)
        center_dist = np.sqrt((centroid_y - center_y) ** 2 + (centroid_x - center_x) ** 2)
    max_dist = np.sqrt(h ** 2 + w ** 2) / 2
    position_score = 1.0 - min(center_dist / max_dist, 1.0)
    total_score = 0.5 * color_var_norm + 0.5 * position_score
    return total_score

# ==================== 官网风格可视化函数 ====================
def visualize_mask(image_np, mask, save_path=None, show=False, color_config=None):
    """
    生成掩膜可视化（半透明覆盖 + 发光边界）
    """
    if color_config is None:
        color_config = VIS_CONFIG_MODEL
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
        cv2.drawContours(glow_layer, contours, -1,
                         [255, 255, 255], 1)
    result = cv2.addWeighted(image_bgr, 1.0, mask_overlay, color_config['mask_alpha'], 0)
    result = cv2.addWeighted(result, 1.0, glow_layer, 0.85, 0)
    original_pixels = image_bgr[mask].astype(np.float32)
    tint_color = np.array(color_config['mask_color'], dtype=np.float32)
    tinted = original_pixels * 0.7 + tint_color * 0.3
    result[mask] = tinted.astype(np.uint8)
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), result)
        print(f"[Saved] {save_path}")
    if show:
        cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
        h, w = result.shape[:2]
        cv2.resizeWindow("Mask", min(w, 1200), min(h, 900))
        cv2.imshow("Mask", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return result

# ---------- 模型加载 ----------
def load_sam_model(checkpoint_path, model_type="vit_b", device="cuda"):
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator

def load_custom_stage1_model(checkpoint_path, device):
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
    return model, config

def predict_custom_stage1(model, config, image_np, device, threshold=0.5):
    h, w = image_np.shape[:2]
    transform = get_validation_transforms(config.img_size)
    transformed = transform(image=image_np, mask=np.zeros((h, w), dtype=np.uint8))
    input_tensor = transformed['image'].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask_resized = resize(probs, (h, w), preserve_range=True, order=1)
    mask = mask_resized > threshold
    return mask

def predict_sam_mask(image_np, mask_generator, strategy='largest'):
    masks = mask_generator.generate(image_np)
    if not masks:
        return np.zeros(image_np.shape[:2], dtype=bool)
    if strategy == 'largest':
        largest = max(masks, key=lambda x: x['area'])
        return largest['segmentation']
    elif strategy == 'quality':
        best_mask = None
        best_score = -1.0
        for ann in masks:
            mask = ann['segmentation']
            area_ratio = ann['area'] / (image_np.shape[0] * image_np.shape[1])
            if area_ratio < 0.01:
                continue
            score = evaluate_mask_quality(mask, image_np)
            if score > best_score:
                best_score = score
                best_mask = mask
        if best_mask is None:
            best_mask = max(masks, key=lambda x: x['area'])['segmentation']
        return best_mask
    elif strategy == 'mcp_score':
        best_mask = None
        best_score = -1.0
        for ann in masks:
            mask = ann['segmentation']
            score = compute_mcp_score(mask, image_np)
            if score > best_score:
                best_score = score
                best_mask = mask
        if best_mask is None:
            best_mask = max(masks, key=lambda x: x['area'])['segmentation']
        return best_mask
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

# ---------- 指标计算 ----------
def compute_iou(pred, true):
    inter = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    return inter / (union + 1e-6)

def compute_dice(pred, true):
    inter = np.logical_and(pred, true).sum()
    return 2 * inter / (pred.sum() + true.sum() + 1e-6)

def compute_pa(pred, true):
    correct = (pred == true).sum()
    return correct / pred.size

# ==================== 批量处理 ====================
def batch_process_and_visualize(img_dir, label_dir, sam_generator, custom_model, custom_config,
                                device, strategy, output_dir, show_vis=False):
    img_files = sorted([f for f in Path(img_dir).glob("*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    valid_pairs = []
    for f in img_files:
        label_f = Path(label_dir) / f"{f.stem}.png"
        if label_f.exists():
            valid_pairs.append((f, label_f))
        else:
            print(f"Warning: No label for {f.name}, skipping")
    print(f"Found {len(valid_pairs)} valid test images")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sam_iou, sam_dice, sam_pa = 0.0, 0.0, 0.0
    custom_iou, custom_dice, custom_pa = 0.0, 0.0, 0.0
    n = len(valid_pairs)

    for img_path, label_path in tqdm(valid_pairs, desc="Processing"):
        image_np = np.array(Image.open(img_path).convert('RGB'))
        true_mask = np.array(Image.open(label_path).convert('L')) > 127

        # 预测
        pred_sam = predict_sam_mask(image_np, sam_generator, strategy)
        pred_custom = predict_custom_stage1(custom_model, custom_config, image_np, device, threshold=0.5)

        # 计算指标
        sam_iou += compute_iou(pred_sam, true_mask)
        sam_dice += compute_dice(pred_sam, true_mask)
        sam_pa += compute_pa(pred_sam, true_mask)
        custom_iou += compute_iou(pred_custom, true_mask)
        custom_dice += compute_dice(pred_custom, true_mask)
        custom_pa += compute_pa(pred_custom, true_mask)

        stem = img_path.stem

        # 1. 保存 SAM 掩膜可视化（紫色）
        vis_sam = out_dir / f"{stem}_sam_{strategy}.png"
        visualize_mask(image_np, pred_sam, save_path=vis_sam, show=show_vis, color_config=VIS_CONFIG_MODEL)

        # 2. 保存自定义模型掩膜可视化（紫色）
        vis_custom = out_dir / f"{stem}_custom.png"
        visualize_mask(image_np, pred_custom, save_path=vis_custom, show=show_vis, color_config=VIS_CONFIG_MODEL)

        # 3. 保存真值掩膜可视化（绿色）
        vis_gt = out_dir / f"{stem}_gt.png"
        visualize_mask(image_np, true_mask, save_path=vis_gt, show=show_vis, color_config=VIS_CONFIG_GT)

    # 输出指标
    print("\n" + "=" * 70)
    print(f"SAM (strategy={strategy}) on {n} images:")
    print(f"  IoU       : {sam_iou/n:.4f}")
    print(f"  Dice      : {sam_dice/n:.4f}")
    print(f"  Pixel Acc : {sam_pa/n:.4f}")
    print("\nCustom Stage1 Model:")
    print(f"  IoU       : {custom_iou/n:.4f}")
    print(f"  Dice      : {custom_dice/n:.4f}")
    print(f"  Pixel Acc : {custom_pa/n:.4f}")
    print("=" * 70)
    print(f"All visualizations saved to: {out_dir}")

# ==================== 单张图片模式（也保存三种可视化）====================
def visualize_single_image(image_path, sam_checkpoint, custom_checkpoint,
                           model_type='vit_b', strategy='largest', device='cuda',
                           label_path=None, output_dir=None):
    image_path = Path(image_path)
    if output_dir is None:
        output_dir = image_path.parent / "single_vis"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading SAM...")
    sam_generator = load_sam_model(sam_checkpoint, model_type, device)
    print("Loading custom model...")
    custom_model, custom_config = load_custom_stage1_model(custom_checkpoint, device)

    print(f"Processing: {image_path.name}")
    image_np = np.array(Image.open(image_path).convert('RGB'))
    pred_sam = predict_sam_mask(image_np, sam_generator, strategy)
    pred_custom = predict_custom_stage1(custom_model, custom_config, image_np, device, threshold=0.5)

    stem = image_path.stem
    vis_sam = output_dir / f"{stem}_sam_{strategy}.png"
    visualize_mask(image_np, pred_sam, save_path=vis_sam, show=False, color_config=VIS_CONFIG_MODEL)
    vis_custom = output_dir / f"{stem}_custom.png"
    visualize_mask(image_np, pred_custom, save_path=vis_custom, show=False, color_config=VIS_CONFIG_MODEL)

    if label_path and Path(label_path).exists():
        true_mask = np.array(Image.open(label_path).convert('L')) > 127
        vis_gt = output_dir / f"{stem}_gt.png"
        visualize_mask(image_np, true_mask, save_path=vis_gt, show=False, color_config=VIS_CONFIG_GT)
        iou = compute_iou(pred_sam, true_mask)
        print(f"SAM IoU vs GT: {iou:.4f}")
    else:
        print("No GT label, skip GT visualization.")
        # 可选显示最后一张
        visualize_mask(image_np, pred_custom, save_path=None, show=True)

    print(f"Saved to {output_dir}")

# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sam_checkpoint', type=str, default=r'D:\edge_download\sam_vit_b_01ec64.pth')
    parser.add_argument('--custom_checkpoint', type=str,
                        default=r'C:\Users\86159\PycharmProjects\disease_evaluation\checkpoints\stage1\best_model.pth')
    parser.add_argument('--test_img_dir', type=str,
                        default=r'C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\用于论文\images')
    parser.add_argument('--test_label_dir', type=str,
                        default=r'C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\用于论文\labels')
    parser.add_argument('--model_type', type=str, default='vit_b')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--strategy', type=str, default='largest',
                        choices=['largest', 'quality', 'mcp_score'])
    parser.add_argument('--vis_output_dir', type=str,
                        default=r"C:\Users\86159\Desktop\leaf_visualizations",
                        help='Output directory for all visualizations')
    parser.add_argument('--show_vis', action='store_true', help='Show each visualization window')
    # 单张模式
    parser.add_argument('--visualize', type=str, default=None,
                        help='Single image mode: path to image')
    parser.add_argument('--vis_label', type=str, default=None)

    args = parser.parse_args()

    if args.visualize:
        visualize_single_image(
            image_path=args.visualize,
            sam_checkpoint=args.sam_checkpoint,
            custom_checkpoint=args.custom_checkpoint,
            model_type=args.model_type,
            strategy=args.strategy,
            device=args.device,
            label_path=args.vis_label,
            output_dir=args.vis_output_dir
        )
        return

    # 默认批量模式
    print(f"Using device: {args.device}")
    print(f"SAM strategy: {args.strategy}")
    print("Loading models...")
    sam_generator = load_sam_model(args.sam_checkpoint, args.model_type, args.device)
    custom_model, custom_config = load_custom_stage1_model(args.custom_checkpoint, args.device)

    batch_process_and_visualize(
        img_dir=args.test_img_dir,
        label_dir=args.test_label_dir,
        sam_generator=sam_generator,
        custom_model=custom_model,
        custom_config=custom_config,
        device=args.device,
        strategy=args.strategy,
        output_dir=args.vis_output_dir,
        show_vis=args.show_vis
    )

if __name__ == "__main__":
    main()