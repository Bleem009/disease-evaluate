#!/usr/bin/env python3
"""
多类别病灶分割推理脚本（支持多种病害）
输入：裁剪后的叶片图像
输出：彩色分割掩膜（不同病害显示不同颜色）和叠加图
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from skimage.transform import resize

from configs.stage2_lesion_config import Stage2Config
from src.data.transforms import get_validation_transforms
import segmentation_models_pytorch as smp


# ================== 修改点 1：定义类别名称和颜色映射 ==================
# 请根据您的实际类别顺序调整（背景为0，依次为各类病害）
CLASS_NAMES = {
    0: 'Background',
    1: 'Powdery Mildew',   # 白粉病
    2: 'Rust',             # 锈病
    3: 'Spot',             # 斑点病
    4: 'Bacterial Blight'  # 细菌性叶枯病
}
# 为每个类别分配不同的叠加颜色（BGR 或 RGB，这里用 RGB）
CLASS_COLORS = {
    0: [0, 0, 0],         # 背景：黑色
    1: [255, 0, 0],       # 白粉病：红色
    2: [0, 255, 0],       # 锈病：绿色
    3: [0, 0, 255],       # 斑点病：蓝色
    4: [255, 255, 0]      # 细菌性叶枯病：黄色
}


def load_multiclass_model(config, checkpoint_path, device):
    """加载多类别分割模型"""
    # ================== 修改点 2：模型输出通道数改为 num_classes ==================
    model = smp.DeepLabV3Plus(
        encoder_name=config.encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=config.num_classes,   # 使用配置中的类别数
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


def predict_multiclass(model, image_path, transform, device):
    """
    对单张图像进行多类别预测
    返回：类别索引掩膜 (H, W)，每个像素值为 0~num_classes-1
    """
    orig_image = Image.open(image_path).convert('RGB')
    orig_np = np.array(orig_image)
    h, w = orig_np.shape[:2]

    # 应用验证集变换（resize + 归一化）
    transformed = transform(image=orig_np, mask=np.zeros((h, w), dtype=np.uint8))
    input_tensor = transformed['image'].unsqueeze(0).to(device)  # [1,3,H,W]

    with torch.no_grad():
        logits = model(input_tensor)                     # [1, num_classes, H, W]
        probs = torch.softmax(logits, dim=1)             # 概率
        pred = torch.argmax(probs, dim=1)                # [1, H, W] 类别索引

    # 获取预测掩膜（在变换后的尺寸上）
    mask_resized = pred[0].cpu().numpy().astype(np.uint8)

    # 恢复到原始图像尺寸（使用最近邻插值保持类别值）
    mask = resize(mask_resized, (h, w), preserve_range=True, order=0).astype(np.uint8)

    return mask, orig_np


def visualize_multiclass(image, mask, save_path=None, show=True):
    """
    可视化多类别分割结果
    不同类别用不同颜色叠加，并显示图例
    """
    overlay = image.copy().astype(np.float32)
    alpha = 0.5

    # 对每个非背景类别进行着色叠加
    for cls, color in CLASS_COLORS.items():
        if cls == 0:
            continue
        area = mask == cls
        if np.any(area):
            overlay[area] = overlay[area] * (1 - alpha) + np.array(color) * alpha

    overlay = overlay.astype(np.uint8)

    # 创建彩色掩膜用于显示（每个类别一种颜色）
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls, color in CLASS_COLORS.items():
        colored_mask[mask == cls] = color

    # 统计每个类别的像素数
    pixel_counts = {}
    total_pixels = mask.size
    for cls, name in CLASS_NAMES.items():
        count = np.sum(mask == cls)
        percent = count / total_pixels * 100
        pixel_counts[name] = (count, percent)

    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    axes[1].imshow(colored_mask)
    axes[1].set_title('Predicted Classes')
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    # 添加图例文字（在图形下方）
    legend_text = "\n".join([f"{name}: {count:,} px ({percent:.1f}%)"
                             for name, (count, percent) in pixel_counts.items()])
    fig.text(0.5, 0.02, legend_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    if show:
        plt.show()
    else:
        plt.close()

    return fig


def main():
    parser = argparse.ArgumentParser(description='Multiclass Lesion Segmentation Inference')
    parser.add_argument('--image', type=str, required=True,
                        help='Input cropped leaf image path')
    parser.add_argument('--checkpoint', type=str,
                        default=r'C:\Users\86159\PycharmProjects\disease_evaluation\checkpoints\stage2\可能输出多类病斑\best_model.pth',
                        help='Path to multiclass model checkpoint')
    parser.add_argument('--output', type=str, default='outputs/multiclass_prediction.png',
                        help='Output visualization path')
    parser.add_argument('--no-display', action='store_true',
                        help='Do not display result (only save)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 加载配置
    config = Stage2Config()
    # 确保配置中有 num_classes
    if not hasattr(config, 'num_classes'):
        raise AttributeError("Stage2Config must have 'num_classes' attribute")
    print(f"Number of classes: {config.num_classes}")

    # 获取验证集变换
    transform = get_validation_transforms(config.img_size)

    # 加载模型
    print("Loading multiclass model...")
    model = load_multiclass_model(config, args.checkpoint, device)

    # 预测
    print(f"Processing image: {args.image}")
    mask, image_np = predict_multiclass(model, args.image, transform, device)

    # 可视化
    visualize_multiclass(image_np, mask, save_path=args.output, show=not args.no_display)

    print(f"Done. Visualization saved to {args.output}")


if __name__ == '__main__':
    main()