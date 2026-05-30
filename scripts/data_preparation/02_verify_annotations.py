# scripts/data_preparation/02_verify_annotations.py
# !/usr/bin/env python3
"""
验证标注文件的正确性
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from src.data.labelme_parser import LabelMeParser


def verify_single_annotation(image_path: Path, label_path: Path, save_vis: bool = False):
    """验证单个标注"""
    print(f"\nVerifying: {image_path.name}")

    # 加载图像
    image = Image.open(image_path).convert('RGB')
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    print(f"  Image size: {w}x{h}")

    # 解析标注
    parser = LabelMeParser()
    mask = parser.parse(label_path, (h, w))

    # 统计
    labeled_pixels = np.sum(mask > 0)
    total_pixels = h * w
    ratio = labeled_pixels / total_pixels * 100

    print(f"  Labeled pixels: {labeled_pixels:,} / {total_pixels:,} ({ratio:.2f}%)")

    # 可视化
    if save_vis:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img_array)
        axes[0].set_title('Image')
        axes[0].axis('off')

        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title(f'Mask ({ratio:.1f}%)')
        axes[1].axis('off')

        # 叠加
        overlay = img_array.copy()
        overlay[mask > 0] = [255, 0, 0]
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')

        plt.tight_layout()
        vis_dir = Path("outputs/verification")
        vis_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(vis_dir / f"{image_path.stem}_verify.png", dpi=150)
        plt.close()

    return {
        'image_size': (w, h),
        'labeled_pixels': int(labeled_pixels),
        'ratio': ratio
    }


def main():
    # 验证Stage 1（叶片标注）
    print("=" * 50)
    print("Verifying Stage 1 (Leaf) Annotations")
    print("=" * 50)

    img_dir = Path("data/raw/field_images/train")
    lbl_dir = Path("data/raw/leaf_annotations/train")

    stats = []
    for img_path in sorted(img_dir.glob("*.jpg"))[:5]:  # 验证前5张
        lbl_path = lbl_dir / f"{img_path.stem}.json"
        if lbl_path.exists():
            stat = verify_single_annotation(img_path, lbl_path, save_vis=True)
            stats.append(stat)

    # 验证Stage 2（病灶标注，如果有）
    print("\n" + "=" * 50)
    print("Verifying Stage 2 (Lesion) Annotations")
    print("=" * 50)

    # 汇总统计
    if stats:
        avg_ratio = np.mean([s['ratio'] for s in stats])
        print(f"\nAverage labeled ratio: {avg_ratio:.2f}%")


if __name__ == "__main__":
    main()