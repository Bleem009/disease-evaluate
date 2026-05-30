#!/usr/bin/env python3
"""
三列对比拼图可视化工具 - 修复间距问题版本
关键修改: 去掉 tight_layout()，使用 subplots_adjust 精确控制
"""

import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import numpy as np
import argparse


def create_comparison_grid(base_dir, output_path='comparison_grid.png',
                           dpi=150, img_size=(400, 400)):
    base_dir = Path(base_dir)
    columns = ['origin', 'SAM', 'SAM3']

    for col in columns:
        col_path = base_dir / col
        if not col_path.exists():
            raise FileNotFoundError(f"文件夹不存在: {col_path}")

    first_col_dir = base_dir / columns[0]
    image_stems = sorted(list(set([
        f.stem for f in first_col_dir.glob('*')
        if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp',
                                '.gif', '.webp', '.tiff']
    ])))

    if not image_stems:
        raise ValueError(f"在 {first_col_dir} 中未找到图片文件")

    n_rows = len(image_stems)
    n_cols = len(columns)

    # 精确计算 figsize：基于图片尺寸，避免多余留白
    img_w_inch = img_size[0] / dpi
    img_h_inch = img_size[1] / dpi
    title_h_inch = 0.35  # 标题行高度（英寸）

    fig_w = n_cols * img_w_inch + 0.15  # 极小总边距
    fig_h = n_rows * img_h_inch + title_h_inch + 0.15

    fig, axes = plt.subplots(
        n_rows + 1, n_cols,
        figsize=(fig_w, fig_h),
        facecolor='white',
        gridspec_kw={
            'height_ratios': [0.3] + [1.0] * n_rows,
            'hspace': 0.1,   # 行间距0
            'wspace': 0.0    # 列间距0
        }
    )

    if n_rows == 1:
        axes = axes.reshape(n_rows + 1, n_cols)

    # ========== 第一行：列标题 ==========
    for col_idx, col_name in enumerate(columns):
        ax = axes[0, col_idx]
        ax.set_facecolor('white')
        ax.text(
            0.5, -0.2, col_name,
            ha='center', va='bottom',
            fontsize=6, fontweight='bold', color='black',
            transform=ax.transAxes
        )
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

    # ========== 后续行：图片内容 ==========
    for row_idx, img_stem in enumerate(image_stems):
        for col_idx, col_name in enumerate(columns):
            ax = axes[row_idx + 1, col_idx]
            ax.set_facecolor('white')

            col_dir = base_dir / col_name
            valid_files = [
                f for f in col_dir.glob(f"{img_stem}.*")
                if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp',
                                        '.gif', '.webp', '.tiff']
            ]

            if valid_files:
                img = Image.open(str(valid_files[0])).convert('RGB')
                img = img.resize(img_size, Image.LANCZOS)
                ax.imshow(np.array(img))
            else:
                ax.text(0.5, 0.5, f"缺失", ha='center', va='center',
                       color='red', fontsize=10, transform=ax.transAxes)

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    # ========== 关键修改: 用 subplots_adjust 替代 tight_layout ==========
    # tight_layout() 会覆盖 gridspec_kw 的 wspace/hspace 设置！
    plt.subplots_adjust(
        left=0.01,    # 左边距 1%
        right=0.99,   # 右边距 1%
        top=0.99,     # 上边距 1%
        bottom=0.01,  # 下边距 1%
        hspace=0.02,  # 行间距（相对于子图高度）
        wspace=0.02   # 列间距（相对于子图宽度）
    )

    # 保存时 bbox_inches=None 避免触发自动裁剪/调整
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_path, dpi=dpi,
        facecolor='white', edgecolor='none',
        bbox_inches=None  # 关键: 不用 'tight'，保持精确尺寸
    )
    plt.close()

    print(f"✅ 拼图已保存至: {output_path.absolute()}")
    print(f"   figsize: {fig_w:.2f}×{fig_h:.2f} inch")
    print(f"   布局: {n_rows} 行 × {n_cols} 列")
    print(f"   图片: {image_stems}")


def main():
    parser = argparse.ArgumentParser(description='三列对比拼图可视化')
    parser.add_argument('--input_dir', type=str,
                        default=r'C:\Users\86159\Desktop\visualizations')
    parser.add_argument('--output', type=str,
                        default=r'C:\Users\86159\Desktop\visualizations/comparison_grid2.png')
    parser.add_argument('--dpi', type=int, default=254)
    parser.add_argument('--img_size', type=int, default=400)
    args = parser.parse_args()

    create_comparison_grid(
        args.input_dir,
        args.output,
        args.dpi,
        img_size=(args.img_size, args.img_size)
    )


if __name__ == "__main__":
    main()