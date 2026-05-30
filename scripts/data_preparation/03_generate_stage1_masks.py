#!/usr/bin/env python3
"""
生成Stage 1的二值掩膜图像（支持多类别文件夹结构）
- 输入根目录结构：
    root/
        category1/
            images/
                img1.jpg
                img1.json
                ...
        category2/
            images/
                img2.jpg
                img2.json
                ...
- 输出根目录：output_dir/
        category1/
            img1.png
        category2/
            img2.png
"""

import sys
from pathlib import Path

# 添加项目根目录到 sys.path（保持原有逻辑）
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

from src.data.labelme_parser import LabelMeParser


def main():
    # 请根据实际情况修改以下路径
    root_dir = Path(r"C:\Users\86159\Desktop\毕设\数据集\images")   # 包含多个类别文件夹的根目录
    output_dir = Path(r"C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\stage1\masks")               # 输出根目录（会自动创建）

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    parser = LabelMeParser()

    # 支持的图片扩展名
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    # 遍历根目录下的每个子文件夹（类别）
    category_dirs = [d for d in root_dir.iterdir() if d.is_dir()]
    if not category_dirs:
        print(f"警告: {root_dir} 下没有子文件夹，请检查路径。")
        return

    for category_dir in tqdm(category_dirs, desc="处理类别"):
        # 类别文件夹下的 images 子文件夹
        images_dir = category_dir / "images"
        if not images_dir.exists() or not images_dir.is_dir():
            print(f"跳过 {category_dir.name}：未找到 images 子文件夹")
            continue

        # 创建该类别在输出目录中的子文件夹
        category_output_dir = output_dir / category_dir.name
        category_output_dir.mkdir(exist_ok=True)

        # 获取所有图片文件
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_dir.glob(f"*{ext}"))

        if not image_files:
            print(f"跳过 {category_dir.name}：images 文件夹中没有图片")
            continue

        # 处理每张图片
        for img_path in tqdm(image_files, desc=f"  {category_dir.name}", leave=False):
            # 对应的 JSON 文件（假设与图片同名，扩展名为 .json）
            json_path = images_dir / f"{img_path.stem}.json"
            if not json_path.exists():
                print(f"警告: 找不到标注文件 {json_path}，跳过")
                continue

            # 加载图片获取尺寸（直接用 PIL 打开）
            try:
                img = Image.open(img_path)
                w, h = img.size
            except Exception as e:
                print(f"无法读取图片 {img_path}: {e}")
                continue

            # 解析标注生成掩膜（0/1 数组）
            try:
                mask = parser.parse(str(json_path), (h, w))  # 注意 parse 方法可能需要路径字符串或 Path
            except Exception as e:
                print(f"解析标注失败 {json_path}: {e}")
                continue

            # 转换为 0-255 并保存
            mask_img = (mask * 255).astype(np.uint8)
            output_path = category_output_dir / f"{img_path.stem}.png"
            Image.fromarray(mask_img).save(output_path)

    # 统计生成的掩膜总数
    total_masks = sum(1 for _ in output_dir.glob("*/*.png"))
    print(f"\n总共生成 {total_masks} 个掩膜文件")
    print(f"保存位置: {output_dir}")


if __name__ == "__main__":
    main()