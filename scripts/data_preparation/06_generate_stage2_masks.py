#!/usr/bin/env python3
"""
从原始的病灶JSON标注生成全图病灶掩膜（与原图同尺寸）
输出到 data/processed/stage2_fullmask/{split}/labels/ 目录
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.data.labelme_parser import LabelMeParser

def main():
    # 路径配置
    lesion_json_root = Path(r"C:/Users/86159/PycharmProjects/disease_evaluation/temp_plantseg_labelme_new")
    output_root = Path(r"C:/Users/86159/PycharmProjects/disease_evaluation/data/processed/stage2_fullmask")
    splits = ['train', 'val', 'test']

    # 病灶类别映射（与训练时一致）
    label_mapping = {
        'powdery': 1,
        'rust': 2,
        'spot': 3,
        'bacterial_blight': 4,
    }
    parser = LabelMeParser(label_mapping=label_mapping)

    for split in splits:
        json_dir = lesion_json_root / split
        if not json_dir.exists():
            print(f"Skip {split}: directory not found")
            continue

        output_label_dir = output_root / split / 'labels'
        output_label_dir.mkdir(parents=True, exist_ok=True)

        json_files = list(json_dir.glob("*.json"))
        print(f"Processing {split}: {len(json_files)} JSON files")

        for json_path in tqdm(json_files):
            # 解析获取图像尺寸和掩膜
            # LabelMeParser.parse 需要知道图像尺寸，我们可以先读取图像文件来获取
            # 但 JSON 中已包含 imageWidth/imageHeight
            import json
            with open(json_path, 'r') as f:
                data = json.load(f)
            h = data.get('imageHeight')
            w = data.get('imageWidth')
            if h is None or w is None:
                # 尝试从图像文件获取（需要知道图像路径）
                # 简化：跳过或手动处理
                print(f"Warning: {json_path} missing image dimensions, skip")
                continue

            mask = parser.parse(json_path, (h, w))  # 返回 (h,w) 的整数数组
            # 保存为 PNG
            out_path = output_label_dir / f"{json_path.stem}.png"
            Image.fromarray(mask.astype(np.uint8)).save(out_path)

        print(f"Saved full masks to {output_label_dir}")

if __name__ == "__main__":
    main()