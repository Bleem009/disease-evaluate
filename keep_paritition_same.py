#!/usr/bin/env python3
"""
根据第一阶段划分的图片，将对应的病灶JSON文件按train/val/test组织到目标目录。
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import shutil
from tqdm import tqdm


def organize_jsons(stage1_img_root: Path, source_json_dir: Path, target_json_root: Path):
    """
    Args:
        stage1_img_root: 第一阶段图片根目录，应包含 train/val/test 子文件夹，每个内有 images/
        source_json_dir: 存放所有病灶 JSON 的目录（文件名与图片主名相同）
        target_json_root: 目标根目录，将在其中创建 train/val/test 子文件夹，存放对应的 JSON
    """
    splits = ['train', 'val', 'test']
    for split in splits:
        img_dir = stage1_img_root / split / 'images'
        if not img_dir.exists():
            print(f"警告: {img_dir} 不存在，跳过 {split}")
            continue

        # 获取该划分下所有图片的主文件名（不含扩展名）
        stems = [p.stem for p in img_dir.glob("*") if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        print(f"\n{split} 划分: {len(stems)} 张图片")

        target_split_dir = target_json_root / split
        target_split_dir.mkdir(parents=True, exist_ok=True)

        success = 0
        for stem in tqdm(stems, desc=f"复制 {split} 病灶标注"):
            src = source_json_dir / f"{stem}.json"
            if not src.exists():
                print(f"  警告: 未找到 {stem}.json，跳过")
                continue
            dst = target_split_dir / f"{stem}.json"
            shutil.copy2(src, dst)
            success += 1

        print(f"  -> 成功复制 {success}/{len(stems)} 个 JSON 到 {target_split_dir}")


if __name__ == "__main__":
    # ========== 请根据实际路径修改 ==========
    stage1_img_root = Path(r"C:/Users/86159/PycharmProjects/disease_evaluation/data/processed/stage1")
    source_json_dir = Path(r"C:/Users/86159/PycharmProjects/disease_evaluation/temp_plantseg_labelme")  # 所有病灶JSON所在目录
    target_json_root = Path(r"C:/Users/86159/PycharmProjects/disease_evaluation/temp_plantseg_labelme_new")
    # =====================================

    organize_jsons(stage1_img_root, source_json_dir, target_json_root)