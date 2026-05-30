#!/usr/bin/env python3
"""
为Stage 2训练准备数据（裁剪叶片区域）
使用已划分好的病灶标注作为基准，在按类别组织的图片目录中查找对应图片和叶片标注，
裁剪叶片区域，生成裁剪后的叶片图像和病灶掩膜PNG。
可选将背景涂黑（仅保留叶片区域原色）。
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.data.labelme_parser import LabelMeParser


def find_image_file(root_dir: Path, filename_stem: str) -> Path:
    """
    在 root_dir 下递归搜索 filename_stem 对应的图片文件（支持多种扩展名）
    """
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        candidates = list(root_dir.glob(f'**/{filename_stem}{ext}'))
        if candidates:
            return candidates[0]  # 返回第一个找到的
    return None


def crop_leaf_region(image: np.ndarray, leaf_mask: np.ndarray, lesion_mask: np.ndarray,
                     padding: int = 10, mask_background: bool = True) -> tuple:
    """
    根据叶片掩膜裁剪图像和病灶掩膜，可选择将背景涂黑。
    Returns:
        cropped_image, cropped_lesion, (x1, y1, x2, y2)
    """
    # 找到叶片区域的边界
    y_indices, x_indices = np.where(leaf_mask > 0)
    if len(y_indices) == 0:
        return None, None, None

    y1, y2 = y_indices.min(), y_indices.max()
    x1, x2 = x_indices.min(), x_indices.max()

    # 添加padding，确保不超出原图边界
    h, w = leaf_mask.shape
    y1 = max(0, y1 - padding)
    y2 = min(h - 1, y2 + padding)
    x1 = max(0, x1 - padding)
    x2 = min(w - 1, x2 + padding)

    # 裁剪图像和掩膜
    cropped_image = image[y1:y2+1, x1:x2+1].copy()
    cropped_leaf = leaf_mask[y1:y2+1, x1:x2+1]
    cropped_lesion = lesion_mask[y1:y2+1, x1:x2+1]

    # 如果要求背景涂黑，将非叶片区域设为黑色
    if mask_background:
        cropped_image[cropped_leaf == 0] = 0

    return cropped_image, cropped_lesion, (x1, y1, x2, y2)


def process_split(split: str,
                  lesion_json_root: Path,
                  image_root: Path,
                  leaf_annotation_root: Path,
                  output_root: Path,
                  padding: int = 10,
                  mask_background: bool = True):
    """
    处理一个划分
    Args:
        split: 'train', 'val', 'test'
        lesion_json_root: 病灶标注根目录，其下应有 split 子文件夹，内含 JSON
        image_root: 图片根目录，将在此递归搜索图片
        leaf_annotation_root: 叶片标注根目录（假设与图片在同一目录）
        output_root: 输出根目录
        padding: 裁剪时的边距
        mask_background: 是否将背景涂黑
    """
    print(f"\n{'=' * 60}")
    print(f"Processing {split} split")
    print(f"{'=' * 60}")

    lesion_dir = lesion_json_root / split
    if not lesion_dir.exists():
        print(f"Warning: {lesion_dir} does not exist, skipping")
        return 0

    output_img_dir = output_root / split / 'images'
    output_mask_dir = output_root / split / 'labels'
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有病灶标注文件
    lesion_json_files = list(lesion_dir.glob("*.json"))
    print(f"Found {len(lesion_json_files)} lesion annotations")

    # 解析器
    leaf_parser = LabelMeParser(label_mapping={'leaf': 1, '叶片': 1})
    lesion_parser = LabelMeParser(label_mapping={
        'powdery': 1,  # 白粉病
        'rust': 2,  # 锈病
        'spot': 3,  # 斑点病
        'bacterial_blight': 4,  # 细菌性叶枯病
        # 根据实际类别添加，确保数值不重复
    })

    success_count = 0
    skipped = []

    for lesion_json_path in tqdm(lesion_json_files, desc=f"Processing {split}"):
        stem = lesion_json_path.stem

        # 查找对应的图片
        img_path = find_image_file(image_root, stem)
        if img_path is None:
            skipped.append(f"{stem}: image not found")
            continue

        # 加载原始图像
        image = np.array(Image.open(img_path).convert('RGB'))
        h, w = image.shape[:2]

        # 查找对应的叶片标注（假设与图片在同一目录）
        leaf_json_path = leaf_annotation_root / img_path.parent.relative_to(image_root) / f"{stem}.json"
        if not leaf_json_path.exists():
            # 如果不在相对路径下，尝试直接在 leaf_annotation_root 下搜索
            leaf_json_path = leaf_annotation_root / f"{stem}.json"
            if not leaf_json_path.exists():
                skipped.append(f"{stem}: leaf JSON not found")
                continue

        # 解析叶片标注
        leaf_mask = leaf_parser.parse(leaf_json_path, (h, w))

        if np.sum(leaf_mask) == 0:
            skipped.append(f"{stem}: empty leaf mask")
            continue

        # 解析病灶标注
        lesion_mask = lesion_parser.parse(lesion_json_path, (h, w))


        # 裁剪叶片区域
        cropped_img, cropped_lesion, bbox = crop_leaf_region(
            image, leaf_mask, lesion_mask, padding, mask_background
        )
        if cropped_img is None:
            skipped.append(f"{stem}: crop failed")
            continue

        # 保存裁剪后的图像
        img_out = output_img_dir / f"{stem}.png"
        Image.fromarray(cropped_img).save(img_out)

        # 保存病灶掩膜（0/255）
        lesion_out = output_mask_dir / f"{stem}.png"
        Image.fromarray(cropped_lesion.astype(np.uint8)).save(lesion_out)

        success_count += 1

    print(f"\nResults:")
    print(f"  Success: {success_count}/{len(lesion_json_files)}")
    print(f"  Skipped: {len(skipped)}")
    if skipped:
        print(f"  First 5 skipped: {skipped[:5]}")

    return success_count


def main():
    # ====== 请根据您的实际路径修改以下变量 ======
    lesion_json_root = Path(r"C:/Users/86159/PycharmProjects/disease_evaluation/temp_plantseg_labelme_new")
    image_root = Path(r"C:/Users/86159/Desktop/毕设/数据集/images")
    leaf_annotation_root = image_root  # 假设叶片标注与图片在同一目录
    output_root = Path(r"C:/Users/86159/PycharmProjects/disease_evaluation/data/processed/stage2")
    padding = 10
    mask_background = True  # 是否将背景涂黑
    splits = ['train', 'val', 'test']  # 根据 lesion_json_root 下的实际文件夹名称
    # =========================================

    print("=" * 60)
    print("Stage 2 Data Preparation (Cropped Leaves + Lesion Masks)")
    print("=" * 60)

    total = 0
    for split in splits:
        total += process_split(
            split,
            lesion_json_root,
            image_root,
            leaf_annotation_root,
            output_root,
            padding,
            mask_background
        )

    print(f"\n{'=' * 60}")
    print(f"All splits processed. Total: {total} images")
    print(f"{'=' * 60}")
    print(f"Output saved to: {output_root}")
    print("\nNext step: Train Stage 2 with:")
    print(f"  images: {output_root}/{{train,val,test}}/images/")
    print(f"  labels: {output_root}/{{train,val,test}}/labels/")


if __name__ == "__main__":
    main()