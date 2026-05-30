import os
import random
import shutil
from pathlib import Path


def split_dataset(
    image_dir,
    label_dir,
    output_dir,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    image_extensions=('.jpg', '.jpeg', '.png', '.bmp'),
    label_extension='.png'   # 假设标签扩展名（可根据需要修改）
):
    """
    将图片和标签随机划分为训练/验证/测试集，并复制到输出目录。

    Args:
        image_dir: 原始图片文件夹路径
        label_dir: 原始标签文件夹路径（假设与图片同名但扩展名可能不同）
        output_dir: 输出根目录（将创建 train/val/test 子文件夹）
        train_ratio, val_ratio, test_ratio: 划分比例（需总和为1）
        image_extensions: 图片文件的扩展名列表
        label_extension: 标签文件的扩展名（如 .png 或 .json）
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例总和必须为1"

    # 获取所有图片文件（不包含扩展名）
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(Path(image_dir).glob(f'*{ext}'))
    image_stems = [p.stem for p in image_paths]  # 不带扩展名的文件名

    if len(image_stems) == 0:
        print("未找到图片文件，请检查路径和扩展名。")
        return

    # 随机打乱
    random.seed(42)  # 可固定种子以确保可重复
    random.shuffle(image_stems)

    # 计算各集合大小
    total = len(image_stems)
    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)
    train_stems = image_stems[:train_end]
    val_stems = image_stems[train_end:val_end]
    test_stems = image_stems[val_end:]

    # 创建输出子目录
    for split in ['train', 'val', 'test']:
        (Path(output_dir) / split / 'images').mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / split / 'labels').mkdir(parents=True, exist_ok=True)

    # 复制文件函数
    def copy_files(stems, split_name):
        for stem in stems:
            # 复制图片
            for ext in image_extensions:
                src_img = Path(image_dir) / f"{stem}{ext}"
                if src_img.exists():
                    dst_img = Path(output_dir) / split_name / 'images' / f"{stem}{ext}"
                    shutil.copy2(src_img, dst_img)
                    break
            else:
                print(f"警告: 未找到图片文件 {stem}")

            # 复制标签（假设标签文件与图片同名但扩展名为 label_extension）
            src_label = Path(label_dir) / f"{stem}{label_extension}"
            if src_label.exists():
                dst_label = Path(output_dir) / split_name / 'labels' / f"{stem}{label_extension}"
                shutil.copy2(src_label, dst_label)
            else:
                print(f"警告: 未找到标签文件 {src_label}")

    # 执行复制
    print(f"总图片数: {total}")
    print(f"训练集: {len(train_stems)} 张")
    print(f"验证集: {len(val_stems)} 张")
    print(f"测试集: {len(test_stems)} 张")

    copy_files(train_stems, 'train')
    copy_files(val_stems, 'val')
    copy_files(test_stems, 'test')

    print("划分完成！")

if __name__ == "__main__":
    # 请根据实际路径修改
    split_dataset(
        image_dir=r"C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\stage1\images",
        label_dir=r"C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\stage1\masks",
        output_dir=r"C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\stage1",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )