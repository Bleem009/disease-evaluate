from PIL import Image
import os
from pathlib import Path

def resize_to_square_stretch(img_path, output_path, target_size=256):
    """
    方式1：强制拉伸至正方形（可能变形）
    target_size: 正方形边长（如 256）
    """
    img = Image.open(img_path)
    img_resized = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    img_resized.save(output_path)
    print(f"已保存（拉伸）：{output_path}")

def process_directory(input_dir, output_dir=None, target_size=256, extensions=('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
    """
    对输入目录下的所有图片进行正方形拉伸处理
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录（若为 None，则输出到同一目录下的 'resize' 子文件夹）
        target_size: 目标正方形边长
        extensions: 需要处理的文件扩展名元组
    """
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise NotADirectoryError(f"{input_dir} 不是一个有效的目录")

    # 确定输出目录
    if output_dir is None:
        # 修复：使用 / 运算符正确拼接路径
        out_path = input_path / "resize"
    else:
        out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 遍历目录下所有文件
    for file_path in input_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            # 生成输出文件名：原名_stretch.扩展名
            stem = file_path.stem
            ext = file_path.suffix
            output_file = out_path / f"{stem}_stretch{ext}"
            resize_to_square_stretch(str(file_path), str(output_file), target_size)

    print("全部处理完成！")

# ----------------- 使用示例 -----------------
if __name__ == "__main__":
    # 请修改为你的实际目录路径
    input_directory = r"C:\Users\86159\Desktop\lesion_visualizations"
    process_directory(input_directory, output_dir=None, target_size=256)