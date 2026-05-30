import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from pathlib import Path


def resize_to_exact_size(img, target_size):
    """强制缩放到目标尺寸，不保持原始比例"""
    return img.resize(target_size, Image.Resampling.LANCZOS)


def clean_category_name(name):
    """清理类别名，使其更易读"""
    cleaned = name.replace('__', ' ').replace('_', ' ')
    cleaned = cleaned.title()
    return cleaned


def visualize_samples_grid(root_dir, img_size=(256, 256), cols_per_row=None,
                           save_path=None, margin_inch=0.15, dpi=150):
    """
    美观版本：所有图片严格同样大小，四周留白一致，无总标题。
    - 每行对应一个子文件夹
    - 每行左侧竖写文件夹名（不加粗）
    - 图片上方显示类别名（与图片间距小）
    - 行间距极小，文件名列与图片列间距小
    - 图片整体尺寸紧凑，四周留白相等
    """
    # 搜集数据
    rows_data = []
    max_cats = 0
    for sub_dir in sorted(Path(root_dir).iterdir()):
        if not sub_dir.is_dir():
            continue
        folder_name = sub_dir.name
        items = []
        for cat_dir in sorted(sub_dir.iterdir()):
            if not cat_dir.is_dir():
                continue
            img_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                img_files.extend(cat_dir.glob(ext))
            if not img_files:
                continue
            items.append((cat_dir.name, img_files[0]))
        if items:
            rows_data.append((folder_name, items))
            max_cats = max(max_cats, len(items))

    if not rows_data:
        print("未找到有效数据")
        return

    if cols_per_row is None:
        cols_per_row = max_cats

    n_rows = len(rows_data)
    n_cols = 1 + cols_per_row

    # 尺寸计算（单位：英寸）
    img_width_inch = img_size[0] / 100
    img_height_inch = img_size[1] / 100

    left_col_width = 0.3          # 左侧竖排文字列宽度
    col_width = img_width_inch
    # 每行额外高度：用于显示类别名
    row_extra = 0.1
    row_height = img_height_inch + row_extra

    # 内容区域总宽度和总高度
    content_width = left_col_width + cols_per_row * col_width
    content_height = n_rows * row_height

    # 四周留白统一为 margin_inch
    fig_width = content_width + 2 * margin_inch
    fig_height = content_height + 5 * margin_inch

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor='white')

    # 设置子图区域，使得内容区域四周留白等于 margin_inch
    left_ratio = margin_inch / fig_width
    right_ratio = 1 - margin_inch / fig_width
    bottom_ratio = margin_inch / fig_height
    top_ratio = 1 - margin_inch / fig_height
    plt.subplots_adjust(left=left_ratio, right=right_ratio,
                        bottom=bottom_ratio, top=top_ratio)

    # 在内容区域内部划分 GridSpec（使用相对坐标 0~1）
    gs = GridSpec(n_rows, n_cols, figure=fig,
                  width_ratios=[left_col_width] + [col_width] * cols_per_row,
                  height_ratios=[row_height] * n_rows,
                  hspace=0.02,   # 极小行间距
                  wspace=0.08)   # 列间距小（文件名离图片近）

    for row_idx, (folder_name, items) in enumerate(rows_data):
        # 左侧：竖写文件夹名，不加粗
        ax_text = fig.add_subplot(gs[row_idx, 0])
        ax_text.axis('off')
        ax_text.text(0.5, 0.5, folder_name,
                     ha='center', va='center',
                     fontsize=12, rotation=90)
        ax_text.set_facecolor('white')

        # 右侧图片
        for col_idx, (cat_name, img_path) in enumerate(items):
            if col_idx >= cols_per_row:
                break
            ax_img = fig.add_subplot(gs[row_idx, 1 + col_idx])

            # 强制缩放到目标尺寸（不保持比例）
            img = Image.open(img_path).convert('RGB')
            img_resized = resize_to_exact_size(img, img_size)

            ax_img.imshow(img_resized)
            ax_img.axis('off')

            # 类别名与图片间距小（pad=4）
            clean_name = clean_category_name(cat_name)
            ax_img.set_title(clean_name, fontsize=10, pad=8, loc='center')
            ax_img.set_facecolor('white')

    if save_path:
        plt.savefig(save_path, facecolor='white', pad_inches=0)   # pad_inches=0 避免额外边距
        print(f"已保存至：{save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    root_directory = r"C:\Users\86159\Desktop\图片示例"
    visualize_samples_grid(root_directory, img_size=(256, 256),
                           save_path="samples_grid.png", margin_inch=0.15)