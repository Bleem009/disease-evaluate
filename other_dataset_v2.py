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
    可视化单个文件夹下的图片文件。
    - 图片上方显示文件名（清理后作为类别名）
    - 所有图片严格同样大小
    - 四周留白一致，无总标题，无文件夹名
    """
    root_path = Path(root_dir)

    # 直接搜集图片文件
    img_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        img_files.extend(root_path.glob(ext))

    if not img_files:
        print("未找到有效图片")
        return

    # 按文件名排序
    img_files = sorted(img_files)

    if cols_per_row is None:
        cols_per_row = len(img_files)

    n_rows = (len(img_files) + cols_per_row - 1) // cols_per_row
    n_cols = cols_per_row

    # 尺寸计算（单位：英寸）
    img_width_inch = img_size[0] / 100
    img_height_inch = img_size[1] / 100

    col_width = img_width_inch
    row_extra = 0.15
    row_height = img_height_inch + row_extra

    content_width = cols_per_row * col_width
    content_height = n_rows * row_height

    fig_width = content_width + 2 * margin_inch
    fig_height = content_height + 2 * margin_inch

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor='white')

    left_ratio = margin_inch / fig_width
    right_ratio = 1 - margin_inch / fig_width
    bottom_ratio = margin_inch / fig_height
    top_ratio = 1 - margin_inch / fig_height
    plt.subplots_adjust(left=left_ratio, right=right_ratio,
                        bottom=bottom_ratio, top=top_ratio)

    gs = GridSpec(n_rows, n_cols, figure=fig,
                  width_ratios=[col_width] * cols_per_row,
                  height_ratios=[row_height] * n_rows,
                  hspace=0.01,
                  wspace=0.08)

    for idx, img_path in enumerate(img_files):
        row_idx = idx // cols_per_row
        col_idx = idx % cols_per_row

        ax_img = fig.add_subplot(gs[row_idx, col_idx])

        img = Image.open(img_path).convert('RGB')
        img_resized = resize_to_exact_size(img, img_size)

        ax_img.imshow(img_resized)
        ax_img.axis('off')

        # 使用文件名（不含扩展名）作为类别名
        cat_name = img_path.stem
        clean_name = clean_category_name(cat_name)
        ax_img.set_title(clean_name, fontsize=14,fontweight='bold', pad=8, loc='center')
        ax_img.set_facecolor('white')

    if save_path:
        plt.savefig(save_path, facecolor='white', pad_inches=0)
        print(f"已保存至：{save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    root_directory = r"C:\Users\86159\Desktop\图片示例\plantseg"
    visualize_samples_grid(root_directory, img_size=(512, 512),
                           save_path="samples_grid.png", margin_inch=0.15)