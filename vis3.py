import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import numpy as np
from PIL import Image

# ==================== 配置区域 ====================
base_dir = Path(r"C:\Users\86159\Desktop\图片示例\数据集实例")
crop_folders = ["水稻", "大豆", "辣椒"]
TARGET_SIZE = (300, 300)
# ==================================================

plt.rcParams['font.sans-serif'] = [
    'WenQuanYi Zen Hei',
    'Noto Sans CJK SC',
    'SimHei',
    'Microsoft YaHei',
    'DejaVu Sans'
]
plt.rcParams['axes.unicode_minus'] = False

all_images = []
all_labels = []

for folder in crop_folders:
    folder_path = base_dir / folder
    if not folder_path.exists():
        print(f"警告: 文件夹不存在: {folder_path}")
        continue

    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    image_files = sorted([f for f in folder_path.iterdir()
                          if f.suffix.lower() in exts and f.is_file()])

    for img_path in image_files:
        all_images.append(img_path)
        all_labels.append(img_path.stem)

n_images = len(all_images)
print(f"共找到 {n_images} 张图片")

if n_images == 0:
    print("未找到图片，请检查路径！")
    exit()

n_rows, n_cols = 3, 3

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12))
fig.patch.set_facecolor('white')

plt.subplots_adjust(wspace=0, hspace=0.15)

for ax in axes.flat:
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('white')

for idx in range(n_rows * n_cols):
    row, col = idx // n_cols, idx % n_cols
    ax = axes[row, col]

    if idx < n_images:
        img_path = all_images[idx]
        label = all_labels[idx]
        try:
            pil_img = Image.open(str(img_path)).convert('RGB')
            pil_img = pil_img.resize(TARGET_SIZE, Image.LANCZOS)
            img_array = np.array(pil_img)
            ax.imshow(img_array)
            ax.set_title(label, fontsize=12, fontweight='bold', pad=8)
        except Exception as e:
            ax.set_title(f"加载失败: {label}", fontsize=9, color='red')
            print(f"错误: {img_path} - {e}")
    else:
        ax.set_visible(False)

plt.savefig("dataset_visualization.png", dpi=200, facecolor='white', bbox_inches='tight')
plt.show()
print("已保存为 dataset_visualization.png")