

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ==================== 读取掩码 ====================
leaf_mask = np.array(Image.open(r"C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\stage1\train\labels\soybean_frog_eye_leaf_spot_Bing_0058.png").convert('L'))   # 叶片掩码
lesion_mask = np.array(Image.open(r"C:\Users\86159\PycharmProjects\disease_evaluation\mask_binary_full.png").convert('L')) # 病灶掩码


# 归一化为 0/1
leaf_mask = (leaf_mask > 0).astype(np.uint8)
lesion_mask = (lesion_mask > 0).astype(np.uint8)

# 只保留叶片内的病灶
lesion_on_leaf = lesion_mask & leaf_mask

# ==================== 生成彩色图 ====================
h, w = leaf_mask.shape

# 叶片：绿色，其余黑色
leaf_vis = np.zeros((h, w, 3), dtype=np.uint8)
leaf_vis[leaf_mask == 1] = [0, 255, 0]

# 病灶（仅叶片内）：红色，其余黑色
lesion_vis = np.zeros((h, w, 3), dtype=np.uint8)
lesion_vis[lesion_on_leaf == 1] = [255, 0, 0]

# ==================== 保存 ====================
Image.fromarray(leaf_vis).save("leaf_green.png")
Image.fromarray(lesion_vis).save("lesion_red.png")

# ==================== 显示（无标题） ====================
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(leaf_vis)
axes[0].axis('off')
axes[1].imshow(lesion_vis)
axes[1].axis('off')
plt.subplots_adjust(wspace=0.05)
plt.show()