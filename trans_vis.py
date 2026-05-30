import cv2
import numpy as np

# 读取掩膜（保持原始深度，不要转彩色）
mask = cv2.imread(r'C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\stage2_fullmask\train\labels\soybean_frog_eye_leaf_spot_Bing_0058.png', cv2.IMREAD_UNCHANGED)

# 方案1：任意非零像素都转为255（最常用，适合你的情况）
_, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

# 方案2：如果原图是浮点数或特殊格式，用NumPy更稳妥
binary_mask = np.where(mask > 0, 255, 0).astype(np.uint8)

# 保存
cv2.imwrite('mask_binary_full.png', binary_mask)