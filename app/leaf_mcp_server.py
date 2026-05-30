#!/usr/bin/env python3
"""
叶片分割 MCP 服务器
使用 FastMCP 将训练好的叶片分割模型封装为 MCP 服务
"""

#!/usr/bin/env python3
import sys
from pathlib import Path
# 添加项目根目录到 Python 路径（必须在其他导入之前）
sys.path.insert(0, str(Path(__file__).parent.parent))

import io
import base64
from io import BytesIO
import torch
import numpy as np
from PIL import Image
from fastmcp import FastMCP
import segmentation_models_pytorch as smp
from skimage.transform import resize

from configs.stage1_leaf_config import Stage1Config
from src.data.transforms import get_validation_transforms

# ---------- 模型加载 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = Stage1Config()
model = smp.DeepLabV3Plus(
    encoder_name=config.encoder_name,
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None
)
checkpoint_path = r"C:\Users\86159\PycharmProjects\disease_evaluation\checkpoints\stage1\best_model.pth"
ckpt = torch.load(str(checkpoint_path), map_location=device)
state_dict = ckpt['model_state_dict']

new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('seg_model.'):
        new_state_dict[k[9:]] = v
    else:
        new_state_dict[k] = v
model.load_state_dict(new_state_dict, strict=False)
model.to(device)
model.eval()
print(f"Leaf segmentation model loaded on {device}")

transform = get_validation_transforms(config.img_size)


def preprocess_image(image_bytes: bytes):
    """预处理：读取图片，resize，归一化，转 tensor"""
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    orig_np = np.array(image)
    h, w = orig_np.shape[:2]
    transformed = transform(image=orig_np, mask=np.zeros((h, w), dtype=np.uint8))
    input_tensor = transformed['image'].unsqueeze(0).to(device)
    return input_tensor, h, w


def postprocess_mask(mask_tensor, orig_h, orig_w):
    """将模型输出的 mask 恢复到原始尺寸"""
    mask_resized = mask_tensor.cpu().numpy().astype(np.uint8)
    mask_original = resize(mask_resized, (orig_h, orig_w), preserve_range=True, order=0).astype(np.uint8)
    return mask_original


# ---------- 创建 MCP 服务器 ----------
mcp = FastMCP("LeafSegmentation")


@mcp.tool()
def segment_leaf(image_data: bytes) -> dict:
    """
    对输入的植物叶片图像进行分割，返回叶片掩膜的 base64 编码。

    参数:
        image_data: 图片的二进制数据

    返回:
        dict: 包含 mask_base64, mask_shape, height, width
    """
    input_tensor, h, w = preprocess_image(image_data)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits)
        pred_mask = (probs[0, 0] > 0.5).byte()
    leaf_mask = postprocess_mask(pred_mask, h, w)

    # 将掩膜保存为 PNG 并编码为 base64
    mask_img = Image.fromarray((leaf_mask * 255).astype(np.uint8))
    buffer = BytesIO()
    mask_img.save(buffer, format="PNG")
    mask_base64 = base64.b64encode(buffer.getvalue()).decode()

    return {
        "mask_base64": mask_base64,
        "mask_shape": leaf_mask.shape,
        "height": h,
        "width": w
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")