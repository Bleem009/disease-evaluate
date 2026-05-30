#!/usr/bin/env python3
"""
SAM 零样本分割 MCP 服务器
将 Segment Anything Model (SAM) 封装为 MCP 服务
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import io
import base64
from io import BytesIO
import torch
import numpy as np
from PIL import Image
from fastmcp import FastMCP
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ---------- 配置 ----------
CHECKPOINT_PATH = r"D:\edge_download\sam_vit_b_01ec64.pth"   # 与原有路径一致
MODEL_TYPE = "vit_b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- 模型加载 ----------
print(f"[SAM Server] Loading SAM on {DEVICE}...")
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)
print("[SAM Server] SAM model loaded.")

# ---------- 创建 MCP 服务器 ----------
mcp = FastMCP("SAMSegmentation")

@mcp.tool()
def segment_with_sam(image_data: bytes) -> dict:
    """
    使用 SAM 对输入图像进行零样本分割，返回面积最大掩膜的 base64 编码。

    参数:
        image_data: 图片的二进制数据

    返回:
        dict: 包含 mask_base64, mask_shape, height, width
    """
    # 解码图片
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    image_np = np.array(image)
    h, w = image_np.shape[:2]

    # 生成所有掩膜
    masks = mask_generator.generate(image_np)
    if not masks:
        return {"mask_base64": "", "mask_shape": (0,0), "height": h, "width": w}

    # 选择面积最大的掩膜
    largest = max(masks, key=lambda x: x['area'])
    mask = largest['segmentation']   # bool array (H, W)

    # 转换为二值图像 (0/255)
    mask_uint8 = (mask * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask_uint8)
    buffer = BytesIO()
    mask_img.save(buffer, format="PNG")
    mask_base64 = base64.b64encode(buffer.getvalue()).decode()

    return {
        "mask_base64": mask_base64,
        "mask_shape": mask.shape,
        "height": h,
        "width": w
    }

if __name__ == "__main__":
    mcp.run(transport="stdio")