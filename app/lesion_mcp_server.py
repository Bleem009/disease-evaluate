#!/usr/bin/env python3
"""
病灶分割 MCP 服务器
将训练好的多类别病灶分割模型封装为 MCP 服务
"""

#!/usr/bin/env python3
import sys
from pathlib import Path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import io
import base64
from io import BytesIO
import torch
import numpy as np
from PIL import Image
from fastmcp import FastMCP
import segmentation_models_pytorch as smp
from skimage.transform import resize

from configs.stage2_lesion_config import Stage2Config
from src.data.transforms import get_validation_transforms

# ---------- 模型加载 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = Stage2Config()
num_classes = getattr(config, 'num_classes', 5)
model = smp.DeepLabV3Plus(
    encoder_name=config.encoder_name,
    encoder_weights=None,
    in_channels=3,
    classes=num_classes,
    activation=None
)
checkpoint_path = r"C:\Users\86159\PycharmProjects\disease_evaluation\checkpoints\stage2\best_model.pth"
ckpt = torch.load(str(checkpoint_path), map_location=device)
state_dict = ckpt['model_state_dict']
# 移除 'seg_model.' 前缀
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('seg_model.'):
        new_state_dict[k[9:]] = v
    else:
        new_state_dict[k] = v
model.load_state_dict(new_state_dict, strict=False)
model.to(device)
model.eval()
print(f"Lesion segmentation model loaded on {device}")

transform = get_validation_transforms(config.img_size)


def preprocess_image(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    orig_np = np.array(image)
    h, w = orig_np.shape[:2]
    transformed = transform(image=orig_np, mask=np.zeros((h, w), dtype=np.uint8))
    input_tensor = transformed['image'].unsqueeze(0).to(device)
    return input_tensor, h, w


def postprocess_mask(mask_tensor, orig_h, orig_w):
    mask_resized = mask_tensor.cpu().numpy().astype(np.uint8)
    mask_original = resize(mask_resized, (orig_h, orig_w), preserve_range=True, order=0).astype(np.uint8)
    return mask_original


mcp = FastMCP("LesionSegmentation")


@mcp.tool()
def segment_lesion(image_data: bytes) -> dict:
    """
    对输入的叶片图像进行病灶（病斑）分割，返回病灶掩膜的 base64 编码。

    参数:
        image_data: 图片的二进制数据

    返回:
        dict: 包含 mask_base64, mask_shape, height, width
    """
    input_tensor, h, w = preprocess_image(image_data)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_mask = torch.argmax(probs, dim=1).byte()
    lesion_mask = postprocess_mask(pred_mask[0], h, w)

    # 二值化：所有非背景（>0）视为病斑
    lesion_binary = (lesion_mask > 0).astype(np.uint8) * 255
    mask_img = Image.fromarray(lesion_binary)
    buffer = BytesIO()
    mask_img.save(buffer, format="PNG")
    mask_base64 = base64.b64encode(buffer.getvalue()).decode()

    return {
        "mask_base64": mask_base64,
        "mask_shape": lesion_mask.shape,
        "height": h,
        "width": w
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")