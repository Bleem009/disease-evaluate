# muti_agent/mcp_tools.py
import aiohttp
import asyncio
import tempfile
import base64
import numpy as np
from PIL import Image
from typing import Dict, Any
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ---------- 原有的本地工具类（作为降级方案）----------
class MCPTool:
    def __init__(self, name: str, description: str, input_schema: Dict):
        self.name = name
        self.description = description
        self.input_schema = input_schema

    async def execute(self, params: Dict) -> Dict:
        raise NotImplementedError


class LeafSegmentationMCP(MCPTool):
    def __init__(self):
        super().__init__(
            name="leaf_segmentation",
            description="Segment leaf region from input image using trained model",
            input_schema={"type": "object", "properties": {"image_path": {"type": "string"}}}
        )
        self.api_url = "http://127.0.0.1:8000/predict/stage1/"

    async def execute(self, params: Dict) -> Dict:
        image_path = params["image_path"]
        async with aiohttp.ClientSession() as session:
            with open(image_path, 'rb') as f:
                form = aiohttp.FormData()
                form.add_field('file', f, filename='image.jpg')
                async with session.post(self.api_url, data=form) as resp:
                    result = await resp.json()
                    return {"mask_path": result["mask_path"], "confidence": 0.9}


class LesionSegmentationMCP(MCPTool):
    def __init__(self):
        super().__init__(
            name="lesion_segmentation",
            description="Segment lesion region from leaf image using trained model",
            input_schema={"type": "object", "properties": {"image_path": {"type": "string"}}}
        )
        self.api_url = "http://127.0.0.1:8000/predict/stage2/"

    async def execute(self, params: Dict) -> Dict:
        image_path = params["image_path"]
        async with aiohttp.ClientSession() as session:
            with open(image_path, 'rb') as f:
                form = aiohttp.FormData()
                form.add_field('file', f, filename='image.jpg')
                async with session.post(self.api_url, data=form) as resp:
                    result = await resp.json()
                    return {"mask_path": result["mask_path"], "confidence": 0.85}


class SAMSegmentationMCP(MCPTool):
    _model = None
    _lock = asyncio.Lock()

    def __init__(self):
        super().__init__(
            name="sam_segmentation",
            description="Segment any object in image using SAM (zero-shot)",
            input_schema={"type": "object", "properties": {"image_path": {"type": "string"}}}
        )
        self._init_model()

    def _init_model(self):
        if SAMSegmentationMCP._model is None:
            checkpoint_path = r"D:\edge_download\sam_vit_b_01ec64.pth"
            model_type = "vit_b"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            sam.to(device=device)
            SAMSegmentationMCP._model = SamAutomaticMaskGenerator(sam)
            print("SAM model loaded.")

    async def execute(self, params: Dict) -> Dict:
        image_path = params["image_path"]
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, self._predict_sync, image_path)
        return result

    def _predict_sync(self, image_path: str) -> Dict:
        image = np.array(Image.open(image_path).convert('RGB'))
        masks = SAMSegmentationMCP._model.generate(image)
        if not masks:
            return {"mask_path": None, "confidence": 0.0}
        largest = max(masks, key=lambda x: x['area'])
        best_mask = largest['segmentation']
        area_ratio = largest['area'] / (image.shape[0] * image.shape[1])
        confidence = min(area_ratio, 1.0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            mask_img = Image.fromarray((best_mask * 255).astype(np.uint8))
            mask_img.save(tmp.name)
            mask_path = tmp.name
        return {"mask_path": mask_path, "confidence": confidence}


# 本地回退工具注册表
LOCAL_MCP_TOOLS = {
    "leaf_segmentation": LeafSegmentationMCP(),
    "lesion_segmentation": LesionSegmentationMCP(),
    "sam_segmentation": SAMSegmentationMCP(),
}

# 全局标志：是否优先使用 MCP 客户端（True 表示使用 MCP 服务器）
USE_MCP_CLIENT = True


# muti_agent/mcp_tools.py (仅展示修改的部分，其他保持不变)

async def get_tool_executor(tool_name: str):
    """
    优先从 MCP 客户端获取工具，如果不可用则回退到本地工具。
    返回一个异步函数，接受参数 dict，返回 dict。
    """
    if USE_MCP_CLIENT:
        try:
            from muti_agent.mcp_client import init_mcp_tools, get_mcp_tool
            await init_mcp_tools()
            mcp_tool = get_mcp_tool(tool_name)
            if mcp_tool:
                async def executor(params: Dict):
                    import base64
                    image_path = params["image_path"]
                    with open(image_path, 'rb') as f:
                        image_data = f.read()
                    # 将二进制图片转为 base64 字符串传递
                    image_base64 = base64.b64encode(image_data).decode('utf-8')
                    # 调用 MCP 工具，参数名需与服务器定义一致
                    result = await mcp_tool.ainvoke({"image_base64": image_base64})
                    mask_base64 = result["mask_base64"]
                    mask_bytes = base64.b64decode(mask_base64)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                        tmp.write(mask_bytes)
                        mask_path = tmp.name
                    return {"mask_path": mask_path, "confidence": 0.9}
                return executor
        except Exception as e:
            print(f"[MCP Client] 初始化或获取工具失败: {e}，将使用本地工具")
    # 回退到本地工具
    tool = LOCAL_MCP_TOOLS.get(tool_name)
    if tool:
        return tool.execute
    raise ValueError(f"Tool {tool_name} not found")