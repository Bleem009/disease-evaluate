import os
import json
from muti_agent.mcp_tools import get_tool_executor
import uuid
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from muti_agent.state import AgentState


load_dotenv()
llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
    openai_api_key=os.getenv("DOUBAO_API_KEY"),
    base_url=os.getenv("DOUBAO_BASE_URL"),
    temperature=0.3
)


import cv2
import numpy as np
from skimage.measure import regionprops
from scipy.spatial.distance import cdist

# ---------- 多掩膜在无监督情况下的判优 ----------
def compute_boundary_gradient(mask_path: str, image_path: str) -> float:
    """
    计算掩膜边界附近的平均梯度强度（越高越好）
    """
    # 读取原图灰度图
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    # 计算梯度幅值
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx ** 2 + sobely ** 2)

    # 读取掩膜
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127
    if mask.sum() == 0:
        return 0.0
    # 提取边界像素（膨胀后减去原掩膜）
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel)
    boundary = (dilated - mask).astype(bool)
    if boundary.sum() == 0:
        return 0.0
    # 边界处的平均梯度
    mean_grad = grad_mag[boundary].mean()
    return float(mean_grad)


def compute_internal_consistency(mask_path: str, image_path: str) -> float:
    """
    计算掩膜内部像素的颜色方差（越低越好）
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return 1e6
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127
    if mask.sum() == 0:
        return 1e6
    # 提取掩膜内像素的 RGB 值
    pixels = img[mask]
    if len(pixels) < 2:
        return 1e6
    # 计算 RGB 三个通道的方差之和（归一化）
    var = np.var(pixels, axis=0).sum() / (255 * 255 * 3)
    return float(var)


def compute_edge_contrast(mask_path: str, image_path: str) -> float:
    """
    计算掩膜边界内外两侧的颜色差异（越大越好）
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127
    if mask.sum() == 0 or mask.sum() == mask.size:
        return 0.0
    # 提取边界（内外各一个像素）
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel)
    eroded = cv2.erode(mask.astype(np.uint8), kernel)
    boundary_outer = (dilated - mask).astype(bool)
    boundary_inner = (mask - eroded).astype(bool)
    if boundary_inner.sum() == 0 or boundary_outer.sum() == 0:
        return 0.0
    mean_inner = img[boundary_inner].mean(axis=0)
    mean_outer = img[boundary_outer].mean(axis=0)
    contrast = np.linalg.norm(mean_inner - mean_outer)
    return float(contrast)


def compute_shape_compactness(mask_path: str) -> float:
    """
    计算掩膜的紧致度（周长²/(4π面积)），叶片通常较紧凑，越接近1越好
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127
    if mask.sum() == 0:
        return 1e6
    props = regionprops(mask.astype(np.uint8))[0]
    perimeter = props.perimeter
    area = props.area
    if area == 0:
        return 1e6
    compactness = (perimeter * perimeter) / (4 * np.pi * area)
    # 返回倒数，使得值越大表示越紧凑
    return 1.0 / (compactness + 1e-6)


def evaluate_mask_quality(mask_path: str, image_path: str) -> dict:
    """
    综合评估掩膜质量，返回总分（越高越好）和各指标
    """
    if not mask_path or not os.path.exists(mask_path):
        return {"total_score": 0.0, "details": {}}
    # 计算各项指标（需要归一化到相似范围）
    grad = compute_boundary_gradient(mask_path, image_path)
    #internal_var = compute_internal_consistency(mask_path, image_path)
    #contrast = compute_edge_contrast(mask_path, image_path)
    compact = compute_shape_compactness(mask_path)

    # 归一化：grad 和 contrast 通常几十到几百，取 log 或直接使用
    # internal_var 在 0-1 之间，越小越好 -> 用 1 - var
    # compact 通常在 0.2-2 之间，越大越好
    norm_grad = min(grad / 50.0, 1.0) if grad > 0 else 0.0
    #norm_contrast = min(contrast / 50.0, 1.0) if contrast > 0 else 0.0
    #norm_internal = 1.0 - min(internal_var, 1.0)
    norm_compact = min(compact, 1.0)

    # 加权总分（可根据需要调整权重）
    total = (0.6 * norm_grad +
             #0.3 * norm_contrast +
             #0.2 * norm_internal +
             0.4 * norm_compact)

    details = {
        "gradient": round(grad, 2),
        #"internal_var": round(internal_var, 4),
        #"contrast": round(contrast, 2),
        "compactness": round(compact, 4),
        "norm_grad": round(norm_grad, 4),
        #"norm_contrast": round(norm_contrast, 4),
        #"norm_internal": round(norm_internal, 4),
        "norm_compact": round(norm_compact, 4)
    }
    return {"total_score": round(total, 4), "details": details}


# ---------- 规划节点 ----------
async def planning_node(state: AgentState) -> AgentState:
    query = state["user_query"]
    history = state.get("conversation_history", [])
    hist_text = "\n".join([f"{m['role']}: {m['content']}" for m in history[-5:]])

    # 获取当前已有的结果（帮助模型判断是否需要重复执行）
    has_leaf_mask = state.get("leaf_mask") is not None
    has_lesion_mask = state.get("lesion_mask") is not None
    has_analysis = state.get("analysis_result") is not None

    prompt = f"""你是任务规划专家。根据用户的问题，从以下可用任务中选择需要执行的任务，以 JSON 列表形式返回。
可用任务：
- leaf_segmentation: 分割出叶片区域，生成叶片掩膜（用于定位叶片位置、轮廓、显示叶片）
- lesion_segmentation: 分割出病灶区域（病斑），生成病灶掩膜（用于显示病斑位置、大小）
- severity_assessment: 计算病斑面积占叶片面积的比例（用于病害严重度评估）

当前状态：
- 是否已有叶片分割结果：{has_leaf_mask}
- 是否已有病斑分割结果：{has_lesion_mask}
- 是否已有严重度评估结果：{has_analysis}

对话历史（最近5条）：
{hist_text}

用户最新问题：{query}

规则（按优先级从高到低）：
1. **语义映射**：
   - 如果用户要求“显示叶片”、“叶片位置”、“叶片轮廓”、“提取叶片”、“叶片在哪” → 选择 ["leaf_segmentation"]
   - 如果用户要求“病斑位置”、“病灶显示”、“病斑分割”、“哪里有病” → 选择 ["lesion_segmentation"]
   - 如果用户要求“病害程度”、“严重不严重”、“病害百分比”、“病情评估” → 选择 ["leaf_segmentation", "lesion_segmentation", "severity_assessment"]
2. **复用已有结果**：如果用户没有要求重新计算，且已有相关结果，尽量不再重复执行（例如已有叶片分割结果，用户问“叶片位置”可以直接回答，不需要再次分割）。但如果用户明确要求“重新分割”、“再次分割”，则需要重新执行。
3. **依赖关系**：severity_assessment 必须同时有叶片和病斑掩膜，如果缺少则自动补充前置任务。
4. **上下文连贯**：如果用户说“在此基础上”、“接着”、“然后”，表示要基于上一次操作继续，应当参考历史中的最近任务和当前状态。
5. **默认行为**：如果无法判断，且涉及植物病害，默认执行完整流程（叶片+病斑+评估）。

只返回 JSON 列表，不要有其他内容。

示例：
- 用户：“叶片的位置” → ["leaf_segmentation"]
- 用户：“在此基础上进行叶片分割”（历史中有图片但无叶片结果） → ["leaf_segmentation"]
- 用户：“重新计算病害度” → ["leaf_segmentation", "lesion_segmentation", "severity_assessment"]
- 用户：“显示病斑”（已有叶片和病斑结果） → [] （直接回答，无需再次分割）
"""
    resp = await llm.ainvoke([HumanMessage(content=prompt)])
    raw_content = resp.content
    print(f"\n[LLM原始响应] 用户问题: {query}")
    print(f"[LLM原始响应] {raw_content}")

    # 解析 JSON，容错处理
    tasks = None
    try:
        # 提取第一个 [...] 内容
        import re
        match = re.search(r'\[.*?\]', raw_content, re.DOTALL)
        if match:
            tasks = json.loads(match.group())
        else:
            tasks = json.loads(raw_content)
        if not isinstance(tasks, list):
            tasks = None
    except Exception as e:
        print(f"JSON 解析失败: {e}")

    # 兜底：根据语义关键词
    if not tasks:
        lower_q = query.lower()
        # 定义农业/病害相关关键词
        agri_keywords = ["叶片", "轮廓", "位置", "显示叶片", "病斑", "病灶", "病斑分割",
                         "严重", "程度", "百分比", "病害度", "植物", "作物", "病害", "虫害"]
        if any(k in lower_q for k in ["叶片", "轮廓", "位置", "显示叶片"]):
            tasks = ["leaf_segmentation"]
        elif any(k in lower_q for k in ["病斑", "病灶", "病斑分割"]):
            tasks = ["lesion_segmentation"]
        elif any(k in lower_q for k in ["严重", "程度", "百分比", "病害度"]):
            tasks = ["leaf_segmentation", "lesion_segmentation", "severity_assessment"]
        elif any(k in lower_q for k in agri_keywords):
            # 包含其他农业关键词但未匹配上述具体类别，默认完整流程
            tasks = ["leaf_segmentation", "lesion_segmentation", "severity_assessment"]
        else:
            # 完全不相关的问题，保持空任务列表
            tasks = []
            print("[规划节点] 用户问题与农业病害无关，任务列表为空")

    # 严重度评估必须包含前置任务
    # 如果任务列表中包含 severity_assessment，则确保 leaf_segmentation 和 lesion_segmentation 都存在
    #if tasks and "severity_assessment" in tasks:
        #if "leaf_segmentation" not in tasks:
            #tasks.append("leaf_segmentation")
        #if "lesion_segmentation" not in tasks:
            #tasks.append("lesion_segmentation")

    # 去重
    tasks = list(set(tasks))
    state["tasks"] = tasks
    state["planned_tasks"] = tasks.copy()
    state["current_step"] = "planning_complete"
    print(f"[最终任务列表] {tasks}\n")
    return state


import asyncio
import numpy as np
from PIL import Image


def calculate_iou(mask1_path: str, mask2_path: str) -> float:
    """计算两个掩膜图像的 IoU（交并比）"""
    if not mask1_path or not mask2_path:
        return 0.0
    m1 = np.array(Image.open(mask1_path).convert('L')) > 127
    m2 = np.array(Image.open(mask2_path).convert('L')) > 127
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return intersection / union if union > 0 else 0.0


# ---------- 推理节点 ----------
# nodes.py 中的 inference_node 完整实现
import os
import tempfile
import numpy as np
from PIL import Image
from muti_agent.state import AgentState


async def inference_node(state: AgentState) -> AgentState:
    """
    推理节点：根据规划好的 tasks 执行叶片分割、病斑分割等任务。
    支持：
    - 跳过已存在的分割结果（避免重复计算）
    - 叶片分割后自动用于病斑分割的裁剪
    - 处理重试/重新分割的请求
    - 通过 MCP 客户端调用分割服务（优先），降级到本地 HTTP API
    """
    query = state.get("user_query", "")
    tasks = state.get("tasks", [])
    if not tasks:
        print("[推理节点] 无任务，跳过推理")
        state["inference_done"] = True
        state["current_step"] = "inference_complete"
        return state

    # 处理“重新/再次”等重试意图：清空已有结果，强制重新推理
    if any(k in query for k in ["重新", "再次", "重试"]):
        if state.get("inference_done"):
            print("[推理节点] 用户要求重新执行，清空已有分割结果")
            state["inference_done"] = False
            state["error"] = None
            state["leaf_mask"] = None
            state["lesion_mask"] = None
            state["analysis_result"] = None

    # 如果已经完成推理且没有要求重做，直接返回
    if state.get("inference_done"):
        print("[推理节点] 推理已完成，跳过")
        return state

    # 验证图片路径
    img_path = state.get("image_path")
    if not img_path or not os.path.exists(img_path):
        state["error"] = "图片不存在，请先上传图片或确保会话中保留了上一张图片。"
        print("[推理节点] 错误：图片路径无效")
        return state

    executed = []

    # ---------- 1. 叶片分割 ----------
    if "leaf_segmentation" in tasks:
        existing_leaf = state.get("leaf_mask")
        if existing_leaf and os.path.exists(existing_leaf) and "重新" not in query:
            print("[推理节点] 复用已有的叶片掩膜")
            executed.append("leaf_segmentation")
        else:
            try:
                # 获取工具执行器（优先 MCP，降级本地）
                your_tool = await get_tool_executor("leaf_segmentation")
                sam_tool = await get_tool_executor("sam_segmentation")

                # 并行调用专用模型和 SAM
                your_task = asyncio.create_task(your_tool({"image_path": img_path}))
                sam_task = asyncio.create_task(sam_tool({"image_path": img_path}))
                your_res, sam_res = await asyncio.gather(your_task, sam_task)

                your_mask = your_res.get("mask_path")
                sam_mask = sam_res.get("mask_path")

                # 调试输出
                if your_mask and sam_mask:
                    iou = calculate_iou(your_mask, sam_mask)
                    print(f"[叶片分割调试] IoU: {iou:.4f}")
                print(f"[叶片分割调试] 专用模型质量总分: {evaluate_mask_quality(your_mask, img_path)['total_score']}")
                print(f"[叶片分割调试] SAM模型质量总分: {evaluate_mask_quality(sam_mask, img_path)['total_score']}")

                # 质量评估择优
                your_quality = evaluate_mask_quality(your_mask, img_path)
                sam_quality = evaluate_mask_quality(sam_mask, img_path)
                if your_quality['total_score'] >= sam_quality['total_score']:
                    selected_mask = your_mask
                    reason = f"Your model (quality={your_quality['total_score']} > {sam_quality['total_score']})"
                else:
                    selected_mask = sam_mask
                    reason = f"SAM (quality={sam_quality['total_score']} > {your_quality['total_score']})"

                state["leaf_mask"] = selected_mask
                print(f"[叶片分割] 集成选择: {reason}")
                executed.append("leaf_segmentation")

                # 调试：保存两个模型的掩膜（可选）
                if os.getenv("DEBUG", "false").lower() == "true":
                    save_dir = Path("temp_vis")
                    save_dir.mkdir(exist_ok=True)
                    if your_mask and Path(your_mask).exists():
                        import shutil
                        your_debug = save_dir / f"debug_your_model_{uuid.uuid4().hex}.png"
                        shutil.copy(your_mask, your_debug)
                        print(f"专用模型掩膜保存: {your_debug}")
                    if sam_mask and Path(sam_mask).exists():
                        sam_debug = save_dir / f"debug_sam_{uuid.uuid4().hex}.png"
                        shutil.copy(sam_mask, sam_debug)
                        print(f"SAM模型掩膜保存: {sam_debug}")

            except Exception as e:
                print(f"[叶片分割] 失败: {e}")
                state["error"] = f"叶片分割失败: {e}"
                state["executed_tasks"] = executed
                state["inference_done"] = True
                return state

    # ---------- 2. 病灶分割 ----------
    if "lesion_segmentation" in tasks:
        existing_lesion = state.get("lesion_mask")
        if existing_lesion and os.path.exists(existing_lesion):
            print("[推理节点] 复用已有的病灶掩膜")
            executed.append("lesion_segmentation")
        else:
            leaf_mask_path = state.get("leaf_mask")
            # 若无叶片掩膜，临时生成一个（仅用于约束，不保存到state）
            temp_leaf_mask = None
            if not leaf_mask_path or not os.path.exists(leaf_mask_path):
                try:
                    temp_tool = await get_tool_executor("leaf_segmentation")
                    temp_res = await temp_tool({"image_path": img_path})
                    temp_leaf_mask = temp_res.get("mask_path")
                    if temp_leaf_mask and os.path.exists(temp_leaf_mask):
                        print("[推理节点] 已自动生成临时叶片掩膜（仅用于约束病灶分割）")
                    else:
                        temp_leaf_mask = None
                except Exception as e:
                    print(f"[推理节点] 生成临时叶片掩膜失败: {e}")

            effective_leaf_mask = leaf_mask_path if leaf_mask_path and os.path.exists(leaf_mask_path) else temp_leaf_mask

            if effective_leaf_mask and os.path.exists(effective_leaf_mask):
                try:
                    # 裁剪原图，只保留叶片区域
                    original = np.array(Image.open(img_path).convert('RGB'))
                    leaf_mask = np.array(Image.open(effective_leaf_mask).convert('L')) > 127
                    masked = original.copy()
                    masked[~leaf_mask] = [0, 0, 0]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        Image.fromarray(masked).save(tmp.name)
                        cropped_path = tmp.name

                    # 调用病灶分割 API（MCP 优先）
                    tool = await get_tool_executor("lesion_segmentation")
                    res = await tool({"image_path": cropped_path})
                    lesion_raw_path = res.get("mask_path")

                    # 将病灶掩膜 resize 回原图尺寸，并用叶片掩膜过滤
                    from skimage.transform import resize
                    lesion_raw = np.array(Image.open(lesion_raw_path).convert('L')) > 0
                    h, w = original.shape[:2]
                    lesion_resized = (resize(lesion_raw.astype(float), (h, w), order=0, preserve_range=True) > 0.5)
                    final_lesion = lesion_resized & leaf_mask

                    # 保存最终掩膜
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                        Image.fromarray((final_lesion * 255).astype(np.uint8)).save(tmp.name)
                        state["lesion_mask"] = tmp.name
                    executed.append("lesion_segmentation")
                    print("[推理节点] 基于叶片掩膜的病灶分割完成（含自动生成掩膜）")

                    # 清理临时文件
                    os.unlink(cropped_path)
                    if temp_leaf_mask and os.path.exists(temp_leaf_mask):
                        os.unlink(temp_leaf_mask)

                except Exception as e:
                    print(f"[推理节点] 基于叶片掩膜的病灶分割失败: {e}，尝试降级方案")
                    # 降级：全图分割 + 叶片过滤
                    tool = await get_tool_executor("lesion_segmentation")
                    res = await tool({"image_path": img_path})
                    lesion_full_path = res.get("mask_path")
                    lesion_full = np.array(Image.open(lesion_full_path).convert('L')) > 0
                    leaf_mask = np.array(Image.open(effective_leaf_mask).convert('L')) > 127
                    final_lesion = lesion_full & leaf_mask
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                        Image.fromarray((final_lesion * 255).astype(np.uint8)).save(tmp.name)
                        state["lesion_mask"] = tmp.name
                    executed.append("lesion_segmentation")
                    if temp_leaf_mask and os.path.exists(temp_leaf_mask):
                        os.unlink(temp_leaf_mask)
            else:
                # 没有任何叶片掩膜，直接全图分割（不推荐，但保底）
                tool = await get_tool_executor("lesion_segmentation")
                res = await tool({"image_path": img_path})
                state["lesion_mask"] = res.get("mask_path")
                executed.append("lesion_segmentation")
                print("[推理节点] 全图病灶分割完成（无任何叶片约束）")

    # 记录已执行的任务
    state["executed_tasks"] = executed
    state["tool_calls"] = executed
    state["inference_done"] = True
    state["current_step"] = "inference_complete"
    return state


# ---------- 评估节点 ----------
async def evaluation_node(state: AgentState) -> AgentState:
    tasks = state.get("tasks", [])
    if "severity_assessment" not in tasks:
        state["analysis_result"] = None
        state["leaf_pixel_count"] = None
        state["lesion_pixel_count"] = None
        state["current_step"] = "evaluation_complete"
        return state

    leaf_path = state.get("leaf_mask")
    lesion_path = state.get("lesion_mask")
    if leaf_path and lesion_path and os.path.exists(leaf_path) and os.path.exists(lesion_path):
        try:
            leaf = np.array(Image.open(leaf_path).convert('L')) > 127
            lesion = np.array(Image.open(lesion_path).convert('L')) > 0
            leaf_area = np.sum(leaf)
            lesion_on_leaf = np.sum(lesion & leaf)
            severity = (lesion_on_leaf / leaf_area * 100) if leaf_area > 0 else 0.0
            state["leaf_pixel_count"] = int(leaf_area)
            state["lesion_pixel_count"] = int(lesion_on_leaf)
            state["analysis_result"] = f"{severity:.1f}%"
        except Exception as e:
            print(f"评估失败: {e}")
            state["analysis_result"] = None
    else:
        state["analysis_result"] = None
    state["current_step"] = "evaluation_complete"
    return state

# ---------- 可视化节点 ----------
# ---------- 可视化节点（基于任务与掩膜，生成叠加图）----------
async def visualization_node(state: AgentState) -> AgentState:
    tasks = state.get("tasks", [])
    leaf_mask_path = state.get("leaf_mask")
    lesion_mask_path = state.get("lesion_mask")
    original_path = state.get("image_path")

    print(f"[可视化] 原始图片: {original_path}, 存在: {Path(original_path).exists() if original_path else False}")
    print(f"[可视化] 叶片掩膜: {leaf_mask_path}, 存在: {Path(leaf_mask_path).exists() if leaf_mask_path else False}")
    print(f"[可视化] 病斑掩膜: {lesion_mask_path}, 存在: {Path(lesion_mask_path).exists() if lesion_mask_path else False}")

    if not original_path or not Path(original_path).exists():
        state["visualization_url"] = None
        print("[可视化] 原图不存在")
        state["current_step"] = "visualization_complete"
        return state

    # 根据任务列表决定是否需要生成对应的可视化
    need_leaf = "leaf_segmentation" in tasks and leaf_mask_path and Path(leaf_mask_path).exists()
    need_lesion = "lesion_segmentation" in tasks and lesion_mask_path and Path(lesion_mask_path).exists()

    if not need_leaf and not need_lesion:
        state["visualization_url"] = None
        print("[可视化] 无有效任务或掩膜，跳过可视化")
        state["current_step"] = "visualization_complete"
        return state

    original = np.array(Image.open(original_path).convert('RGB'))
    alpha = 0.5
    green = np.array([0, 255, 0])
    red = np.array([255, 0, 0])

    # 仅叶片
    if need_leaf and not need_lesion:
        mask = np.array(Image.open(leaf_mask_path).convert('L')) > 127
        overlay = original.copy()
        overlay[mask] = (overlay[mask] * (1 - alpha) + green * alpha).astype(np.uint8)
        filename = f"leaf_{uuid.uuid4().hex}.png"
    # 仅病斑
    elif need_lesion and not need_leaf:
        mask = np.array(Image.open(lesion_mask_path).convert('L')) > 0
        overlay = original.copy()
        overlay[mask] = (overlay[mask] * (1 - alpha) + red * alpha).astype(np.uint8)
        filename = f"lesion_{uuid.uuid4().hex}.png"
    # 两者都需要：生成叠加图（叶片绿 + 病斑红）
    else:
        leaf_mask = np.array(Image.open(leaf_mask_path).convert('L')) > 127
        lesion_mask = np.array(Image.open(lesion_mask_path).convert('L')) > 0
        overlay = original.copy()
        overlay[leaf_mask] = (overlay[leaf_mask] * (1 - alpha) + green * alpha).astype(np.uint8)
        overlay[lesion_mask] = (overlay[lesion_mask] * (1 - alpha) + red * alpha).astype(np.uint8)
        filename = f"combined_{uuid.uuid4().hex}.png"

    save_dir = Path("temp_vis")
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / filename
    Image.fromarray(overlay).save(save_path)
    state["visualization_url"] = f"/vis/{filename}"
    print(f"[可视化] 可视化生成成功: {state['visualization_url']}")

    state["current_step"] = "visualization_complete"
    return state
# ---------- 报告节点 ----------
# ---------- 报告节点（修正版）----------
async def reporting_node(state: AgentState) -> AgentState:
    """
    报告节点：基于 LLM 生成回复，不再硬编码关键词匹配。
    将当前状态（任务、分割结果、分析数据、可视化URL）组织成结构化上下文，
    交给 LLM 生成自然语言回复。
    """
    query = state["user_query"]
    tasks = state.get("tasks", [])
    if not tasks:
        # 直接返回固定拒绝消息，不调用 LLM
        answer = "抱歉，我只能处理植物叶片病害相关的问题（如叶片分割、病斑检测、病害严重度评估等）。请上传植物叶片图片并描述您的具体需求。"
        state["messages"].append({"role": "assistant", "content": answer})
        state["current_step"] = "reporting_complete"
        return state

    leaf_mask_ok = state.get("leaf_mask") and Path(state["leaf_mask"]).exists()
    lesion_mask_ok = state.get("lesion_mask") and Path(state["lesion_mask"]).exists()
    analysis = state.get("analysis_result")
    leaf_pixels = state.get("leaf_pixel_count")
    lesion_pixels = state.get("lesion_pixel_count")
    vis_url = state.get("visualization_url")
    error = state.get("error")

    # 构建任务执行状态摘要
    task_status = []
    if "leaf_segmentation" in tasks:
        task_status.append(f"- 叶片分割: {'成功' if leaf_mask_ok else '失败'}")
    if "lesion_segmentation" in tasks:
        task_status.append(f"- 病斑分割: {'成功' if lesion_mask_ok else '失败'}")
    if "severity_assessment" in tasks:
        if analysis:
            task_status.append(f"- 病害严重度评估: 成功，病斑占比 {analysis}")
            if leaf_pixels and lesion_pixels:
                task_status.append(f"  (叶片像素数: {leaf_pixels}, 病斑像素数: {lesion_pixels})")
        else:
            task_status.append("- 病害严重度评估: 失败")

    # 上下文信息
    context = f"""## 用户问题
{query}

## 执行的任务
{chr(10).join(task_status) if task_status else '无任务（可能问题超出范围或安全拦截）'}

## 附加信息
- 可视化图片是否生成: {'是' if vis_url else '否'}
- 系统错误信息: {error if error else '无'}
- 对话历史摘要: {state.get('conversation_history', [])[-3:]}  # 最近3条

## 回复要求
请根据以上信息，用自然、友好、专业的语言回答用户。
- 如果任务成功，明确告知结果，并在有可视化图片时提示用户查看。
- 如果任务部分失败，解释可能原因并给出建议（如检查图片质量、重新上传）。
- 如果用户问题与病害无关，说明你的能力范围并友好回应，不要回答与本系统无关的内容。
- 回复要简洁、有帮助，不要重复技术细节（除非用户询问）。
- 不要编造数据，只使用上面提供的信息。
"""

    try:
        resp = await llm.ainvoke([HumanMessage(content=context)])
        answer = resp.content
    except Exception as e:
        print(f"LLM 生成回复失败: {e}")
        # 降级回复
        if leaf_mask_ok or lesion_mask_ok:
            answer = "处理完成，请查看上方图片。如需详细数据，请重新描述您的需求。"
        else:
            answer = "处理过程中出现错误，请稍后重试或联系技术支持。"

    state["messages"].append({"role": "assistant", "content": answer})
    state["current_step"] = "reporting_complete"
    return state


import threading
import time

def clean_old_files(directory: Path, max_age_seconds: int = 3600):
    while True:
        now = time.time()
        for f in directory.glob("*"):
            if f.is_file() and (now - f.stat().st_mtime) > max_age_seconds:
                f.unlink()
        time.sleep(1800)  # 每半小时执行一次

# 在启动时开启后台清理线程（可选）
threading.Thread(target=clean_old_files, args=(Path("temp_vis"),), daemon=True).start()
threading.Thread(target=clean_old_files, args=(Path("temp_uploads"),), daemon=True).start()