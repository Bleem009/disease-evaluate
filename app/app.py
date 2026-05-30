import sys
from pathlib import Path

# 添加项目根目录到 sys.path（使得 agents, configs, src 可导入）
sys.path.insert(0, str(Path(__file__).parent))

import io
import tempfile
import uuid
import json
import asyncio
from typing import Optional, Dict, List

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import segmentation_models_pytorch as smp
from skimage.transform import resize

from configs.stage1_leaf_config import Stage1Config
from configs.stage2_lesion_config import Stage2Config
from src.data.transforms import get_validation_transforms

# 导入 Agent 工作流
from muti_agent.graph import agent_workflow
from muti_agent.state import AgentState

from fastapi.staticfiles import StaticFiles

# 创建必要的目录
UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
VIS_DIR = Path("temp_vis")
VIS_DIR.mkdir(exist_ok=True)

# -------------------- 初始化 FastAPI --------------------
app = FastAPI(title="Plant Disease Diagnosis API")

# 静态文件服务（用于可视化图片）
app.mount("/vis", StaticFiles(directory=str(VIS_DIR)), name="vis")

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- 加载分割模型 --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Stage1 叶片分割模型
config_stage1 = Stage1Config()
model_stage1 = smp.DeepLabV3Plus(
    encoder_name=config_stage1.encoder_name,
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None
)
ckpt1 = torch.load(r"C:\Users\86159\PycharmProjects\disease_evaluation\checkpoints\stage1\best_model.pth",
                   map_location=device)
model_stage1.load_state_dict(ckpt1['model_state_dict'])
model_stage1.to(device)
model_stage1.eval()
print("Stage1 model loaded.")

# Stage2 病灶分割模型
config_stage2 = Stage2Config()
num_classes = getattr(config_stage2, 'num_classes', 5)
model_stage2 = smp.DeepLabV3Plus(
    encoder_name=config_stage2.encoder_name,
    encoder_weights=None,
    in_channels=3,
    classes=num_classes,
    activation=None
)
ckpt2 = torch.load(
    r"C:\Users\86159\PycharmProjects\disease_evaluation\checkpoints\stage2\可能输出多类病斑\best_model.pth",
    map_location=device)
model_stage2.load_state_dict(ckpt2['model_state_dict'])
model_stage2.to(device)
model_stage2.eval()
print("Stage2 model loaded.")

# 预处理函数
transform = get_validation_transforms(config_stage1.img_size)


def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    orig_np = np.array(image)
    h, w = orig_np.shape[:2]
    transformed = transform(image=orig_np, mask=np.zeros((h, w), dtype=np.uint8))
    input_tensor = transformed['image'].unsqueeze(0).to(device)
    return input_tensor, orig_np, h, w


def postprocess_mask(mask_tensor, orig_h, orig_w):
    mask_resized = mask_tensor.cpu().numpy().astype(np.uint8)
    mask_original = resize(mask_resized, (orig_h, orig_w), preserve_range=True, order=0).astype(np.uint8)
    return mask_original


# -------------------- 分割 API --------------------
@app.post("/predict/stage1/")
async def predict_stage1(file: UploadFile = File(...)):
    input_tensor, orig_np, h, w = preprocess_image(await file.read())
    with torch.no_grad():
        logits = model_stage1(input_tensor)
        probs = torch.sigmoid(logits)
        pred_mask = (probs[0, 0] > 0.5).byte()
    leaf_mask = postprocess_mask(pred_mask, h, w)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        mask_img = Image.fromarray((leaf_mask * 255).astype(np.uint8))
        mask_img.save(tmp.name)
        mask_path = tmp.name
    return {"mask_path": mask_path, "mask_shape": leaf_mask.shape}


@app.post("/predict/stage2/")
async def predict_stage2(file: UploadFile = File(...)):
    input_tensor, orig_np, h, w = preprocess_image(await file.read())
    with torch.no_grad():
        logits = model_stage2(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_mask = torch.argmax(probs, dim=1).byte()
    lesion_mask = postprocess_mask(pred_mask[0], h, w)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        mask_img = Image.fromarray(lesion_mask.astype(np.uint8))
        mask_img.save(tmp.name)
        mask_path = tmp.name
    return {"mask_path": mask_path, "mask_shape": lesion_mask.shape}


# 存储完整会话状态，而不仅仅是消息历史
sessions: Dict[str, AgentState] = {}

# 存储每个会话的任务状态更新队列
task_status_queues: Dict[str, asyncio.Queue] = {}


def get_task_status_queue(session_id: str) -> asyncio.Queue:
    """获取或创建会话的任务状态队列"""
    if session_id not in task_status_queues:
        task_status_queues[session_id] = asyncio.Queue()
    return task_status_queues[session_id]


async def push_task_status(session_id: str, status: dict):
    """推送任务状态更新到队列"""
    if session_id in task_status_queues:
        await task_status_queues[session_id].put(status)


@app.get("/task-stream/{session_id}")
async def task_stream(session_id: str, request: Request):
    """SSE 端点：实时推送任务状态更新"""
    queue = get_task_status_queue(session_id)

    async def event_generator():
        try:
            while True:
                # 检查客户端是否断开
                if await request.is_disconnected():
                    break

                try:
                    # 等待队列中的新消息，设置超时以便定期检查连接
                    status = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {json.dumps(status, ensure_ascii=False)}\n\n"

                    # 如果是完成状态，结束流
                    if status.get("event") == "complete":
                        break
                except asyncio.TimeoutError:
                    # 发送心跳保持连接
                    yield f"data: {json.dumps({'event': 'heartbeat'})}\n\n"
        finally:
            # 清理队列
            if session_id in task_status_queues:
                del task_status_queues[session_id]

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.post("/chat")
async def chat(
        session_id: Optional[str] = Form(None),
        message: str = Form(...),
        image: UploadFile = File(None)
):
    if not session_id:
        session_id = str(uuid.uuid4())

    # 1. 获取或初始化会话状态
    if session_id in sessions:
        state = sessions[session_id].copy()  # 浅拷贝，避免污染原对象
        # 重置本次请求相关的字段
        state["user_query"] = message
        state["error"] = None
        state["current_step"] = "start"
        state["inference_done"] = False
        # 保留之前的 conversation_history，并追加新用户消息
        history = state.get("conversation_history", [])
        history.append({"role": "user", "content": message})
        state["conversation_history"] = history
        state["messages"] = history.copy()
        # 重置任务状态
        state["planned_tasks"] = []
        state["executed_tasks"] = []
        state["tasks"] = []
    else:
        # 全新会话
        history = [{"role": "user", "content": message}]
        state: AgentState = {
            "messages": history.copy(),
            "user_query": message,
            "image_path": None,
            "leaf_mask": None,
            "lesion_mask": None,
            "analysis_result": None,
            "current_step": "start",
            "error": None,
            "tasks": None,
            "conversation_history": [],
            "leaf_pixel_count": None,
            "lesion_pixel_count": None,
            "visualization_url": None,
            "inference_done": False,
            "safety_check_passed": True,
            "safety_violation_reason": None,
            "retry_count": 0,
            "tool_calls": [],
            "planned_tasks": [],
            "executed_tasks": [],
            "redundant_ops": [],
            "missing_tasks": [],
            "task_source": None,
        }

    # 2. 处理新上传的图片（覆盖旧图片）
    if image:
        # 删除旧图片文件（如果有）
        old_img = state.get("image_path")
        if old_img and Path(old_img).exists():
            Path(old_img).unlink()
        # 保存新图片到会话专属目录（按 session_id 隔离）
        session_dir = UPLOAD_DIR / session_id
        session_dir.mkdir(exist_ok=True)
        new_img_path = session_dir / f"original_{uuid.uuid4().hex}.jpg"
        content = await image.read()
        with open(new_img_path, "wb") as f:
            f.write(content)
        state["image_path"] = str(new_img_path)
        # 新图片来了，之前的分割结果作废
        state["leaf_mask"] = None
        state["lesion_mask"] = None
        state["analysis_result"] = None
        state["inference_done"] = False

    # 初始化任务状态队列
    get_task_status_queue(session_id)

    # 推送初始状态
    await push_task_status(session_id, {
        "event": "init",
        "planned_tasks": [],
        "executed_tasks": [],
        "current_task": None,
        "analysis_result": None,
        "error": None
    })

    # 3. 运行工作流（使用流式回调）
    try:
        # 使用 astream 来获取中间状态
        final_state = None
        async for event in agent_workflow.astream(state):
            # event 是一个字典，key 是节点名称，value 是该节点的输出状态
            for node_name, node_state in event.items():
                final_state = node_state

                # 根据节点推送不同的状态更新
                if node_name == "planning":
                    # 规划完成，推送规划的任务列表
                    planned = node_state.get("planned_tasks", [])
                    await push_task_status(session_id, {
                        "event": "planning_complete",
                        "planned_tasks": planned,
                        "executed_tasks": [],
                        "current_task": planned[0] if planned else None,
                        "analysis_result": None,
                        "error": node_state.get("error")
                    })

                elif node_name == "inference":
                    # 推理进行中/完成
                    executed = node_state.get("executed_tasks", [])
                    planned = node_state.get("planned_tasks", [])
                    # 找出当前正在执行的任务
                    pending = [t for t in planned if t not in executed and t != "severity_assessment"]
                    current = pending[0] if pending else (
                        "severity_assessment" if "severity_assessment" in planned and "severity_assessment" not in executed else None)

                    await push_task_status(session_id, {
                        "event": "inference_progress",
                        "planned_tasks": planned,
                        "executed_tasks": executed,
                        "current_task": current,
                        "analysis_result": None,
                        "error": node_state.get("error")
                    })

                elif node_name == "evaluation":
                    # 评估完成
                    executed = node_state.get("executed_tasks", [])
                    if "severity_assessment" in node_state.get("planned_tasks", []):
                        executed = list(set(executed + ["severity_assessment"]))

                    await push_task_status(session_id, {
                        "event": "evaluation_complete",
                        "planned_tasks": node_state.get("planned_tasks", []),
                        "executed_tasks": executed,
                        "current_task": None,
                        "analysis_result": node_state.get("analysis_result"),
                        "error": node_state.get("error")
                    })

                elif node_name == "visualization":
                    # 可视化完成
                    await push_task_status(session_id, {
                        "event": "visualization_complete",
                        "planned_tasks": node_state.get("planned_tasks", []),
                        "executed_tasks": node_state.get("executed_tasks", []),
                        "current_task": None,
                        "analysis_result": node_state.get("analysis_result"),
                        "error": node_state.get("error")
                    })

        if final_state is None:
            final_state = state

        assistant_message = final_state["messages"][-1]["content"] if final_state.get("messages") else "处理完成"
        vis_url = final_state.get("visualization_url")

    except Exception as e:
        import traceback
        traceback.print_exc()
        assistant_message = f"处理出错: {str(e)}"
        vis_url = None
        final_state = state  # 保持原有状态，但标记错误
        final_state["error"] = str(e)

        # 推送错误状态
        await push_task_status(session_id, {
            "event": "error",
            "planned_tasks": state.get("planned_tasks", []),
            "executed_tasks": state.get("executed_tasks", []),
            "current_task": None,
            "analysis_result": None,
            "error": str(e)
        })

    # 推送完成状态
    await push_task_status(session_id, {
        "event": "complete",
        "planned_tasks": final_state.get("planned_tasks", []),
        "executed_tasks": final_state.get("executed_tasks", []),
        "current_task": None,
        "analysis_result": final_state.get("analysis_result"),
        "error": final_state.get("error")
    })

    # 4. 更新会话存储（合并回写关键字段）
    sessions[session_id] = final_state

    task_status = {
        "planned_tasks": final_state.get("planned_tasks", []),
        "executed_tasks": final_state.get("executed_tasks", []),
        "analysis_result": final_state.get("analysis_result"),
        "error": final_state.get("error")
    }
    return {
        "session_id": session_id,
        "reply": assistant_message,
        "visualization_url": vis_url,
        "task_status": task_status
    }


# -------------------- 启动 --------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
