import os
import re
import json
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from muti_agent.state import AgentState

load_dotenv()

safety_llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
    openai_api_key=os.getenv("DOUBAO_API_KEY"),
    base_url=os.getenv("DOUBAO_BASE_URL"),
    temperature=0.0
)

# 仅作为快速兜底，不依赖 tasks
FAST_BLACKLIST = [
    r"删除.*数据", r"drop\s+table", r"rm\s+-rf", r"格式化",
    r"获取.*密码", r"窃取", r"恶意", r"攻击", r"注入",
    r"入侵", r"破坏", r"病毒", r"删除所有", r"篡改", r"爆破"
]

SAFETY_SYSTEM_PROMPT = """你是一个安全审核专家。判断用户问题是否包含恶意意图或超出系统功能范围。
系统功能仅限于：植物叶片病害严重度测定（叶片分割、病斑分割、病害程度评估）。
输出 JSON：{"is_safe": boolean, "reason": "简短理由"}"""

async def safety_agent_node(state: AgentState) -> AgentState:
    query = state.get("user_query", "")
    print(f"[安全节点] 收到用户问题: {query}")

    # 1. 快速黑名单（不依赖 tasks）
    for pattern in FAST_BLACKLIST:
        if re.search(pattern, query, re.IGNORECASE):
            print(f"[安全节点] 快速拦截: {pattern}")
            state["safety_check_passed"] = False
            state["safety_violation_reason"] = f"快速拦截: {pattern}"
            state["error"] = "请求被安全策略拦截"
            return state

    # 2. 调用 LLM 判断（不依赖 tasks，避免空任务误判）
    try:
        response = await safety_llm.ainvoke([
            SystemMessage(content=SAFETY_SYSTEM_PROMPT),
            HumanMessage(content=f"用户问题：{query}")
        ])
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]
        result = json.loads(content)
        is_safe = result.get("is_safe", True)
        reason = result.get("reason", "")
    except Exception as e:
        print(f"[安全节点] LLM 解析失败: {e}")
        is_safe = True
        reason = "解析失败，默认放行"

    if not is_safe:
        print(f"[安全节点] LLM 拦截: {reason}")
        state["safety_check_passed"] = False
        state["safety_violation_reason"] = reason
        state["error"] = f"安全拦截: {reason}"
        return state

    print("[安全节点] 通过")
    state["safety_check_passed"] = True
    state["safety_violation_reason"] = None
    # 注意：不要修改 current_step，保持原有值，由 graph 的条件边继续流转
    return state