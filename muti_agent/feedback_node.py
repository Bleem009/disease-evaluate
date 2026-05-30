from muti_agent.state import AgentState


async def feedback_node(state: AgentState) -> AgentState:
    """检查推理和评估结果，决定是否需要重试或重新规划"""
    retry_count = state.get("retry_count", 0)
    max_retries = 3

    # 如果已有错误且未超过重试次数，重新规划
    if state.get("error") and retry_count < max_retries:
        state["retry_count"] = retry_count + 1
        state["error"] = None  # 清除错误
        state["current_step"] = "replanning"
        # 清空任务列表，强制重新规划
        state["tasks"] = None
        state["inference_done"] = False
        # 添加反馈信息到历史
        feedback_msg = f"上次执行失败: {state.get('safety_violation_reason', '未知错误')}，正在重试 (第{retry_count + 1}次)"
        state["feedback_message"] = feedback_msg
        # 返回，让图重新从规划节点开始
        state["current_step"] = "feedback_complete"
        return state

    # 如果成功或超过重试次数，继续到报告节点
    state["current_step"] = "feedback_complete"
    return state