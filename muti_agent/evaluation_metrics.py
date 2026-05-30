# muti_agent/evaluation_metrics.py
from typing import List, Dict, Any
from muti_agent.state import AgentState

def compute_metrics(final_state: AgentState, expected_tasks: List[str], expected_chain: List[str] = None) -> Dict[str, Any]:
    """
    计算任务处理正确性指标
    新增 expected_chain 参数用于评估工具链顺序
    """
    # 安全获取列表字段，若为 None 则转为空列表
    planned = final_state.get("planned_tasks") or []
    executed = final_state.get("executed_tasks") or []
    redundant = final_state.get("redundant_ops") or []
    missing = final_state.get("missing_tasks") or []
    safety_passed = final_state.get("safety_check_passed", True)

    # 如果 executed 为空但实际有推理结果，则根据掩膜推断
    if not executed:
        inferred = []
        if final_state.get("leaf_mask") is not None:
            inferred.append("leaf_segmentation")
        if final_state.get("lesion_mask") is not None:
            inferred.append("lesion_segmentation")
        if final_state.get("analysis_result") is not None and final_state["analysis_result"] != "":
            inferred.append("severity_assessment")
        if inferred:
            executed = inferred

    # 匹配率
    if expected_tasks:
        correct = set(expected_tasks) & set(executed)
        match_rate = len(correct) / len(expected_tasks)
    else:
        match_rate = 1.0 if len(executed) == 0 else 0.0

    # 冗余率（基于计划任务）
    if planned:
        redundant_ops = [t for t in planned if t not in expected_tasks]
        redundant_rate = len(redundant_ops) / len(planned)
    else:
        redundant_rate = 0.0

    # 遗漏率
    if expected_tasks:
        missing_tasks = [t for t in expected_tasks if t not in executed]
        missing_rate = len(missing_tasks) / len(expected_tasks)
    else:
        missing_rate = 0.0

    correctness = match_rate * (1 - redundant_rate) * (1 - missing_rate)
    no_redundant_rate = 1 - redundant_rate
    safety_intercepted = 0 if safety_passed else 1

    # ========== 新增：工具链顺序评估 ==========
    chain_metrics = compute_chain_metrics(executed, expected_chain)

    return {
        "tool_match_rate": round(match_rate, 4),
        "redundant_rate": round(redundant_rate, 4),
        "missing_rate": round(missing_rate, 4),
        "task_correctness": round(correctness, 4),
        "no_redundant_rate": round(no_redundant_rate, 4),
        "safety_intercepted": safety_intercepted,
        "chain_correctness": chain_metrics["chain_correctness"],
        "chain_existence_score": chain_metrics["existence_score"],
        "chain_order_correct": chain_metrics["order_correct"],
    }

def compute_chain_metrics(executed_tasks: List[str], expected_chain: List[str]) -> Dict:
    """
    评估工具链顺序正确性
    expected_chain: 期望的工具调用顺序列表，如 ["leaf_segmentation", "lesion_segmentation"]
    """
    if not expected_chain:
        return {"chain_correctness": 1.0, "existence_score": 1.0, "order_correct": True}

    # 检查存在性
    missing = [t for t in expected_chain if t not in executed_tasks]
    existence_score = (len(expected_chain) - len(missing)) / len(expected_chain)

    # 检查顺序：提取 executed_tasks 中与 expected_chain 共有的元素，比较顺序
    common = [t for t in executed_tasks if t in expected_chain]
    order_correct = (common == expected_chain)

    # 综合得分（存在性 + 顺序，各占50%）
    chain_correctness = (existence_score + (1.0 if order_correct else 0.0)) / 2
    return {
        "chain_correctness": round(chain_correctness, 4),
        "existence_score": round(existence_score, 4),
        "order_correct": order_correct
    }