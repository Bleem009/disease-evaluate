from typing import TypedDict, List, Optional, Dict, Any

class AgentState(TypedDict):
    messages: List[Dict[str, Any]]
    user_query: str
    image_path: Optional[str]
    leaf_mask: Optional[str]
    lesion_mask: Optional[str]
    analysis_result: Optional[str]
    current_step: str
    error: Optional[str]
    tasks: Optional[List[str]]
    conversation_history: Optional[List[Dict[str, Any]]]
    leaf_pixel_count: Optional[int]
    lesion_pixel_count: Optional[int]
    visualization_url: Optional[str]
    inference_done: Optional[bool]
    # 安全与评估字段
    safety_check_passed: Optional[bool]      # 安全校验是否通过
    safety_violation_reason: Optional[str]   # 违规原因
    task_plan_valid: Optional[bool]          # 任务计划是否有效
    feedback_message: Optional[str]          # 反馈信息
    retry_count: Optional[int]               # 重试次数
    # 评估指标记录
    tool_calls: Optional[List[str]]           # 实际调用的工具列表
    tool_match_rate: Optional[float]          # 工具调用匹配率
    redundant_ops: Optional[List[str]]        # 冗余操作
    missing_tasks: Optional[List[str]]        # 遗漏任务
    task_source: Optional[str]                # "LLM" 或 "关键词匹配（兜底）"
    sam_leaf_mask: Optional[str]  # SAM 生成的叶片掩膜路径
    sam_confidence: Optional[float]  # SAM 掩膜置信度
    ensemble_decision: Optional[str]  # 集成决策理由（如 "Consensus", "Our model", "SAM"）