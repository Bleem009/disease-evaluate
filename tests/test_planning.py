import asyncio
from muti_agent.state import AgentState
from muti_agent.nodes import planning_node

async def test_planning(user_query: str):
    state: AgentState = {
        "user_query": user_query,
        "conversation_history": [],
        "messages": [],
        "image_path": None,
        "leaf_mask": None,
        "lesion_mask": None,
        "analysis_result": None,
        "current_step": "start",
        "error": None,
        "tasks": None,
        "leaf_pixel_count": None,
        "lesion_pixel_count": None,
        "visualization_url": None,
        "inference_done": False,
        "safety_check_passed": None,
        "safety_violation_reason": None,
        "task_plan_valid": None,
        "feedback_message": None,
        "retry_count": 0,
        "planned_tasks": None,
        "executed_tasks": None,
        "redundant_ops": None,
        "missing_tasks": None,
        "tool_calls": None,
        # 新增
        "task_source": None,
    }
    result = await planning_node(state)
    print(f"最终任务: {result['tasks']} (来源: {result['task_source']})")
    print("-" * 50)

async def main():
    test_queries = [
        "请帮我分割叶片",
        "病斑严重吗",
        "叶片上有黄色斑点",
        "这个病害度多少",
    ]
    for q in test_queries:
        await test_planning(q)

if __name__ == "__main__":
    asyncio.run(main())