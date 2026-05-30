# batch_experiment.py
import asyncio
import torch
import os
import csv
import json
from pathlib import Path
from dotenv import load_dotenv
from muti_agent.graph import agent_workflow
from muti_agent.state import AgentState
from muti_agent.evaluation_metrics import compute_metrics

load_dotenv()

def load_test_cases_from_json(file_path: str = "generated_test_cases_small.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        cases = json.load(f)
    formatted = []
    for case in cases:
        formatted.append({
            "id": case["id"],
            "query": case["query"],
            "expected_tasks": case.get("expected_tasks", []),
            "expected_chain": case.get("expected_chain", None),   # 新增
            "image_path": case.get("image_path"),
            "safety_expected": case.get("safety_expected", False),
            "category": case.get("category", "unknown")
        })
    return formatted

# CSV 输出字段增加链相关指标
CSV_FIELDS = [
    "test_id", "user_query", "success", "tool_match_rate",
    "redundant_rate", "missing_rate", "task_correctness",
    "no_redundant_rate", "safety_intercepted", "safety_expected",
    "chain_correctness", "chain_existence_score", "chain_order_correct",
    "error", "final_answer"
]

async def run_test_case(test_case: dict) -> dict:
    img_path = test_case.get("image_path")
    expected_tasks = test_case["expected_tasks"]
    expected_chain = test_case.get("expected_chain")

    # 如果期望任务需要图片但图片不存在，直接返回失败
    if expected_tasks and (not img_path or not Path(img_path).exists()):
        return {
            "test_id": test_case["id"],
            "user_query": test_case["query"],
            "success": False,
            "error": f"图片不存在: {img_path}",
            "tool_match_rate": 0,
            "redundant_rate": 0,
            "missing_rate": 1.0,
            "task_correctness": 0,
            "no_redundant_rate": 0,
            "safety_intercepted": 0,
            "chain_correctness": 0,
            "chain_existence_score": 0,
            "chain_order_correct": False,
            "final_answer": ""
        }

    initial_state: AgentState = {
        "messages": [{"role": "user", "content": test_case["query"]}],
        "user_query": test_case["query"],
        "image_path": img_path,
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
    }

    try:
        final_state = await agent_workflow.ainvoke(initial_state)
        metrics = compute_metrics(final_state, expected_tasks, expected_chain)
        metrics["test_id"] = test_case["id"]
        metrics["user_query"] = test_case["query"]
        metrics["success"] = (final_state.get("error") is None and final_state.get("safety_check_passed", True))
        metrics["final_answer"] = final_state["messages"][-1]["content"] if final_state["messages"] else ""
        metrics["error"] = final_state.get("error") or ""
        metrics["safety_expected"] = test_case.get("safety_expected", False)
        filtered = {k: metrics.get(k) for k in CSV_FIELDS}
        return filtered
    except Exception as e:
        return {
            "test_id": test_case["id"],
            "user_query": test_case["query"],
            "success": False,
            "error": str(e),
            "tool_match_rate": 0,
            "redundant_rate": 0,
            "missing_rate": 1.0 if expected_tasks else 0,
            "task_correctness": 0,
            "no_redundant_rate": 0,
            "safety_intercepted": 0,
            "safety_expected": test_case.get("safety_expected", False),
            "chain_correctness": 0,
            "chain_existence_score": 0,
            "chain_order_correct": False,
            "final_answer": "",
        }

async def main():
    TEST_CASES = load_test_cases_from_json()
    START_IDX = 0  # 从第START_IDX个开始
    TEST_CASES = TEST_CASES[START_IDX:]
    print(f"已跳过前 {START_IDX} 个测试用例，剩余 {len(TEST_CASES)} 个")
    results = []
    for case in TEST_CASES:
        print(f"Running test case {case['id']}: {case['query']}")
        result = await run_test_case(case)
        results.append(result)
        print(f"  -> match_rate={result.get('tool_match_rate')}, correctness={result.get('task_correctness')}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_file = "experiment_results_small.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n实验结果已保存到 {output_file}")

    # 统计
    valid_results = [r for r in results if "图片不存在" not in r.get("error", "")]
    if valid_results:
        total = len(valid_results)
        avg_match = sum(r["tool_match_rate"] for r in valid_results) / total
        avg_correctness = sum(r["task_correctness"] for r in valid_results) / total
        avg_chain = sum(r["chain_correctness"] for r in valid_results) / total
        print(f"总体平均工具匹配率: {avg_match:.4f}")
        print(f"总体平均任务正确性: {avg_correctness:.4f}")
        print(f"总体平均工具链正确性: {avg_chain:.4f}")

        safety_cases = [r for r in valid_results if r.get("safety_expected", False)]
        if safety_cases:
            safety_success = sum(1 for r in safety_cases if r["safety_intercepted"] == 1)
            print(f"安全拦截率: {safety_success/len(safety_cases):.4f} ({safety_success}/{len(safety_cases)})")
    else:
        print("没有有效的测试用例")

if __name__ == "__main__":
    asyncio.run(main())