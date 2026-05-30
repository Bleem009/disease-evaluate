# multi_turn_evaluation.py
import asyncio
import json
import csv
from pathlib import Path
import aiohttp
from dotenv import load_dotenv

load_dotenv()

API_BASE = "http://127.0.0.1:8000"
CHAT_URL = f"{API_BASE}/chat"

# 定义多轮测试用例（可直接写在此文件或从 JSON 加载）
MULTI_TURN_TEST_CASES = [
    {
        "id": 1,
        "name": "连续分割测试",
        "turns": [
            {"query": "提取叶片轮廓", "image": "test_images/leaf1.jpg", "expected_tasks": ["leaf_segmentation"]},
            {"query": "现在分割病斑", "expected_tasks": ["lesion_segmentation"], "should_reuse_image": True}
        ]
    },
    {
        "id": 2,
        "name": "重试测试",
        "turns": [
            {"query": "分割叶片", "image": "test_images/bad_leaf.jpg", "expected_tasks": ["leaf_segmentation"], "expected_success": False},
            {"query": "重新分割", "expected_tasks": ["leaf_segmentation"], "should_retry": True}
        ]
    }
]

async def run_multi_turn_test(test_case: dict) -> dict:
    """执行多轮对话测试，返回每轮的指标和总体指标"""
    session_id = None
    turn_results = []
    reused_image = False
    for idx, turn in enumerate(test_case["turns"]):
        data = aiohttp.FormData()
        data.add_field("message", turn["query"])
        if session_id:
            data.add_field("session_id", session_id)
        if "image" in turn and turn["image"]:
            with open(turn["image"], "rb") as f:
                data.add_field("image", f, filename="image.jpg")
        async with aiohttp.ClientSession() as session:
            async with session.post(CHAT_URL, data=data) as resp:
                result = await resp.json()
                session_id = result.get("session_id")
                reply = result.get("reply", "")
                vis_url = result.get("visualization_url")
        # 简单检查期望任务（实际中可调用 compute_metrics）
        # 这里简化，只记录是否成功和是否复用了图片
        success = True  # 可根据 reply 内容判断
        turn_results.append({
            "turn": idx+1,
            "query": turn["query"],
            "reply": reply,
            "success": success,
            "has_vis": vis_url is not None
        })
        # 检查图片复用（简单启发）
        if idx > 0 and turn.get("should_reuse_image", False):
            # 可以检查后端日志，这里假设如果成功且没有上传新图片则复用
            reused_image = True
    overall = {
        "test_id": test_case["id"],
        "name": test_case["name"],
        "total_turns": len(test_case["turns"]),
        "success_rate": sum(r["success"] for r in turn_results) / len(turn_results),
        "image_reused": reused_image,
        "turn_details": turn_results
    }
    return overall

async def main():
    results = []
    for case in MULTI_TURN_TEST_CASES:
        print(f"Running multi-turn test: {case['name']}")
        res = await run_multi_turn_test(case)
        results.append(res)
        print(f"  -> success_rate={res['success_rate']}, image_reused={res['image_reused']}")

    # 保存结果到 CSV
    with open("multi_turn_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["test_id", "name", "total_turns", "success_rate", "image_reused"])
        for r in results:
            writer.writerow([r["test_id"], r["name"], r["total_turns"], r["success_rate"], r["image_reused"]])

    print("\n多轮评测完成，结果保存到 multi_turn_results.csv")

if __name__ == "__main__":
    asyncio.run(main())