# scripts/evaluation/eval_end2end.py
# !/usr/bin/env python3
"""
端到端评估（两阶段一起）
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from PIL import Image
from tqdm import tqdm
import json

from configs.stage1_leaf_config import Stage1Config
from configs.stage2_lesion_config import Stage2Config
from src.inference.predictor import TwoStagePredictor
from src.data.labelme_parser import LabelMeParser


def calculate_severity_error(pred_severity, gt_severity):
    """计算严重度误差"""
    return abs(pred_severity - gt_severity)


def main():
    # 配置
    config1 = Stage1Config()
    config2 = Stage2Config()

    # 创建预测器
    predictor = TwoStagePredictor(
        stage1_model_path=config1.output_dir / "best_model.pth",
        stage2_model_path=config2.output_dir / "best_model.pth",
        device='cuda'
    )

    # 测试数据
    test_img_dir = Path("data/raw/field_images/test")
    test_leaf_lbl_dir = Path("data/raw/leaf_annotations/test")
    test_lesion_lbl_dir = Path("data/processed/stage2/test/labels")

    # 解析器
    leaf_parser = LabelMeParser(label_mapping={'leaf': 1})
    lesion_parser = LabelMeParser(label_mapping={'lesion': 1, 'disease': 1})

    results = []

    for img_path in tqdm(list(test_img_dir.glob("*.jpg"))):
        # 预测
        pred = predictor.predict(img_path)

        # 加载真值
        h, w = pred['leaf_mask'].shape

        # 叶片真值
        leaf_lbl_path = test_leaf_lbl_dir / f"{img_path.stem}.json"
        gt_leaf = leaf_parser.parse(leaf_lbl_path, (h, w))

        # 病灶真值（需要裁剪后对应）
        # 这里简化处理，实际应该根据bbox裁剪
        lesion_lbl_path = test_lesion_lbl_dir / f"{img_path.stem}.json"
        gt_lesion = lesion_parser.parse(lesion_lbl_path, (h, w))

        # 计算叶片IoU
        intersection = np.sum((pred['leaf_mask'] > 0) & (gt_leaf > 0))
        union = np.sum((pred['leaf_mask'] > 0) | (gt_leaf > 0))
        leaf_iou = intersection / (union + 1e-6)

        # 计算病灶IoU
        intersection = np.sum((pred['lesion_mask'] > 0) & (gt_lesion > 0))
        union = np.sum((pred['lesion_mask'] > 0) | (gt_lesion > 0))
        lesion_iou = intersection / (union + 1e-6)

        # 严重度真值
        gt_severity = np.sum(gt_lesion) / np.sum(gt_leaf) * 100 if np.sum(gt_leaf) > 0 else 0

        results.append({
            'filename': img_path.name,
            'leaf_iou': float(leaf_iou),
            'lesion_iou': float(lesion_iou),
            'pred_severity': float(pred['severity']),
            'gt_severity': float(gt_severity),
            'severity_error': float(abs(pred['severity'] - gt_severity))
        })

    # 汇总
    avg_leaf_iou = np.mean([r['leaf_iou'] for r in results])
    avg_lesion_iou = np.mean([r['lesion_iou'] for r in results])
    mae_severity = np.mean([r['severity_error'] for r in results])

    summary = {
        'num_samples': len(results),
        'avg_leaf_iou': float(avg_leaf_iou),
        'avg_lesion_iou': float(avg_lesion_iou),
        'mae_severity': float(mae_severity),
        'details': results
    }

    print("\n" + "=" * 50)
    print("End-to-End Evaluation Results")
    print("=" * 50)
    print(f"  Samples: {summary['num_samples']}")
    print(f"  Avg Leaf IoU: {avg_leaf_iou:.4f}")
    print(f"  Avg Lesion IoU: {avg_lesion_iou:.4f}")
    print(f"  Severity MAE: {mae_severity:.2f}%")

    # 保存
    out_dir = Path("outputs/evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "end2end_eval.json", 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()