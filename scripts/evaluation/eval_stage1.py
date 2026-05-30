# scripts/evaluation/eval_stage1.py
# !/usr/bin/env python3
"""
评估Stage 1模型（叶片分割）
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from configs.stage1_leaf_config import Stage1Config
from src.data.datasets import Stage1Dataset
from src.data.transforms import get_validation_transforms
from src.training.metrics import iou_score, pixel_accuracy, dice_score
from src.inference.predictor import OneStagePredictor


def evaluate_model(model_path, test_img_dir, test_label_dir, device='cuda'):
    """评估模型"""
    config = Stage1Config()

    # 创建数据集
    dataset = Stage1Dataset(
        image_dir=test_img_dir,
        label_dir=test_label_dir,
        transform=get_validation_transforms(config.img_size)
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )

    # 加载模型
    predictor = OneStagePredictor(
        model_path=model_path,
        encoder_name='mobilenet_v2',
        device=device,
        img_size=config.img_size
    )

    # 评估
    metrics = {'iou': [], 'pixel_acc': [], 'dice': []}

    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch['image']
        masks = batch['mask'].numpy()

        for i in range(len(images)):
            # 反归一化图像用于预测
            img_tensor = images[i]
            img_np = img_tensor.permute(1, 2, 0).numpy()
            img_np = (img_np * np.array([0.229, 0.224, 0.225]) +
                      np.array([0.485, 0.456, 0.406]))
            img_np = (img_np * 255).astype(np.uint8)

            # 预测
            pred_mask, _ = predictor.predict(img_np)

            # 调整mask尺寸
            gt_mask = masks[i]
            if pred_mask.shape != gt_mask.shape:
                import cv2
                pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)

            # 计算指标
            pred_tensor = torch.from_numpy(pred_mask).unsqueeze(0).unsqueeze(0).float()
            gt_tensor = torch.from_numpy(gt_mask).unsqueeze(0)

            metrics['iou'].append(iou_score(pred_tensor, gt_tensor))
            metrics['pixel_acc'].append(pixel_accuracy(pred_tensor, gt_tensor))
            metrics['dice'].append(dice_score(pred_tensor, gt_tensor))

    # 汇总
    results = {k: float(np.mean(v)) for k, v in metrics.items()}
    results['std_iou'] = float(np.std(metrics['iou']))

    print("\n" + "=" * 50)
    print("Stage 1 Evaluation Results")
    print("=" * 50)
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    # 保存
    out_dir = Path("outputs/evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "stage1_eval.json", 'w') as f:
        json.dump(results, f, indent=2)

    return results


def main():
    config = Stage1Config()

    results = evaluate_model(
        model_path=config.output_dir / "best_model.pth",
        test_img_dir=config.val_img_dir,
        test_label_dir=config.val_label_dir
    )


if __name__ == "__main__":
    main()