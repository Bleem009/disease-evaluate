# scripts/inference/predict_single.py
# !/usr/bin/env python3
"""
单张图像推理
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
from configs.stage1_leaf_config import Stage1Config
from configs.stage2_lesion_config import Stage2Config
from src.inference.predictor import TwoStagePredictor
from src.inference.visualizer import ResultVisualizer


def main():
    parser = argparse.ArgumentParser(description='Predict disease severity')
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--output', type=str, default='outputs/prediction_result.png',
                        help='Output visualization path')
    parser.add_argument('--no-display', action='store_true', help='Do not display result')
    args = parser.parse_args()

    # 配置
    config1 = Stage1Config()
    config2 = Stage2Config()

    # 创建预测器
    print("Loading models...")
    predictor = TwoStagePredictor(
        #stage1_model_path=config1.output_dir / "best_model.pth",
        #stage2_model_path=config2.output_dir / "best_model.pth",
        stage1_model_path=r"C:\Users\86159\PycharmProjects\disease_evaluation\checkpoints\stage1\best_model.pth",
        stage2_model_path=r"C:\Users\86159\PycharmProjects\disease_evaluation\checkpoints\pretrained\2stage_optimized.ptl",
        device='cuda'
    )

    # 预测
    print(f"Processing: {args.image}")
    result = predictor.predict(args.image, return_intermediate=True)

    # 打印结果
    print("\n" + "=" * 50)
    print("Prediction Result")
    print("=" * 50)
    print(f"  Leaf pixels: {result['leaf_pixels']:,}")
    print(f"  Lesion pixels: {result['lesion_pixels']:,}")
    print(f"  Disease severity: {result['severity']:.2f}%")
    print("=" * 50)

    # 可视化
    from PIL import Image
    import numpy as np

    image = np.array(Image.open(args.image).convert('RGB'))

    visualizer = ResultVisualizer()
    visualizer.plot_result(
        image=image,
        leaf_mask=result['leaf_mask'],
        lesion_mask=result['lesion_mask'],
        severity=result['severity'],
        save_path=args.output,
        show=not args.no_display
    )

    print(f"\nVisualization saved to: {args.output}")


if __name__ == "__main__":
    main()