#!/usr/bin/env python3
"""
交互式 SAM 分割可视化工具
支持：点击交互、多掩膜选择、实时可视化
效果类似 SAM 官网的交互式分割展示
"""

import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import cv2
import argparse
from segment_anything import sam_model_registry, SamPredictor

# ============ 可视化配置（可调整颜色） ============
VIS_CONFIG = {
    'mask_alpha': 0.6,  # 掩膜透明度
    'mask_color': [128, 0, 128],  # 掩膜颜色 (BGR格式) - 紫色
    'boundary_color': [255, 0, 255],  # 边界高亮色 - 亮紫/粉色
    'boundary_thickness': 2,
    'point_color_positive': [0, 255, 0],  # 正样本点（绿）
    'point_color_negative': [0, 0, 255],  # 负样本点（红）
    'point_radius': 8,
    'contour_thickness': 2,
}


class InteractiveSAMVisualizer:
    def __init__(self, checkpoint_path, model_type="vit_h", device="cuda"):
        """初始化 SAM 模型"""
        print(f"Loading SAM model: {model_type} on {device}")
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)
        self.device = device

        # 状态变量
        self.image = None  # 原始图像
        self.image_bgr = None  # BGR格式用于显示
        self.display_image = None  # 当前显示图像
        self.points = []  # 点击的点 [(x, y, label), ...]
        self.point_labels = []  # 1=正样本(前景), 0=负样本(背景)
        self.current_mask = None  # 当前最佳掩膜
        self.all_masks = []  # 所有生成的掩膜

    def load_image(self, image_path):
        """加载图像并初始化预测器"""
        self.image = np.array(Image.open(image_path).convert('RGB'))
        self.image_bgr = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        self.display_image = self.image_bgr.copy()
        self.predictor.set_image(self.image)
        self.points = []
        self.point_labels = []
        self.current_mask = None
        print(f"Image loaded: {self.image.shape}")
        return self.image

    def predict_mask(self):
        """基于当前点击点预测掩膜"""
        if len(self.points) == 0:
            return None

        input_points = np.array(self.points)
        input_labels = np.array(self.point_labels)

        # SAM 预测：支持多掩膜输出（3个不同粒度）
        masks, scores, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,  # 输出3个不同粒度的掩膜
        )

        # 选择得分最高的掩膜（通常是最精确的）
        best_idx = np.argmax(scores)
        self.current_mask = masks[best_idx]
        self.all_masks = masks
        self.scores = scores

        print(f"Generated {len(masks)} masks, best score: {scores[best_idx]:.3f}")
        return self.current_mask

    def create_visualization(self, show_all_masks=False):
        """
        创建可视化图像
        效果类似官网：半透明掩膜 + 发光边界 + 点击点标记
        """
        if self.current_mask is None:
            return self.image_bgr.copy()

        # 基础图像
        vis = self.image_bgr.copy()
        h, w = vis.shape[:2]

        if show_all_masks and len(self.all_masks) > 0:
            # 显示所有3个掩膜的对比（调试用）
            return self._create_multi_mask_view()

        # 1. 创建半透明掩膜覆盖层
        mask_overlay = np.zeros_like(vis)
        mask_color = VIS_CONFIG['mask_color']
        mask_overlay[self.current_mask] = mask_color

        # 2. 提取掩膜边界（实现官网的"发光边缘"效果）
        mask_uint8 = self.current_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 3. 绘制边界（双层实现发光效果）
        # 外层光晕
        glow = np.zeros_like(vis)
        cv2.drawContours(glow, contours, -1, VIS_CONFIG['boundary_color'],
                         VIS_CONFIG['boundary_thickness'] + 2)
        # 内层实线
        cv2.drawContours(glow, contours, -1, [255, 255, 255],
                         VIS_CONFIG['boundary_thickness'])

        # 4. 混合所有层
        # 先混合掩膜
        vis = cv2.addWeighted(vis, 1.0, mask_overlay, VIS_CONFIG['mask_alpha'], 0)
        # 再混合边界光晕
        vis = cv2.addWeighted(vis, 1.0, glow, 0.8, 0)

        # 5. 绘制点击点
        for (x, y), label in zip(self.points, self.point_labels):
            color = VIS_CONFIG['point_color_positive'] if label == 1 else VIS_CONFIG['point_color_negative']
            # 外圈白色边框
            cv2.circle(vis, (x, y), VIS_CONFIG['point_radius'] + 2, [255, 255, 255], -1)
            # 内圈颜色
            cv2.circle(vis, (x, y), VIS_CONFIG['point_radius'], color, -1)
            # 中心点
            cv2.circle(vis, (x, y), 3, [255, 255, 255], -1)

        return vis

    def _create_multi_mask_view(self):
        """创建多掩膜对比视图（调试用）"""
        h, w = self.image_bgr.shape[:2]
        # 水平拼接3个掩膜结果
        views = []
        colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]  # 红绿蓝

        for i, (mask, score) in enumerate(zip(self.all_masks, self.scores)):
            vis = self.image_bgr.copy()
            overlay = np.zeros_like(vis)
            overlay[mask] = colors[i]
            vis = cv2.addWeighted(vis, 1.0, overlay, 0.5, 0)

            # 添加文字标签
            label = f"Mask {i + 1} (score: {score:.3f})"
            cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, [255, 255, 255], 2)
            views.append(vis)

        return np.hstack(views)

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标点击回调"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 左键 = 正样本点（前景）
            self.points.append([x, y])
            self.point_labels.append(1)
            print(f"Added positive point at ({x}, {y})")
            self._update()

        elif event == cv2.EVENT_RBUTTONDOWN:
            # 右键 = 负样本点（背景/排除）
            self.points.append([x, y])
            self.point_labels.append(0)
            print(f"Added negative point at ({x}, y)")
            self._update()

        elif event == cv2.EVENT_MBUTTONDOWN:
            # 中键 = 撤销上一个点
            if len(self.points) > 0:
                self.points.pop()
                self.point_labels.pop()
                print("Undo last point")
                self._update()

    def _update(self):
        """更新预测和显示"""
        if len(self.points) > 0:
            self.predict_mask()
        else:
            self.current_mask = None
        self.display_image = self.create_visualization()

    def run(self, image_path):
        """运行交互式可视化"""
        self.load_image(image_path)

        # 创建窗口
        window_name = "SAM Interactive Segmentation (L:+, R:-, M:Undo, S:Save, Q:Quit)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        print("\n" + "=" * 50)
        print("交互式 SAM 分割工具")
        print("=" * 50)
        print("左键点击: 添加正样本点（分割目标区域）")
        print("右键点击: 添加负样本点（排除区域）")
        print("中键点击: 撤销上一个点")
        print("按 'S': 保存当前结果")
        print("按 'Q' 或 ESC: 退出")
        print("=" * 50 + "\n")

        self.display_image = self.image_bgr.copy()

        while True:
            cv2.imshow(window_name, self.display_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # Q 或 ESC
                break

            elif key == ord('s'):  # 保存结果
                self.save_result()

            elif key == ord('a'):  # 切换显示所有掩膜
                self.display_image = self.create_visualization(show_all_masks=True)

        cv2.destroyAllWindows()

    def save_result(self):
        """保存当前可视化结果"""
        if self.current_mask is None:
            print("No mask to save!")
            return

        # 保存可视化图像
        vis_path = "sam_result_visualization.png"
        cv2.imwrite(vis_path, self.display_image)

        # 保存二值掩膜
        mask_path = "sam_result_mask.png"
        cv2.imwrite(mask_path, self.current_mask.astype(np.uint8) * 255)

        # 保存叠加图（原图+掩膜）
        overlay = self.image_bgr.copy()
        overlay[self.current_mask] = VIS_CONFIG['mask_color']
        overlay_path = "sam_result_overlay.png"
        cv2.imwrite(overlay_path, overlay)

        print(f"Saved: {vis_path}, {mask_path}, {overlay_path}")


# ============ 批量处理模式（类似您原代码的评估流程） ============
class BatchMaskVisualizer:
    """批量处理并生成官网风格的可视化结果"""

    @staticmethod
    def visualize_prediction(image, mask, save_path=None, alpha=0.6):
        """
        生成类似 SAM 官网的可视化图像
        紫色半透明掩膜 + 粉色发光边界
        """
        if isinstance(image, str):
            image = np.array(Image.open(image).convert('RGB'))

        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        h, w = image_bgr.shape[:2]

        # 创建掩膜层
        overlay = np.zeros_like(image_bgr)
        overlay[mask] = VIS_CONFIG['mask_color']  # 紫色

        # 提取边界
        mask_uint8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 创建发光边界
        boundary = np.zeros_like(image_bgr)
        cv2.drawContours(boundary, contours, -1, VIS_CONFIG['boundary_color'], 3)
        cv2.drawContours(boundary, contours, -1, [255, 255, 255], 1)

        # 混合
        result = cv2.addWeighted(image_bgr, 1.0, overlay, alpha, 0)
        result = cv2.addWeighted(result, 1.0, boundary, 0.8, 0)

        if save_path:
            cv2.imwrite(save_path, result)
            print(f"Saved visualization to {save_path}")

        return result

    @staticmethod
    def create_comparison_grid(images, masks, names=None, cols=3):
        """创建对比网格（类似论文中的展示）"""
        rows = (len(images) + cols - 1) // cols
        cell_h, cell_w = 400, 400  # 每个单元格大小

        grid = np.ones((rows * cell_h, cols * cell_w, 3), dtype=np.uint8) * 255

        for idx, (img, mask) in enumerate(zip(images, masks)):
            row = idx // cols
            col = idx % cols

            # 生成可视化
            vis = BatchMaskVisualizer.visualize_prediction(img, mask)
            vis = cv2.resize(vis, (cell_w - 20, cell_h - 40))

            # 放置到网格
            y_start = row * cell_h + 20
            x_start = col * cell_w + 10
            grid[y_start:y_start + vis.shape[0], x_start:x_start + vis.shape[1]] = vis

            # 添加名称标签
            if names and idx < len(names):
                label = names[idx]
                cv2.putText(grid, label, (x_start, y_start - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, [0, 0, 0], 2)

        return grid


# ============ 主函数 ============
def main():
    parser = argparse.ArgumentParser(description="SAM 交互式分割可视化工具")
    parser.add_argument('--checkpoint', type=str, default=r'D:\edge_download\sam_vit_b_01ec64.pth')
    parser.add_argument('--image', type=str, default=r'C:\Users\86159\Desktop\3$Z48A`)}OZ]WBNO[ER(K(4.png')
    parser.add_argument('--model_type', type=str, default='vit_b',
                        choices=['vit_h', 'vit_l', 'vit_b'])
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_vis', action='store_true',
                        help='批量可视化模式（需要配合其他脚本使用）')
    args = parser.parse_args()

    if args.batch_vis:
        # 批量模式：读取图像和掩膜，生成可视化
        # 这里可以集成到您的 predict_stage1_sam.py 中
        print("批量可视化模式 - 请集成到您的评估脚本中")
        print("使用方式: BatchMaskVisualizer.visualize_prediction(image, mask, 'output.png')")

    else:
        # 交互模式
        visualizer = InteractiveSAMVisualizer(
            args.checkpoint,
            args.model_type,
            args.device
        )

        if args.image:
            visualizer.run(args.image)
        else:
            print("请提供图像路径: --image <path>")


if __name__ == "__main__":
    main()