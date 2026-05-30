# src/inference/visualizer.py
"""结果可视化（补充端到端版本）"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import Optional


class End2EndVisualizer:
    """端到端结果可视化"""

    COLORS = {
        'leaf': [0, 255, 0],  # 绿色
        'lesion': [255, 0, 0],  # 红色
    }

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def visualize_prediction(
            self,
            result: dict,
            save_path: Optional[Path] = None,
            show: bool = True
    ):
        """
        可视化端到端预测结果

        Args:
            result: End2EndPredictor.predict()的输出
        """
        orig_image = result['original_image']
        leaf_mask = result['leaf_mask']
        lesion_mask = result['lesion_mask']
        severity = result['severity']
        info = result['processing_info']

        # 创建叠加图
        overlay = orig_image.copy()

        # 叶片区域（绿色半透明）
        leaf_color = np.array(self.COLORS['leaf'])
        overlay[leaf_mask > 0] = overlay[leaf_mask > 0] * (1 - self.alpha) + leaf_color * self.alpha

        # 病灶区域（红色半透明）
        lesion_color = np.array(self.COLORS['lesion'])
        overlay[lesion_mask > 0] = overlay[lesion_mask > 0] * (1 - self.alpha) + lesion_color * self.alpha

        # 创建子图
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3)

        # 原始图像
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(orig_image)
        ax1.set_title('Original Image')
        ax1.axis('off')

        # 叶片掩膜
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(leaf_mask, cmap='Greens')
        leaf_pixels = result['leaf_pixels']
        ax2.set_title(f'Leaf Mask\n{leaf_pixels:,} pixels')
        ax2.axis('off')

        # 病灶掩膜
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(lesion_mask, cmap='Reds')
        lesion_pixels = result['lesion_pixels']
        ax3.set_title(f'Lesion Mask\n{lesion_pixels:,} pixels ({severity:.2f}%)')
        ax3.axis('off')

        # 裁剪后的叶片（Stage 2输入）
        ax4 = fig.add_subplot(gs[1, 0])
        if result['cropped_leaf'] is not None:
            ax4.imshow(result['cropped_leaf'])
            h, w = result['cropped_leaf'].shape[:2]
            ax4.set_title(f'Cropped Leaf ({w}x{h})')
        else:
            ax4.text(0.5, 0.5, 'Crop Failed', ha='center', va='center')
            ax4.set_title('Cropped Leaf')
        ax4.axis('off')

        # 叠加结果
        ax5 = fig.add_subplot(gs[1, 1:])
        ax5.imshow(overlay.astype(np.uint8))
        ax5.set_title(f'Final Result - Severity: {severity:.2f}%')
        ax5.axis('off')

        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=[c / 255 for c in self.COLORS['leaf']], label='Leaf'),
            Patch(facecolor=[c / 255 for c in self.COLORS['lesion']], label='Lesion')
        ]
        ax5.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def visualize_comparison(
            self,
            result_pred: dict,
            gt_leaf_mask: Optional[np.ndarray] = None,
            gt_lesion_mask: Optional[np.ndarray] = None,
            save_path: Optional[Path] = None
    ):
        """对比预测和真值（如果有）"""
        # 实现对比可视化...
        pass