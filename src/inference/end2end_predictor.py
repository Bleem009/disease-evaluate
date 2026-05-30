# src/inference/end2end_predictor.py
"""
端到端两阶段预测器
训练好后使用，完全自动：原始图 → 叶片分割 → 裁剪 → 病灶分割 → 结果
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Union, Optional, Dict
import segmentation_models_pytorch as smp

from ..data.transforms import get_validation_transforms


class End2EndPredictor:
    """
    端到端两阶段预测器

    使用训练好的两个模型，自动完成：
    1. 叶片分割（Stage 1）
    2. 根据预测结果裁剪
    3. 病灶分割（Stage 2）
    4. 结果映射回原图
    """

    def __init__(
            self,
            stage1_model_path: Path,
            stage2_model_path: Path,
            stage1_encoder: str = 'mobilenet_v2',
            stage2_encoder: str = 'mobilenet_v2',  # 或resnet50
            device: str = 'cuda',
            img_size: int = 640,
            conf_threshold: float = 0.5
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        self.conf_threshold = conf_threshold

        print(f"Loading End2End Predictor...")
        print(f"  Device: {self.device}")

        # 加载Stage 1模型
        print(f"  Stage 1: {stage1_encoder}")
        self.stage1_model = self._load_model(stage1_model_path, stage1_encoder)

        # 加载Stage 2模型
        print(f"  Stage 2: {stage2_encoder}")
        self.stage2_model = self._load_model(stage2_model_path, stage2_encoder)

        # 预处理
        self.transform = get_validation_transforms(img_size)

        print("  Predictor ready!")

    def _load_model(self, model_path: Path, encoder_name: str):
        """加载模型"""
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=1
        )

        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        return model

    @torch.no_grad()
    def predict(self, image: Union[Path, str, np.ndarray]) -> Dict:
        """
        端到端预测完整流程

        Args:
            image: 输入图像（路径或numpy数组，RGB格式）

        Returns:
            包含以下键的字典:
            - original_image: 原始图像
            - leaf_mask: 预测的叶片掩膜（原图尺寸）
            - lesion_mask: 预测的病灶掩膜（原图尺寸）
            - cropped_leaf: 裁剪后的叶片图像（调试用）
            - severity: 病害严重度（%）
            - leaf_pixels: 叶片像素数
            - lesion_pixels: 病灶像素数
            - processing_info: 处理信息
        """
        # ========== Step 1: 加载原始图像 ==========
        if isinstance(image, (Path, str)):
            orig_image = np.array(Image.open(image).convert('RGB'))
        else:
            orig_image = image.copy()

        orig_h, orig_w = orig_image.shape[:2]

        # ========== Step 2: Stage 1 叶片分割 ==========
        # 预处理
        input_tensor = self.transform(image=orig_image)['image'].unsqueeze(0).to(self.device)

        # 推理
        stage1_output = self.stage1_model(input_tensor)
        stage1_prob = torch.sigmoid(stage1_output)

        # 上采样到原始尺寸
        leaf_mask = F.interpolate(
            stage1_prob,
            size=(orig_h, orig_w),
            mode='bilinear',
            align_corners=False
        ).squeeze().cpu().numpy()

        # 二值化
        leaf_mask = (leaf_mask > self.conf_threshold).astype(np.uint8)

        # 形态学优化（可选）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)

        leaf_pixels = int(np.sum(leaf_mask))

        # 检查是否检测到叶片
        if leaf_pixels == 0:
            return {
                'original_image': orig_image,
                'leaf_mask': leaf_mask,
                'lesion_mask': np.zeros((orig_h, orig_w), dtype=np.uint8),
                'cropped_leaf': None,
                'severity': 0.0,
                'leaf_pixels': 0,
                'lesion_pixels': 0,
                'processing_info': {
                    'stage1_success': False,
                    'error': 'No leaf detected'
                }
            }

        # ========== Step 3: 裁剪叶片区域 ==========
        cropped_leaf, bbox = self._crop_leaf(orig_image, leaf_mask, padding=20)

        if cropped_leaf is None:
            return {
                'original_image': orig_image,
                'leaf_mask': leaf_mask,
                'lesion_mask': np.zeros((orig_h, orig_w), dtype=np.uint8),
                'cropped_leaf': None,
                'severity': 0.0,
                'leaf_pixels': leaf_pixels,
                'lesion_pixels': 0,
                'processing_info': {
                    'stage1_success': True,
                    'crop_success': False,
                    'error': 'Failed to crop leaf'
                }
            }

        crop_h, crop_w = cropped_leaf.shape[:2]

        # ========== Step 4: Stage 2 病灶分割 ==========
        stage2_input = self.transform(image=cropped_leaf)['image'].unsqueeze(0).to(self.device)

        stage2_output = self.stage2_model(stage2_input)
        stage2_prob = torch.sigmoid(stage2_output)

        # 上采样到裁剪尺寸
        lesion_mask_cropped = F.interpolate(
            stage2_prob,
            size=(crop_h, crop_w),
            mode='bilinear',
            align_corners=False
        ).squeeze().cpu().numpy()

        # 二值化
        lesion_mask_cropped = (lesion_mask_cropped > self.conf_threshold).astype(np.uint8)

        # ========== Step 5: 映射回原图 ==========
        x1, y1, x2, y2 = bbox
        lesion_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        lesion_mask[y1:y2, x1:x2] = lesion_mask_cropped

        # 只在叶片区域内保留病灶（逻辑AND）
        lesion_mask = lesion_mask * leaf_mask

        lesion_pixels = int(np.sum(lesion_mask))
        severity = (lesion_pixels / leaf_pixels * 100) if leaf_pixels > 0 else 0.0

        return {
            'original_image': orig_image,
            'leaf_mask': leaf_mask,
            'lesion_mask': lesion_mask,
            'cropped_leaf': cropped_leaf,
            'severity': severity,
            'leaf_pixels': leaf_pixels,
            'lesion_pixels': lesion_pixels,
            'processing_info': {
                'stage1_success': True,
                'crop_success': True,
                'stage2_success': True,
                'original_size': (orig_h, orig_w),
                'crop_bbox': bbox,
                'cropped_size': (crop_h, crop_w)
            }
        }

    def _crop_leaf(self, image: np.ndarray, mask: np.ndarray, padding: int = 20):
        """根据掩膜裁剪叶片"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None, None

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)

        cropped = image[y1:y2, x1:x2]

        return cropped, (x1, y1, x2, y2)

    def predict_batch(
            self,
            image_paths: list,
            batch_size: int = 1  # 端到端通常batch=1
    ) -> list:
        """批量预测"""
        results = []
        for path in image_paths:
            results.append(self.predict(path))
        return results