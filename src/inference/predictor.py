# src/inference/predictor.py
"""两阶段预测器"""

import os

# 必须在导入torch前设置
os.environ['TORCH_CNNPACK'] = '0'
os.environ['USE_XNNPACK'] = '0'
os.environ['TORCH_DISABLE_XNNPACK'] = '1'

import warnings

warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, Union
import segmentation_models_pytorch as smp

# 禁用XNNPACK
torch.backends.xnnpack.enabled = False

from ..data.transforms import get_validation_transforms


class TwoStagePredictor:
    """
    两阶段病害分割预测器
    """

    def __init__(
            self,
            stage1_model_path: Path,
            stage2_model_path: Path,
            device: str = 'cuda',
            img_size: int = 640,
            conf_threshold: float = 0.5,
            use_imagenet_only: bool = False  # 新增：仅使用ImageNet预训练
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        self.conf_threshold = conf_threshold

        # 加载Stage 1
        if use_imagenet_only or not Path(stage1_model_path).exists():
            print("Using ImageNet pretrained weights for Stage 1")
            self.stage1_model = self._create_imagenet_model('mobilenet_v2')
        else:
            try:
                self.stage1_model = self._load_model_safe(stage1_model_path, 'mobilenet_v2')
            except Exception as e:
                print(f"Failed to load Stage 1 from {stage1_model_path}: {e}")
                print("Falling back to ImageNet pretrained weights")
                self.stage1_model = self._create_imagenet_model('mobilenet_v2')

        # 加载Stage 2
        if use_imagenet_only or not Path(stage2_model_path).exists():
            print("Using ImageNet pretrained weights for Stage 2")
            self.stage2_model = self._create_imagenet_model('resnet50')
        else:
            try:
                self.stage2_model = self._load_model_safe(stage2_model_path, 'resnet50')
            except Exception as e:
                print(f"Failed to load Stage 2 from {stage2_model_path}: {e}")
                print("Falling back to ImageNet pretrained weights")
                self.stage2_model = self._create_imagenet_model('resnet50')

        # 预处理
        self.transform = get_validation_transforms(img_size)

        print(f"Predictor initialized on {self.device}")

    def _create_imagenet_model(self, encoder_name: str):
        """创建ImageNet预训练模型"""
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights='imagenet',  # 使用ImageNet预训练
            in_channels=3,
            classes=1,
            activation=None
        )
        model.to(self.device)
        model.eval()
        return model

    def _load_model_safe(self, model_path: Path, encoder_name: str):
        """安全加载模型（支持多种格式）"""
        model_path = Path(model_path)

        # 创建模型架构（无预训练）
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None
        )

        # 尝试加载权重
        try:
            # 首先尝试作为普通checkpoint加载
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('model_state_dict',
                                            checkpoint.get('state_dict', checkpoint))
            else:
                raise ValueError("Checkpoint format not recognized")

            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded weights from {model_path}")

        except Exception as e:
            print(f"Standard loading failed: {e}")

            # 尝试作为TorchScript加载并提取
            try:
                # 使用torch.jit.load但禁用优化
                with torch.no_grad():
                    # 先尝试加载到CPU
                    temp_model = torch.jit.load(model_path, map_location='cpu')

                    # 提取参数
                    state_dict = {}
                    for name, param in temp_model.named_parameters():
                        state_dict[name] = param.data

                    # 适配键名
                    adapted_dict = {}
                    for k, v in state_dict.items():
                        # 移除常见前缀
                        new_k = k.replace('module.', '').replace('_orig_mod.', '')
                        # 适配可能的编码器名称差异
                        if 'encoder.model' in new_k and 'mobilenet' in encoder_name:
                            new_k = new_k.replace('encoder.model', 'encoder')
                        adapted_dict[new_k] = v

                    model.load_state_dict(adapted_dict, strict=False)
                    print(f"Loaded weights from TorchScript: {model_path}")

            except Exception as e2:
                raise RuntimeError(f"Failed to load model: {e2}")

        model.to(self.device)
        model.eval()
        return model

    # ... 其余方法保持不变 ...

    @torch.no_grad()
    def predict(self, image: Union[Path, str, np.ndarray], return_intermediate: bool = False):
        """执行预测"""
        # 加载图像
        if isinstance(image, (Path, str)):
            orig_image = np.array(Image.open(image).convert('RGB'))
        else:
            orig_image = image.copy()

        orig_h, orig_w = orig_image.shape[:2]

        # Stage 1: 叶片分割
        input_tensor = self.transform(image=orig_image)['image'].unsqueeze(0).to(self.device)

        stage1_output = self.stage1_model(input_tensor)
        stage1_prob = torch.sigmoid(stage1_output)

        leaf_mask = F.interpolate(
            stage1_prob,
            size=(orig_h, orig_w),
            mode='bilinear',
            align_corners=False
        ).squeeze().cpu().numpy()

        leaf_mask = (leaf_mask > self.conf_threshold).astype(np.uint8)

        # 形态学优化
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)

        leaf_pixels = int(np.sum(leaf_mask))

        if leaf_pixels == 0:
            result = {
                'leaf_mask': leaf_mask,
                'lesion_mask': np.zeros_like(leaf_mask),
                'severity': 0.0,
                'leaf_pixels': 0,
                'lesion_pixels': 0
            }
            if return_intermediate:
                result['cropped_leaf'] = None
            return result

        # 裁剪叶片
        cropped_leaf, bbox = self._crop_leaf(orig_image, leaf_mask, padding=20)

        if cropped_leaf is None or cropped_leaf.size == 0:
            result = {
                'leaf_mask': leaf_mask,
                'lesion_mask': np.zeros_like(leaf_mask),
                'severity': 0.0,
                'leaf_pixels': leaf_pixels,
                'lesion_pixels': 0
            }
            if return_intermediate:
                result['cropped_leaf'] = None
            return result

        crop_h, crop_w = cropped_leaf.shape[:2]

        # Stage 2: 病灶分割
        stage2_input = self.transform(image=cropped_leaf)['image'].unsqueeze(0).to(self.device)

        stage2_output = self.stage2_model(stage2_input)
        stage2_prob = torch.sigmoid(stage2_output)

        lesion_mask_cropped = F.interpolate(
            stage2_prob,
            size=(crop_h, crop_w),
            mode='bilinear',
            align_corners=False
        ).squeeze().cpu().numpy()

        lesion_mask_cropped = (lesion_mask_cropped > self.conf_threshold).astype(np.uint8)

        # 映射回原图
        x1, y1, x2, y2 = bbox
        lesion_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        lesion_mask[y1:y2, x1:x2] = lesion_mask_cropped

        lesion_mask = lesion_mask * leaf_mask
        lesion_pixels = int(np.sum(lesion_mask))
        severity = (lesion_pixels / leaf_pixels * 100) if leaf_pixels > 0 else 0.0

        result = {
            'leaf_mask': leaf_mask,
            'lesion_mask': lesion_mask,
            'severity': severity,
            'leaf_pixels': leaf_pixels,
            'lesion_pixels': lesion_pixels
        }

        if return_intermediate:
            result['cropped_leaf'] = cropped_leaf
            result['bbox'] = bbox
            result['lesion_mask_cropped'] = lesion_mask_cropped

        return result

    def _crop_leaf(self, image: np.ndarray, mask: np.ndarray, padding: int = 20):
        """裁剪叶片区域"""
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

    def predict_batch(self, images: list, batch_size: int = 4):
        """批量预测"""
        results = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            for img in batch:
                results.append(self.predict(img))
        return results