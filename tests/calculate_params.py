#!/usr/bin/env python3
"""
统计四个模型的参数量：
1. SAM (Segment Anything Model, ViT-B)
2. 叶片分割模型 (Stage1, DeepLabV3+)
3. 病灶分割模型 - 标准 (不带分类头, DeepLabV3+)
4. 病灶分割模型 - 带分类头 (DeepLabV3+WithClassifier)
"""

import sys
from pathlib import Path

# 添加项目根目录（如果使用项目内的 config）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from segment_anything import sam_model_registry

from configs.stage1_leaf_config import Stage1Config
from configs.stage2_lesion_config import Stage2Config


# ==================== 用户配置 ====================
# (根据你的实际情况修改)
SAM_CHECKPOINT = r"D:\edge_download\sam_vit_b_01ec64.pth"   # SAM权重路径
SAM_MODEL_TYPE = "vit_b"                                    # SAM模型类型
STAGE1_CONFIG = Stage1Config()   # 叶片分割模型配置
STAGE2_CONFIG = Stage2Config()   # 病灶分割模型配置
# =================================================


# ------------------------- 带分类头的阶段2模型定义 -------------------------
class DeepLabV3PlusWithClassifier(nn.Module):
    """与训练时完全一致，包含分类头"""
    def __init__(self, encoder_name, encoder_weights, num_classes, in_channels=3):
        super().__init__()
        self.seg_model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=None
        )
        # 获取 encoder 输出通道数，用于分类头
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 224, 224)
            encoder_features = self.seg_model.encoder(dummy)
            if isinstance(encoder_features, (list, tuple)):
                feat = encoder_features[-1]
            else:
                feat = encoder_features
            c = feat.shape[1]
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c, num_classes)

    def forward(self, x):
        # 推理时只返回分割结果，但分类头参数仍存在于模型中
        return self.seg_model(x)


# ------------------------- 参数统计函数 -------------------------
def count_parameters(model, model_name=""):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{model_name:40s} Total: {total:>12,} ({total/1e6:>7.2f} M)  "
          f"Trainable: {trainable:>12,} ({trainable/1e6:>7.2f} M)")
    return total, trainable


# ------------------------- 主函数 -------------------------
def main():
    print("\n" + "="*80)
    print("Parameter Statistics for Four Models")
    print("="*80)

    # ---------- 1. SAM模型 ----------
    print("\n[1/4] Loading SAM model...")
    # SAM 需要 checkpoint 才能实例化（内部加载权重），参数量与 checkpoint 无关
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.eval()
    count_parameters(sam, "SAM (ViT-B)")

    # ---------- 2. 叶片分割模型 (Stage1) ----------
    print("\n[2/4] Building Stage1 Leaf Segmentation model...")
    model_stage1 = smp.DeepLabV3Plus(
        encoder_name=STAGE1_CONFIG.encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    )
    count_parameters(model_stage1, "Stage1 Leaf (DeepLabV3+)")

    # ---------- 3. 标准病灶分割模型 (Stage2, 不带分类头) ----------
    print("\n[3/4] Building Stage2 standard lesion model (without classifier head)...")
    model_stage2_std = smp.DeepLabV3Plus(
        encoder_name=STAGE2_CONFIG.encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=STAGE2_CONFIG.num_classes,
        activation=None
    )
    count_parameters(model_stage2_std, "Stage2 Lesion (Standard)")

    # ---------- 4. 带分类头的病灶分割模型 (Stage2 with classifier) ----------
    print("\n[4/4] Building Stage2 lesion model with classifier head...")
    model_stage2_cls = DeepLabV3PlusWithClassifier(
        encoder_name=STAGE2_CONFIG.encoder_name,
        encoder_weights=None,
        num_classes=STAGE2_CONFIG.num_classes,
        in_channels=3
    )
    count_parameters(model_stage2_cls, "Stage2 Lesion (With Classifier)")

    print("\n" + "="*80)
    print("Note: SAM parameters are from the specified checkpoint; "
          "other models are built from configs without loading weights.")
    print("="*80)


if __name__ == "__main__":
    main()