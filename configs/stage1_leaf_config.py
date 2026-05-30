from dataclasses import dataclass, field
from typing import Tuple
from pathlib import Path
from configs.base_config import BaseConfig
import torch
@dataclass
class Stage1Config(BaseConfig):
    """阶段1：叶片分割"""

    # 数据路径 - 根据你的实际路径修改
    train_img_dir = Path(r"C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\stage1\train\images")
    train_label_dir= Path(r"C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\stage1\train\labels")
    val_img_dir = Path(r"C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\stage1\val\images")
    val_label_dir = Path(r"C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\stage1\val\labels")
    test_img_dir = Path(r"C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\stage1\test\images")
    test_label_dir = Path(r"C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\stage1\test\labels")

    encoder_name: str = "mobilenet_v2"

    @property
    def pretrained_path(self) -> Path:
        return self.checkpoint_dir / "pretrained" / "1stage_optimized.ptl"

    @property
    def output_dir(self) -> Path:
        return self.checkpoint_dir / "stage1"

    @property
    def log_dir(self) -> Path:
        return super().log_dir / "stage1"

    aug_rotation: int = 90
    aug_scale: Tuple[float, float] = (0.8, 1.2)

    def __post_init__(self):
        super().__post_init__()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)