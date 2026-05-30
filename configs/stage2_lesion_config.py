from dataclasses import dataclass, field
from typing import Tuple
from pathlib import Path
from configs.base_config import BaseConfig
import torch
@dataclass
class Stage2Config(BaseConfig):
    """阶段2：病灶分割"""

    encoder_name = 'resnet50'
    output_dir = Path("checkpoints/stage2")
    log_dir = Path("logs/stage2")
    aug_rotation = 30
    aug_scale = (0.8, 1.2)
    num_classes=5

    # 数据路径
    train_img_dir: Path = field(default_factory=lambda: Path(r"C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\stage2\train\images"))
    train_label_dir: Path = field(default_factory=lambda: Path(r"C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\stage2\train\labels"))
    val_img_dir: Path = field(default_factory=lambda: Path(r"C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\stage2\val\images"))
    val_label_dir: Path = field(default_factory=lambda: Path(r"C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\stage2\val\labels"))
    test_img_dir = Path(r"C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\stage2\test\images")  # 可选
    test_label_dir = Path(r"C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\stage2\test\labels")  # 可选

    encoder_name: str = "resnet50"

    @property
    def pretrained_path(self) -> Path:
        return self.checkpoint_dir / "pretrained" / "2stage_optimized.ptl"

    @property
    def output_dir(self) -> Path:
        return self.checkpoint_dir / "stage2"

    @property
    def log_dir(self) -> Path:
        return super().log_dir / "stage2"

    aug_rotation: int = 180
    aug_scale: Tuple[float, float] = (0.7, 1.3)

    def __post_init__(self):
        super().__post_init__()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)