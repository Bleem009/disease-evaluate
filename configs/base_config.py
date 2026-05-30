# configs/base_config.py
"""基础配置"""

from dataclasses import dataclass, field
from typing import Tuple, Optional
from pathlib import Path
import torch


@dataclass
class BaseConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    project_root: Path = Path(__file__).parent.parent

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def checkpoint_dir(self) -> Path:
        return self.project_root / "checkpoints"

    @property
    def log_dir(self) -> Path:
        return self.project_root / "logs"

    # ========== 训练参数（从头训练需要调整）==========
    num_epochs: int = 200  # 增加轮数，小数据集需要更多迭代
    batch_size: int = 8  # 如果显存允许，可以增大
    num_workers: int = 4

    # ========== 学习率（从头训练使用更大学习率）==========
    # 原论文微调：encoder=0.01, decoder=0.001
    # 从头训练：使用相同或稍大
    lr_encoder: float = 0.01
    lr_decoder: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 1e-4  # 可以稍微增大正则化

    lr_power: float = 0.9

    img_size: int = 640
    early_stop_patience: int = 50  # 增加耐心值，避免过早停止

    # ========== 预训练设置 ==========
    use_imagenet_pretrained: bool = True  # 使用ImageNet预训练
    use_paper_pretrained: bool = False  # 不使用论文权重（关键修改）

    def __post_init__(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class Stage1Config(BaseConfig):
    """阶段1：叶片分割 - MobileNetV2（轻量级，适合小数据）"""

    # 数据路径
    train_img_dir=Path(r"C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\stage1\train\images")
    train_label_dir= Path(r"C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\stage1\train\labels")
    val_img_dir=Path(r"C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\stage1\val\images")
    val_label_dir=Path(r"C:\Users\86159\PycharmProjects\disease_evaluation\data\processed\stage1\val\images")

    encoder_name: str = "mobilenet_v2"

    # 不使用论文权重
    paper_pretrained_path: Optional[Path] = None

    @property
    def output_dir(self) -> Path:
        return self.checkpoint_dir / "stage1_scratch"  # 新目录，避免覆盖

    @property
    def log_dir(self) -> Path:
        return super().log_dir / "stage1_scratch"

    # 数据增强（小数据集需要更强增强）
    aug_rotation: int = 180  # 增大旋转范围
    aug_scale: Tuple[float, float] = (0.7, 1.3)  # 更大尺度变化

    def __post_init__(self):
        super().__post_init__()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class Stage2Config(BaseConfig):
    """阶段2：病灶分割 - ResNet50（需要更多数据，考虑换轻量级）"""

    # 如果数据很少（<200张），建议改用mobilenet_v2
    # encoder_name: str = "resnet50"  # 原论文
    encoder_name: str = "mobilenet_v2"  # 小数据建议改用这个

    # 数据路径
    train_img_dir: Path = field(default_factory=lambda: Path("data/processed/stage2/train/images"))
    train_label_dir: Path = field(default_factory=lambda: Path("data/processed/stage2/train/labels"))
    val_img_dir: Path = field(default_factory=lambda: Path("data/processed/stage2/val/images"))
    val_label_dir: Path = field(default_factory=lambda: Path("data/processed/stage2/val/labels"))

    paper_pretrained_path: Optional[Path] = None

    @property
    def output_dir(self) -> Path:
        return self.checkpoint_dir / "stage2_scratch"

    @property
    def log_dir(self) -> Path:
        return super().log_dir / "stage2_scratch"

    # 更强的数据增强（病灶更难分割）
    aug_rotation: int = 180
    aug_scale: Tuple[float, float] = (0.6, 1.4)

    # 小数据用更多epoch
    num_epochs: int = 400

    def __post_init__(self):
        super().__post_init__()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)