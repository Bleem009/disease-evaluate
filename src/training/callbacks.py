# src/training/callbacks.py
"""训练回调函数"""

import torch
from pathlib import Path
import json
from typing import Dict, Any
import numpy as np


class ModelCheckpoint:
    """模型检查点保存"""

    def __init__(
            self,
            filepath: Path,
            monitor: str = 'val_iou',
            mode: str = 'max',
            save_best_only: bool = True,
            save_weights_only: bool = False,
            verbose: int = 1
    ):
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.verbose = verbose

        self.best_score = float('-inf') if mode == 'max' else float('inf')
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, model, epoch: int, logs: Dict[str, Any]):
        current = logs.get(self.monitor)
        if current is None:
            return

        # 判断是否更好
        if self.mode == 'max':
            improved = current > self.best_score
        else:
            improved = current < self.best_score

        if improved:
            self.best_score = current
            if self.verbose:
                print(f"\nEpoch {epoch}: {self.monitor} improved to {current:.4f}, saving model")

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_score': self.best_score,
                'logs': logs
            }

            torch.save(checkpoint, self.filepath)

        elif not self.save_best_only:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'logs': logs
            }
            torch.save(checkpoint, self.filepath.parent / f"epoch_{epoch}.pth")


class EarlyStopping:
    """早停"""

    def __init__(
            self,
            monitor: str = 'val_loss',
            mode: str = 'min',
            patience: int = 10,
            min_delta: float = 1e-4,
            verbose: int = 1
    ):
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, logs: Dict[str, Any]) -> bool:
        current = logs.get(self.monitor)
        if current is None:
            return False

        if self.best_score is None:
            self.best_score = current
            return False

        # 判断是否改善
        if self.mode == 'min':
            improved = current < (self.best_score - self.min_delta)
        else:
            improved = current > (self.best_score + self.min_delta)

        if improved:
            self.best_score = current
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"\nEarlyStopping counter: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"\nEarly stopping triggered! Best {self.monitor}: {self.best_score:.4f}")

        return self.early_stop


class LRScheduler:
    """学习率调度包装"""

    def __init__(self, scheduler):
        self.scheduler = scheduler

    def step(self, metrics=None, epoch=None):
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(metrics)
        else:
            self.scheduler.step()

    def get_last_lr(self):
        return self.scheduler.get_last_lr()


class TensorBoardLogger:
    """TensorBoard日志（简化版，不依赖tensorboard）"""

    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history = []

    def add_scalar(self, tag: str, value: float, step: int):
        self.history.append({
            'step': step,
            'tag': tag,
            'value': value
        })

    def add_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        for tag, value in tag_scalar_dict.items():
            self.add_scalar(f"{main_tag}/{tag}", value, step)

    def close(self):
        # 保存为JSON
        with open(self.log_dir / "scalars.json", 'w') as f:
            json.dump(self.history, f, indent=2)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()