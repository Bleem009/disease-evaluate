# src/training/trainer.py
"""通用训练器"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict, Optional, Callable
import json
import time
from tqdm import tqdm
import logging

from ..utils.logger import setup_logger


class SegmentationTrainer:
    """分割模型训练器"""

    def __init__(
            self,
            model: nn.Module,
            config,
            train_loader: DataLoader,
            val_loader: DataLoader,
            criterion: nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: Optional = None,
            metrics: Optional[Dict[str, Callable]] = None
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = metrics or {}

        self.device = torch.device(config.device)
        self.model.to(self.device)

        # 日志
        self.logger = setup_logger("trainer", config.log_dir / "train.log")
        self.writer = SummaryWriter(config.log_dir / "tensorboard")

        # 训练状态
        self.current_epoch = 0
        self.best_metric = 0.0
        self.patience_counter = 0
        self.history = []

        # 检查点路径
        self.best_model_path = config.output_dir / "best_model.pth"
        self.latest_model_path = config.output_dir / "latest_model.pth"

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """
        在指定数据加载器上评估模型（可用于测试集）
        Args:
            loader: 数据加载器（如 test_loader）
        Returns:
            包含损失和各项指标的字典
        """
        self.model.eval()
        total_loss = 0.0
        metric_sums = {name: 0.0 for name in self.metrics.keys()}

        for batch in tqdm(loader, desc="Evaluation"):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            total_loss += loss.item()

            for name, metric_fn in self.metrics.items():
                metric_sums[name] += metric_fn(outputs, masks)

        num_batches = len(loader)
        results = {'loss': total_loss / num_batches}
        for name, value in metric_sums.items():
            results[name] = value / num_batches

        return results

    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        metric_sums = {name: 0.0 for name in self.metrics.keys()}

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        for batch in pbar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)

            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # 计算指标
            with torch.no_grad():
                for name, metric_fn in self.metrics.items():
                    metric_sums[name] += metric_fn(outputs, masks)

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # 平均
        num_batches = len(self.train_loader)
        results = {'loss': total_loss / num_batches}
        for name, value in metric_sums.items():
            results[name] = value / num_batches

        return results

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        metric_sums = {name: 0.0 for name in self.metrics.keys()}

        for batch in tqdm(self.val_loader, desc="Validation"):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            total_loss += loss.item()

            for name, metric_fn in self.metrics.items():
                metric_sums[name] += metric_fn(outputs, masks)

        num_batches = len(self.val_loader)
        results = {'loss': total_loss / num_batches}
        for name, value in metric_sums.items():
            results[name] = value / num_batches

        return results

    def train(self):
        """完整训练流程"""
        self.logger.info(f"Starting training for {self.config.num_epochs} epochs")

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch + 1
            start_time = time.time()

            # 训练
            train_results = self.train_epoch()

            # 验证
            val_results = self.validate()

            # 学习率调整
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_results['loss'])
                else:
                    self.scheduler.step()

            # 记录
            epoch_time = time.time() - start_time
            self._log_epoch(train_results, val_results, epoch_time)

            # 保存检查点
            self._save_checkpoint(val_results)

            # 早停检查
            if self._check_early_stop(val_results):
                self.logger.info("Early stopping triggered")
                break

        self._save_history()
        self.writer.close()
        self.logger.info("Training completed!")

    def _log_epoch(self, train_res, val_res, epoch_time):
        """记录epoch信息"""
        log_str = f"Epoch {self.current_epoch}/{self.config.num_epochs} | "
        log_str += f"Time: {epoch_time:.1f}s | "
        log_str += f"Train Loss: {train_res['loss']:.4f} | "
        log_str += f"Val Loss: {val_res['loss']:.4f}"

        for name, value in val_res.items():
            if name != 'loss':
                log_str += f" | Val {name}: {value:.4f}"

        self.logger.info(log_str)

        # TensorBoard
        self.writer.add_scalar('Loss/train', train_res['loss'], self.current_epoch)
        self.writer.add_scalar('Loss/val', val_res['loss'], self.current_epoch)
        for name, value in val_res.items():
            if name != 'loss':
                self.writer.add_scalar(f'Metrics/{name}', value, self.current_epoch)

        # 记录到history
        record = {
            'epoch': self.current_epoch,
            'train': train_res,
            'val': val_res,
            'time': epoch_time
        }
        self.history.append(record)

    def _save_checkpoint(self, val_results):
        """保存模型检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_results': val_results,
            'config': self.config
        }

        # 总是保存最新
        torch.save(checkpoint, self.latest_model_path)

        # 保存最佳
        current_metric = val_results.get('iou', val_results['loss'])
        is_better = current_metric > self.best_metric if 'iou' in val_results else current_metric < self.best_metric

        if is_better:
            self.best_metric = current_metric
            torch.save(checkpoint, self.best_model_path)
            self.logger.info(f"  -> Saved best model (metric: {current_metric:.4f})")
            self.patience_counter = 0
        else:
            self.patience_counter += 1

    def _check_early_stop(self, val_results) -> bool:
        """检查是否应该早停"""
        return self.patience_counter >= self.config.early_stop_patience

    def _save_history(self):
        """保存训练历史"""
        history_path = self.config.output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

