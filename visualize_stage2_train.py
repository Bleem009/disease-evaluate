#!/usr/bin/env python3
"""
可视化阶段二模型（病灶多分类分割）的训练历史
生成一张 2x2 布局的综合图表：Loss、mIoU、mDice、训练时间
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline

# ==================== 配置 ====================
JSON_PATH = r"C:\Users\86159\PycharmProjects\disease_evaluation\checkpoints\stage2\training_history.json"   # 请修改为实际路径
# =============================================

# 读取数据
with open(JSON_PATH, 'r') as f:
    data = json.load(f)

epochs = [entry['epoch'] for entry in data]
train_loss = [entry['train']['loss'] for entry in data]
val_loss   = [entry['val']['loss'] for entry in data]
train_miou = [entry['train']['miou'] for entry in data]
val_miou   = [entry['val']['miou'] for entry in data]
train_mdice= [entry['train']['mdice'] for entry in data]
val_mdice  = [entry['val']['mdice'] for entry in data]
time_epoch = [entry['time'] for entry in data]

# 计算验证集最佳值及对应 epoch
best_loss_epoch = np.argmin(val_loss) + 1
best_loss_val = val_loss[best_loss_epoch-1]
best_miou_epoch = np.argmax(val_miou) + 1
best_miou_val = val_miou[best_miou_epoch-1]
best_mdice_epoch = np.argmax(val_mdice) + 1
best_mdice_val = val_mdice[best_mdice_epoch-1]
mean_time = np.mean(time_epoch)

# ==================== 全局样式 ====================
sns.set_theme(style="darkgrid", context="talk")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'

# 创建 2x2 子图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
ax1, ax2, ax3, ax4 = axes.flatten()

# -------------------- 1. Loss --------------------
ax1.plot(epochs, train_loss, label='Train', marker='o', markersize=3, alpha=0.6, color='royalblue')
ax1.plot(epochs, val_loss, label='Validation', marker='s', markersize=3, alpha=0.6, color='orangered')
# 平滑曲线
if len(epochs) > 3:
    x_smooth = np.linspace(min(epochs), max(epochs), 300)
    spl_train = make_interp_spline(epochs, train_loss, k=3)(x_smooth)
    spl_val = make_interp_spline(epochs, val_loss, k=3)(x_smooth)
    ax1.plot(x_smooth, spl_train, alpha=0.3, linewidth=1.5, color='royalblue')
    ax1.plot(x_smooth, spl_val, alpha=0.3, linewidth=1.5, color='orangered')
# 最佳值横线
ax1.axhline(y=best_loss_val, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
ax1.text(0.98, best_loss_val, f'Best Val {best_loss_val:.4f} @ E{best_loss_epoch}',
         transform=ax1.get_yaxis_transform(), ha='right', va='bottom',
         fontsize=10, color='red', bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))
ax1.set_title('Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend(loc='upper right')
ax1.grid(True, linestyle='--', alpha=0.5)
# 设置 y 轴范围，留出标注空间
ylim1 = ax1.get_ylim()
ax1.set_ylim(ylim1[0], ylim1[1] + (ylim1[1]-ylim1[0])*0.05)

# -------------------- 2. mIoU --------------------
ax2.plot(epochs, train_miou, label='Train', marker='o', markersize=3, alpha=0.6, color='royalblue')
ax2.plot(epochs, val_miou, label='Validation', marker='s', markersize=3, alpha=0.6, color='orangered')
if len(epochs) > 3:
    spl_train = make_interp_spline(epochs, train_miou, k=3)(x_smooth)
    spl_val = make_interp_spline(epochs, val_miou, k=3)(x_smooth)
    ax2.plot(x_smooth, spl_train, alpha=0.3, linewidth=1.5, color='royalblue')
    ax2.plot(x_smooth, spl_val, alpha=0.3, linewidth=1.5, color='orangered')
ax2.axhline(y=best_miou_val, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
ax2.text(0.98, best_miou_val, f'Best Val {best_miou_val:.4f} @ E{best_miou_epoch}',
         transform=ax2.get_yaxis_transform(), ha='right', va='bottom',
         fontsize=10, color='red', bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))
ax2.set_title('mIoU (Mean Intersection over Union)')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('mIoU')
ax2.legend(loc='lower right')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.set_ylim(0, 1.05)  # mIoU 范围 0~1，留出标注空间

# -------------------- 3. mDice --------------------
ax3.plot(epochs, train_mdice, label='Train', marker='o', markersize=3, alpha=0.6, color='royalblue')
ax3.plot(epochs, val_mdice, label='Validation', marker='s', markersize=3, alpha=0.6, color='orangered')
if len(epochs) > 3:
    spl_train = make_interp_spline(epochs, train_mdice, k=3)(x_smooth)
    spl_val = make_interp_spline(epochs, val_mdice, k=3)(x_smooth)
    ax3.plot(x_smooth, spl_train, alpha=0.3, linewidth=1.5, color='royalblue')
    ax3.plot(x_smooth, spl_val, alpha=0.3, linewidth=1.5, color='orangered')
ax3.axhline(y=best_mdice_val, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
ax3.text(0.98, best_mdice_val, f'Best Val {best_mdice_val:.4f} @ E{best_mdice_epoch}',
         transform=ax3.get_yaxis_transform(), ha='right', va='bottom',
         fontsize=10, color='red', bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))
ax3.set_title('mDice (Mean Dice Coefficient)')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('mDice')
ax3.legend(loc='lower right')
ax3.grid(True, linestyle='--', alpha=0.5)
ax3.set_ylim(0, 1.05)

# -------------------- 4. 训练时间 --------------------
ax4.plot(epochs, time_epoch, marker='o', markersize=4, color='teal', linewidth=2, label='Time per Epoch')
ax4.axhline(y=mean_time, color='crimson', linestyle='--', linewidth=2, label=f'Mean = {mean_time:.1f} s')
# 填充高于/低于平均值的区域
above = np.where(np.array(time_epoch) > mean_time)[0]
below = np.where(np.array(time_epoch) <= mean_time)[0]
if len(above) > 0:
    ax4.fill_between(np.array(epochs)[above], mean_time, np.array(time_epoch)[above],
                     color='lightcoral', alpha=0.3, interpolate=True)
if len(below) > 0:
    ax4.fill_between(np.array(epochs)[below], mean_time, np.array(time_epoch)[below],
                     color='lightgreen', alpha=0.3, interpolate=True)
ax4.set_title('Training Time per Epoch')
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Time (seconds)')
ax4.legend(loc='best')
ax4.grid(True, linestyle='--', alpha=0.5)

# 全局调整
plt.suptitle('Stage2 Training History (Multi-class Lesion Segmentation)', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# 可选：保存图片（取消注释）
# fig.savefig('stage2_training.png', dpi=300, bbox_inches='tight')