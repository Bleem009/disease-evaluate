import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline

# ======================== 设置美观样式 ========================
sns.set_theme(style="darkgrid", context="talk", palette="Set2")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['lines.linewidth'] = 2.0

# ======================== 读取数据 ========================
with open(r'C:\Users\86159\PycharmProjects\disease_evaluation\checkpoints\stage1\training_history.json', 'r') as f:
    data = json.load(f)

epochs = list(range(1, len(data) + 1))
train_loss = [e['train']['loss'] for e in data]
val_loss = [e['val']['loss'] for e in data]
train_iou = [e['train']['iou'] for e in data]
val_iou = [e['val']['iou'] for e in data]
train_dice = [e['train']['dice'] for e in data]
val_dice = [e['val']['dice'] for e in data]
train_pacc = [e['train']['pixel_acc'] for e in data]
val_pacc = [e['val']['pixel_acc'] for e in data]
time_epoch = [e['time'] for e in data]


# ======================== 辅助函数：获取验证集最值及轮次 ========================
def get_best_val_epoch(metric_values, mode='max'):
    """返回 (最佳值, 对应epoch)"""
    if mode == 'max':
        best_val = max(metric_values)
    else:  # 'min'
        best_val = min(metric_values)
    best_epoch = epochs[metric_values.index(best_val)]
    return best_val, best_epoch


# 针对各指标计算最佳值及轮次
best_loss_val, best_loss_epoch = get_best_val_epoch(val_loss, mode='min')
best_iou_val, best_iou_epoch = get_best_val_epoch(val_iou, mode='max')
best_dice_val, best_dice_epoch = get_best_val_epoch(val_dice, mode='max')
best_pacc_val, best_pacc_epoch = get_best_val_epoch(val_pacc, mode='max')

# ======================== 图1：四个核心指标子图 ========================
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
metrics_info = [
    ('Loss', train_loss, val_loss, 'loss', 'min', best_loss_val, best_loss_epoch),
    ('IoU', train_iou, val_iou, 'iou', 'max', best_iou_val, best_iou_epoch),
    ('Dice', train_dice, val_dice, 'dice', 'max', best_dice_val, best_dice_epoch),
    ('Pixel Accuracy', train_pacc, val_pacc, 'pixel_acc', 'max', best_pacc_val, best_pacc_epoch)
]

for ax, (title, train_vals, val_vals, ylabel, mode, best_val, best_epoch) in zip(axes.flat, metrics_info):
    # 绘制原始数据点（半透明）
    ax.plot(epochs, train_vals, label='Train', marker='o', markersize=3, alpha=0.6, linestyle='-')
    ax.plot(epochs, val_vals, label='Validation', marker='s', markersize=3, alpha=0.6, linestyle='-')

    # 添加平滑曲线（可选）
    if len(epochs) > 3:
        x_smooth = np.linspace(min(epochs), max(epochs), 300)
        spl_train = make_interp_spline(epochs, train_vals, k=3)(x_smooth)
        spl_val = make_interp_spline(epochs, val_vals, k=3)(x_smooth)
        ax.plot(x_smooth, spl_train, alpha=0.4, linewidth=2, label='_nolegend_')
        ax.plot(x_smooth, spl_val, alpha=0.4, linewidth=2, label='_nolegend_')

    # 添加验证集最值横线及标注
    if mode == 'min':
        line_label = f'Best Val (min): {best_val:.4f} @ Epoch {best_epoch}'
    else:
        line_label = f'Best Val (max): {best_val:.4f} @ Epoch {best_epoch}'
    ax.axhline(y=best_val, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label=line_label)

    # 在图中合适位置添加文本（避免重叠，放在右上方或线附近）
    # 根据纵轴范围动态调整文本位置
    y_min, y_max = ax.get_ylim()
    text_x = epochs[-1] * 0.95  # 靠近右边界
    text_y = best_val + (y_max - y_min) * 0.05 if best_val < (y_max + y_min) / 2 else best_val - (y_max - y_min) * 0.05
    ax.annotate(f'{best_val:.4f}', xy=(best_epoch, best_val), xytext=(text_x, text_y),
                arrowprops=dict(arrowstyle='->', color='red', lw=0.8),
                fontsize=9, color='red', bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel.replace('_', ' ').title())
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.5)

plt.suptitle('Training & Validation Metrics (with Best Validation Values)', fontsize=18, y=1.02)
plt.tight_layout()
plt.show()

# ======================== 图2：训练时间 + 平均值横线 + 填充区域 ========================
mean_time = np.mean(time_epoch)
fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(epochs, time_epoch, marker='o', markersize=4, color='teal', linewidth=2, label='Time per Epoch')
ax2.axhline(y=mean_time, color='crimson', linestyle='--', linewidth=2, label=f'Mean Time = {mean_time:.1f} s')
ax2.fill_between(epochs, mean_time, time_epoch, where=(np.array(time_epoch) > mean_time),
                 color='lightcoral', alpha=0.3, interpolate=True, label='Above Average')
ax2.fill_between(epochs, mean_time, time_epoch, where=(np.array(time_epoch) <= mean_time),
                 color='lightgreen', alpha=0.3, interpolate=True, label='Below Average')
ax2.set_title('Training Time per Epoch', fontsize=16, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Time (seconds)')
ax2.legend(loc='best')
ax2.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# ======================== 可选：高级图表（热力图、箱线图） ========================
# 3. 验证指标相关性热力图
fig3, ax3 = plt.subplots(figsize=(8, 6))
val_metrics = np.array([val_loss, val_iou, val_dice, val_pacc]).T
corr = np.corrcoef(val_metrics, rowvar=False)
sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm',
            xticklabels=['Loss', 'IoU', 'Dice', 'Pixel Acc'],
            yticklabels=['Loss', 'IoU', 'Dice', 'Pixel Acc'], ax=ax3)
ax3.set_title('Correlation among Validation Metrics', fontsize=14)
plt.tight_layout()
plt.show()

# 4. 训练时间阶段箱线图
stages = ['Early (1-50)', 'Mid (51-100)', 'Late (101-195)']
early = time_epoch[:50]
mid = time_epoch[50:100]
late = time_epoch[100:]
fig4, ax4 = plt.subplots(figsize=(8, 5))
bp = ax4.boxplot([early, mid, late], labels=stages, patch_artist=True,
                 boxprops=dict(facecolor='lightblue'), medianprops=dict(color='red'))
ax4.set_title('Training Time Distribution by Stage', fontsize=14)
ax4.set_ylabel('Seconds')
ax4.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()