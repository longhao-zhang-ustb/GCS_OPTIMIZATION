import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ====================== 1. 全局配置（顶刊配色竖线 + 放大字体） ======================
top_journal_colors = [
    '#1f77b4',  # 顶刊经典蓝（非参照组默认色）
    '#ff7f0e',  # 顶刊经典橙
    '#2ca02c',  # 顶刊经典绿
    '#d62728',  # 顶刊经典红
    '#9467bd',  # 顶刊经典紫
    '#8c564b',  # 顶刊经典棕
    '#e377c2'   # 顶刊经典粉
]
# 顶刊配色竖线（深海军蓝，醒目且专业，符合顶刊视觉规范）
ref_line_color = "#4B545F"  # 顶刊常用深海军蓝（比默认蓝更深沉，更显专业）
ref_line_width = 2.5        # 竖线宽度（适中，不突兀）
ref_line_style = (0, (3, 1))# 顶刊常用细虚线（散段适中，美观专业）

# 全局字体配置：Times New Roman（顶刊指定字体），保持放大字体
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12  # 基础字体大小（顶刊常用字号）

# ====================== 2. 数据准备（Group1~Group7，模拟GCS评分对应F1数据） ======================
feature_labels = ['All features', 'Group1', 'Group2', 'Group3', 'Group4', 'Group5', 'Group6']

# 模拟5轮F1值（替换为你的真实GCS对应数据）
f1_data = [
    [0.82, 0.80, 0.83, 0.79, 0.81],  # Group1（参照组）
    [0.85, 0.87, 0.84, 0.86, 0.85],  # Group2
    [0.83, 0.81, 0.82, 0.80, 0.82],  # Group3
    [0.90, 0.92, 0.89, 0.91, 0.90],  # Group4
    [0.88, 0.89, 0.87, 0.88, 0.89],  # Group5
    [0.92, 0.93, 0.91, 0.94, 0.92],  # Group6
    [0.86, 0.85, 0.87, 0.86, 0.85]   # Group7
]

# 提前定义参照组索引
ref_group_idx = 0
ref_group_x = ref_group_idx + 1  # Group1对应的X轴位置
ref_group_label = feature_labels[ref_group_idx]

data_matrix = f1_data
group_means = [np.mean(data) for data in data_matrix]  # 每组均值
group_stds = [np.std(data, ddof=1) for data in data_matrix]  # 每组样本标准差
flattened_data = [val for sublist in data_matrix for val in sublist]  # 展平数据

# ====================== 3. 统计检验（以Group1为参照，通用显著性结果） ======================
# 生成组别标签（用于统计检验）
group_labels_stat = []
for idx in range(len(feature_labels)):
    group_labels_stat.extend([idx] * len(data_matrix[idx]))

# 单因素ANOVA
f_stat, anova_p = stats.f_oneway(*data_matrix)
print(f"One-way ANOVA Result: F-statistic={f_stat.round(4)}, p-value={anova_p.round(6)}")
if anova_p < 0.05:
    print("Conclusion: Significant overall differences among groups, performing Tukey HSD test.")
else:
    print("Conclusion: No significant overall differences among groups.")
print("-" * 60)

# Tukey HSD事后检验
tukey_result = pairwise_tukeyhsd(np.array(flattened_data), np.array(group_labels_stat), alpha=0.05)
print(f"Tukey HSD Post-hoc Test Result:")
print(tukey_result)

# 整理显著性结果：(组索引1, 组索引2) -> (p值, 是否显著)
sig_results = {}
for row in tukey_result.summary().data[1:]:
    g1 = int(row[0])
    g2 = int(row[1])
    p_val = float(row[3])
    is_sig = row[4] == 'reject'
    sig_results[(g1, g2)] = (p_val, is_sig)
    sig_results[(g2, g1)] = (p_val, is_sig)

print(f"\nReference Group: {ref_group_label} (Mean F1={group_means[ref_group_idx].round(4)})")

# ====================== 4. 显著性符号映射函数 ======================
def get_sig_symbol(p_val):
    if p_val < 0.001:
        return '***'
    elif p_val < 0.01:
        return '**'
    elif p_val < 0.05:
        return '*'
    else:
        return 'ns'

# ====================== 5. 绘制横向扁形子图（顶刊配色竖线标记参照组） ======================
# 扁形布局：顶刊常用比例（宽18，高4）
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 4), sharey=True)
axes = [ax1, ax2]
split_ratios = ["7:3 (Train:Test)", "6:4 (Train:Test)"]  # 两组标题

# 统一标注参数
y_label_offset = 0.012
y_mean_std_offset = 0.018
y_min_global = min(flattened_data) - 0.08
y_max_global = max(flattened_data) + 0.08

for idx, (ax, split_ratio) in enumerate(zip(axes, split_ratios)):
    # 第一步：绘制顶刊配色竖线（标记参照组，先画竖线避免被遮挡）
    ax.axvline(
        x=ref_group_x,
        ymin=0,
        ymax=1,
        color=ref_line_color,
        linewidth=ref_line_width,
        alpha=0.9,  # 透明度适中，专业不刺眼
        linestyle=ref_line_style,
        zorder=2
    )

    # 第二步：绘制箱线图（顶刊经典配色）
    box_plot = ax.boxplot(
        data_matrix,
        patch_artist=True,
        labels=feature_labels,
        medianprops={'color': 'white', 'linewidth': 2},  # 白色中位数线，顶刊常用
        whiskerprops={'color': 'black', 'linewidth': 1.2},
        capprops={'color': 'black', 'linewidth': 1.2},
        flierprops={'marker': 'None'},
        widths=0.6
    )

    # 为箱线图分配顶刊配色
    for i, patch in enumerate(box_plot['boxes']):
        patch.set_facecolor(top_journal_colors[i])
        patch.set_alpha(0.7)  # 填充透明度，顶刊常用样式

    # 第三步：叠加透明散点（与箱线图配色呼应）
    for group_idx, (data, color) in enumerate(zip(data_matrix, top_journal_colors)):
        x_jitter = np.random.normal(group_idx + 1, 0.04, size=len(data))
        ax.scatter(
            x_jitter, data,
            color=color,
            s=40,
            alpha=0.6,
            edgecolors='black',
            linewidth=0.8,
            zorder=3  # 散点在竖线上方，不被遮挡
        )

    # 图例（匹配顶刊配色竖线样式）
    ax.plot([], [], color=ref_line_color, linewidth=ref_line_width, linestyle=ref_line_style,
            alpha=0.9, label=f'Reference Group ({ref_group_label})')
    # 设置不带边框的图例
    ax.legend(loc='upper right', fontsize=12, framealpha=0)  # 顶刊常用不带边框图例

    # 每组上方标注显著性符号（顶刊常用字号）
    for group_idx in range(len(feature_labels)):
        if group_idx == ref_group_idx:
            continue
        p_val, is_sig = sig_results[(ref_group_idx, group_idx)]
        sig_symbol = get_sig_symbol(p_val)
        current_group_max = max(data_matrix[group_idx])
        label_y_pos = current_group_max + y_label_offset
        ax.text(group_idx + 1, label_y_pos, sig_symbol, ha='center', va='bottom',
                fontsize=14, fontweight='bold', zorder=4)

    # 每组下方标注均值±标准差（顶刊常用样式）
    for group_idx in range(len(feature_labels)):
        current_group_min = min(data_matrix[group_idx])
        label_y_pos = current_group_min - y_mean_std_offset
        mean_std_text = f"{group_means[group_idx]:.3f} ± {group_stds[group_idx]:.3f}"
        ax.text(group_idx + 1, label_y_pos, mean_std_text, ha='center', va='top',
                fontsize=13, color='black', zorder=4)

    # 子图标题与X轴标签（顶刊常用字号和样式）
    # ax.set_title(split_ratio, fontsize=14, fontweight='bold', pad=8)
    ax.set_xlabel('Groups', fontsize=12, fontweight='bold', labelpad=12)

    # X轴刻度标签
    ax.set_xticks(range(1, len(feature_labels)+1))
    ax.set_xticklabels(feature_labels, fontsize=13)

    # Y轴范围与网格线（顶刊常用浅网格）
    ax.set_ylim(y_min_global, y_max_global)
    ax.grid(axis='y', alpha=0.2, linestyle='-', linewidth=0.8, zorder=1)  # 网格线在竖线下方
    ax.set_axisbelow(True)

# ====================== 6. 全局配置（顶刊样式优化） ======================
# 左侧子图Y轴标签
ax1.set_ylabel('Macro Average F1-score', fontsize=12, fontweight='bold', labelpad=10)
# Y轴刻度标签
y_ticks = ax1.get_yticks()
plt.yticks(ticks=y_ticks, fontsize=10)

# 左侧子图全局说明（顶刊常用注释样式）
sig_annotation = "Note: *p<0.05, **p<0.01, ***p<0.001, ns=not significant (vs All features)"
ax1.text(0.02, 0.95, sig_annotation, ha='left', va='top',
         fontsize=12, style='italic', transform=ax1.transAxes,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7))

# 调整子图间距（顶刊常用紧凑布局）
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.subplots_adjust(wspace=0.08)

# ====================== 7. 显示图片 ======================
plt.show()
