import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from matplotlib.ticker import StrMethodFormatter, FuncFormatter

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

levels = ["SO", "HS", "LC", "MC", "SC"]
academic_colors = ["#20c70e", "#88af06", "#ff0e0e", "#273ed6", "#3d0a6d"]  # 学术配色
level_color = dict(zip(levels, academic_colors))

# ---------------------- 1. 统一数据（保持所有子图数据一致） ----------------------
dtProcessed = pd.read_csv(r'zigong_data\\20251227_final_processed_data.csv')
# # # 首先去掉dtProcessed中的language, motion, open_one's_eyes的-1项
# dtProcessed = dtProcessed[dtProcessed['language'] != -1]
# dtProcessed = dtProcessed[dtProcessed['motion'] != -1]
# dtProcessed = dtProcessed[dtProcessed['open_one\'s_eyes'] != -1]
# # 计算language, motion, open_one's_eyes的总分
# dtProcessed['GCS_Total'] = dtProcessed['language'] + dtProcessed['motion'] + dtProcessed['open_one\'s_eyes']
# # 将总分转为整数
# dtProcessed['GCS_Total'] = dtProcessed['GCS_Total'].astype(int)
# # 统计GCS_Total与consciousness的对应关系
# gcs_consciousness = dtProcessed.groupby('consciousness')['GCS_Total'].value_counts().unstack(fill_value=0)
# # 以热力图形式绘制出来
# plt.figure(figsize=(10, 4))
# # 自定义标注内容（添加千位分隔符）
# annot_data = np.array([["{:,.0f}".format(val) for val in row] for row in gcs_consciousness.values])
# ax = sns.heatmap(gcs_consciousness, annot=annot_data, fmt='', cmap='Greens', annot_kws={'size': 13}, cbar_kws={'label': 'Count'})
# cbar = ax.collections[0].colorbar
# cbar.set_label('Count', fontsize=13)
# cbar.ax.tick_params(labelsize=13)
# cbar.ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
# ax.set_yticklabels(['SO', 'HS', 'LC', 'MC', 'SC'])
# plt.xlabel('GCS Total Score', fontsize=13)
# plt.xticks(fontsize=13)
# plt.yticks(fontsize=13)
# plt.ylabel('Arousal Level', fontsize=13)
# plt.show()
# exit()

data = []
# 遍历dtProcessed中的每一条数据，并按照{'Verbal':1, 'Motor':1, 'Eye':1, 'Consciousness':'Sober'}的格式组织数据
for index, row in enumerate(dtProcessed.values.tolist()):
    data.append({
        'Verbal': row[12],
        'Motor': row[11],
        'Eye': row[10],
        'Consciousness': levels[int(row[13])]
    })
df = pd.DataFrame(data)

# -------------------------- 2. Jitter Function --------------------------
def add_jitter(data, jitter_amount=0.15):
    """添加轻微抖动避免散点重叠"""
    jitter = np.random.uniform(-jitter_amount, jitter_amount, size=len(data))
    return data + jitter

df['Verbal_jitter'] = add_jitter(df['Verbal'])
df['Motor_jitter'] = add_jitter(df['Motor'])
df['Eye_jitter'] = add_jitter(df['Eye'])

# -------------------------- 3. Plot Layout --------------------------
fig = plt.figure(figsize=(16, 14))
gs = fig.add_gridspec(
    3, 3, 
    hspace=0.15, wspace=0.2,
    height_ratios=[0.25, 1, 0.05],
    width_ratios=[0.3, 1, 0.25]
)
ax_top = fig.add_subplot(gs[0, 1])    # 上方：言语评分分布
ax_right = fig.add_subplot(gs[1, 2])  # 右侧：睁眼评分分布
ax_left = fig.add_subplot(gs[1, 0])   # 左侧：运动评分分布
ax_main = fig.add_subplot(gs[1, 1])   # 中心：散点图

# -------------------------- 4. Main Scatter Plot --------------------------
ax_main.set_xlabel('GCS Verbal Score', fontsize=13)
ax_main.set_ylabel('GCS Motor Score', fontsize=13)
ax_main.set_xlim(-2, 6)
ax_main.set_ylim(-2, 7)
# 设置x轴字体大小
ax_main.tick_params(axis='x', labelsize=13)
# 设置y轴字体大小
ax_main.tick_params(axis='y', labelsize=13)
ax_main.grid(alpha=0.3, linestyle='--', color='#cccccc')

ax_main_twin = ax_main.twinx()
ax_main_twin.set_ylabel('GCS Eye Opening Score', fontsize=13, rotation=-90, labelpad=15)
ax_main_twin.set_ylim(0.5, 4.5)
ax_main_twin.tick_params(axis='y', labelsize=13)

# 绘制散点（大小固定）
for level in levels:
    subset = df[df['Consciousness'] == level]
    ax_main.scatter(
        subset['Verbal_jitter'], subset['Motor_jitter'],
        s=25,  # 固定大小，取消与Eye评分的关联
        c=level_color[level], alpha=0.5,
        label=level, edgecolors='white', linewidth=0.6
    )
    ax_main_twin.scatter(
        subset['Verbal_jitter'], subset['Eye_jitter'],
        s=25,  # 次轴标记也固定大小
        c=level_color[level], alpha=0.5, edgecolors='white', linewidth=0.6
    )

# 在散点图上y主轴等于0处添加一条红色点划线
ax_main.axhline(0, color='gray', linestyle=':', linewidth=1.5, zorder=10)
# 在散点图上x轴等于0处添加一条红色虚线
ax_main.axvline(0, color='gray', linestyle=':', linewidth=1.5, zorder=10)
# 两条点划线的左上侧区域设置为不同颜色
ax_main.fill_betweenx(
    ax_main.get_ylim(), ax_main.get_xlim()[0], 0,
    color='gray', alpha=0.1, zorder=9
)
ax_main.fill_between(
    ax_main.get_xlim(), ax_main.get_ylim()[0], 0,
    color='gray', alpha=0.1, zorder=9
)

# ax_main.legend(loc='upper right', bbox_to_anchor=(1, 1), frameon=True, fancybox=True, shadow=True)
# 使用ncol参数将图例排成一行，loc和bbox_to_anchor定位到右下角
ax_main.legend(loc='lower right', ncol=len(levels), 
               frameon=False, fontsize=13)

def plot_horizontal_density(ax, data, data_jitter, title, color, linecolor):
    """水平归一化密度直方图（上方）"""
    # 获取唯一的值并排序（包括-1值）
    unique_values = np.sort(np.unique(data))
    unique_values = np.insert(unique_values, 1, 0)
    
    # -------------------------- 核心修改1：自定义bins，确保柱子中心与散点图横坐标对齐 --------------------------
    # 计算bins：每个唯一值为中心，柱子宽度设为1（与散点图坐标步长匹配）
    bin_edges = np.array([val - 0.5 for val in unique_values] + [unique_values[-1] + 0.5])
    # 强制柱子宽度为1，与散点图横坐标刻度一一对应
    bins = bin_edges  # bins格式：[x-0.5, x+0.5] 确保柱子中心在x上
    
    # 计算归一化直方图（density=True）
    counts, bins, patches = ax.hist(
        data_jitter, bins=bins,  # 使用自定义bins
        density=True,
        color=color, alpha=0.6, edgecolor='gray', linewidth=1,
        width=1.0  # -------------------------- 核心修改2：强制柱子宽度为1 --------------------------
    )
    
    # 标注数值（归一化密度值）
    bin_centers = 0.5 * (bins[1:] + bins[:-1])  # 柱子中心位置（与散点图横坐标完全一致）
    for i, (patch, count) in enumerate(zip(patches, counts)):
        # 在柱子上方标注数值（保留2位小数）
        ax.text(
            bin_centers[i], count + 0.01, f'{count:.2f}',
            ha='center', va='bottom', fontsize=13
        )
    
    # 绘制核密度曲线（归一化）
    if len(data) > 1:  # 确保有足够的数据点来计算KDE
        kde = stats.gaussian_kde(data)
        # -------------------------- 核心修改3：KDE范围与散点图x轴范围对齐 --------------------------
        x_range = np.linspace(-2, 6, 200)  # 与散点图xlim(-1.5,5.5)匹配
        ax.plot(x_range, kde(x_range), color=linecolor, linewidth=2, alpha=0.7)

    
    # 样式优化,将标题上移
    ax.set_title(title, fontsize=13, pad=20)
    ax.set_ylabel('Density', fontsize=13)  # 纵轴改为密度
    ax.set_xlim(-2, 6)
    # ax.set_xlim(-2, 6)  # 与散点图x轴范围完全匹配
    ax.tick_params(
        axis='x',        # 仅作用于X轴
        which='both',    # 同时控制主/次刻度（避免漏次要刻度线）
        bottom=False,    # 关闭底部刻度线（划线）
        labelbottom=False# 可选：关闭刻度标签（如需仅隐藏划线，可注释这行）
    )
    ax.tick_params(axis='both', labelsize=13)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.grid(alpha=0.2, linestyle='--', color='#cccccc')
    
def plot_vertical_density(ax, data, title, color, is_left=True, linecolor='gray'):
    """垂直密度直方图（左右侧分离逻辑）"""
    ax.clear()
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    # 关键修改：左侧直方图范围固定为与主散点图一致的-1.5到6.5
    # if is_left:
    y_min, y_max = -1.5, 6.5  # 与主散点图Y轴范围完全匹配
    # 计算bins：确保覆盖整个范围，每个柱子宽度为1.0
    bins = np.arange(y_min, y_max+1, 1)
    bin_width = bins[1] - bins[0]  # 柱子宽度为1.0
    
    counts, bins = np.histogram(data, bins=bins, density=True)
    density_max = np.ceil(counts.max() * 10) / 10 if counts.size > 0 else 0.1
    density_min = 0
    y_centers = 0.5 * (bins[1:] + bins[:-1])

    # 生成刻度
    density_max = density_max + 0.5 if is_left else density_max + 1
    xticks = np.linspace(density_min, density_max, 4)
    xtick_labels = [f'{x:.2f}' for x in xticks]

    if is_left:
        patches = ax.barh(
            y_centers, 
            width=counts,
            height=bin_width,  # 使用计算出的固定宽度
            color=color, alpha=0.7, edgecolor='gray', linewidth=1,
            left=np.zeros_like(counts)
        )

        ax.set_xlim(density_max, density_min)
        ax.set_ylim(y_min-0.5, y_max+0.5)  # 应用与主散点图匹配的范围

        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, fontsize=13)
        ax.set_xlabel('Density', fontsize=13)
        ax.spines['bottom'].set_visible(True)
        ax.spines['right'].set_visible(True)

        for patch, count in zip(patches, counts):
            text_x = count + 0.05
            text_y = patch.get_y() + patch.get_height()/2
            ax.text(text_x, text_y, f'{count:.2f}',
                    ha='right', va='center', fontsize=13, color='black')

        if len(data) > 1:
            kde = stats.gaussian_kde(data)
            y_range = np.linspace(y_min - 0.5, y_max + 0.5, 300)  # 使用与主散点图匹配的范围
            kde_values = kde(y_range)
            ax.plot(kde_values, y_range, color=linecolor, linewidth=2, zorder=12, alpha=0.7)

    else:
        # 绘制朝右的柱子（从左侧0位置向右绘制）
        y_min = 0.5
        y_max = 4.5
        y_centers = [1, 2, 3, 4]
        counts = counts[2:6]
        patches = ax.barh(
            y_centers, 
            width=counts,  # 宽度为密度值
            height=bin_width,
            color=color, alpha=0.7, edgecolor='gray', linewidth=1,
            left=np.zeros_like(counts)  # 从0位置开始绘制
        )

        # X轴设置
        ax.set_xlim(density_min, density_max + 0.1)
        ax.set_ylim(y_min, y_max)

        # 底部显示刻度
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, fontsize=13)
        ax.set_xlabel('Density', fontsize=13)
        ax.spines['bottom'].set_visible(True)
        ax.spines['right'].set_visible(False)

        # 柱子内标注密度值
        for patch, count in zip(patches, counts):
            # 文字位置在柱子内部靠右
            text_x = count + 0.3
            text_y = patch.get_y() + patch.get_height()/2
            ax.text(text_x, text_y, f'{count:.2f}',
                    ha='right', va='center', fontsize=13, color='black')

        # 核密度曲线（扩展y轴范围以确保完整显示）
        kde = stats.gaussian_kde(data)
        y_range = np.linspace(y_min, y_max, 300)  # 增加采样点数以获得更平滑的曲线
        kde_values = kde(y_range)
        ax.plot(kde_values, y_range, color=linecolor, linewidth=2, zorder=12, alpha=0.7)

    ax.set_title(title, fontsize=13, pad=10)

# 上方：言语评分归一化密度分布
plot_horizontal_density(ax_top, df['Verbal'], df['Verbal_jitter'], 
                        'GCS Verbal Score Density Distribution', 'gray', 'gray')

# 左侧：运动评分归一化密度分布（朝左+刻度在右侧+数值标注)
plot_vertical_density(
    ax_left, df['Motor'],
    'GCS Motor Score Density Distribution', 'gray',
    is_left=True, linecolor='gray'
)

# 右侧：睁眼评分归一化密度分布（朝右+数值标注）
plot_vertical_density(
    ax_right, df['Eye'],
    'GCS Eye Opening Score Density Distribution', 'gray',
    is_left=False, linecolor='gray'
)

# -------------------------- 6. Final Style Adjustments --------------------------
ax_right.set_yticks([])
ax_top.set_xticks(np.arange(1, 6))
ax_main.set_xticks(np.arange(-2, 7))
ax_main.set_yticks(np.arange(-2, 8))
ax_main_twin.set_yticks(np.arange(1, 5))

# plt.show()
# 设置保存图片的dpi
plt.tight_layout()
plt.savefig(r'exp_image\\20251230_gcs_density_distribution.jpg', dpi=500, bbox_inches='tight')
