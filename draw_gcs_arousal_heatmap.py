import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker

# -------------------------- 1. 准备5行13列数据（核心：替换为你的真实数据即可） --------------------------
# 方式1：手动定义（示例：5行13列，行=意识水平、列=GCS3-15）
data = np.array([
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 12, 5, 420, 9852],  # 第1行
    [0, 2, 0, 1, 35, 284, 2279, 2155, 2448, 2908, 5402, 26401, 349],  # 第2行
    [39, 629, 3218, 6038, 6673, 1695, 400, 22, 2, 8, 0, 1, 0],  # 第3行
    [176, 8714, 9549, 2088, 917, 115, 0, 0, 0, 0, 0, 0, 0],  # 第4行
    [2695, 25, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0] # 第5行
])

# -------------------------- 2. 定义横纵轴标签（匹配你的需求：GCS3-15、5种意识水平英文） --------------------------
x_labels = list(range(3, 16))  # 横轴：GCS评分3-15（共13个）
y_labels = ['SO', 'HS', 'LC', 'MC', 'SC']  # 纵轴：5种意识水平

# -------------------------- 3. 全局字体设置：Times New Roman（全图生效） --------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 14,
    'axes.unicode_minus': False
})

def format_num(x):
    """将数值格式化为带千位分隔符的整数，0保持不变"""
    return f"{int(x):,.0f}"

# 生成带千位分隔符的标注文本（5行13列，与数据一一对应）
annot_text = np.vectorize(format_num)(data)

# -------------------------- 4. 核心：绘制Seaborn热力图（无任何额外逻辑） --------------------------
plt.figure(figsize=(16, 5))  # 画布大小（宽16，高5，适配13列横轴）
ax = sns.heatmap(
    data,
    annot=True,        # 格子内显示数值
    fmt=',',           # 数值格式：整数（无需小数）
    cmap='Blues',     # 配色（医疗/论文适配，可换Blues/Greens/RdYlGn）
    linewidths=0.2,    # 格子间分隔线宽度
    linecolor="#1A1212",# 分隔线颜色（浅灰，清晰不突兀）
    xticklabels=x_labels,  # 横轴刻度标签（GCS3-15）
    yticklabels=y_labels,  # 纵轴刻度标签（5种意识水平）
    cbar_kws={'label': 'Count', 'shrink': 0.8, 'format': mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")}  # 颜色条：标签+长度适配
)

# -------------------------- 5. 简单标注（可按需修改/删除） --------------------------
plt.xlabel('GCS Total Score', fontsize=14, labelpad=10)
plt.ylabel('Arousal Level', fontsize=14, labelpad=10)
plt.xticks(rotation=0)  # 横轴刻度水平显示
plt.yticks(rotation=0)  # 纵轴刻度水平显示

# -------------------------- 6. 显示/保存图片 --------------------------
plt.tight_layout()  # 防止标签裁剪
# plt.savefig('heatmap_GCS_AROUSAL.jpg', dpi=1000, bbox_inches='tight', facecolor='white')  # 保存300dpi高清图
plt.show()
