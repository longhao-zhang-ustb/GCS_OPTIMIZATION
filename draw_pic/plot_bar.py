import matplotlib.pyplot as plt
import numpy as np

# -------------------------- Global configuration --------------------------
plt.rcParams.update({
    'font.sans-serif': ['Times New Roman'],
    'axes.unicode_minus': False,
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.linewidth': 1.2, 
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'grid.linewidth': 0.8,
})

TOP_JOURNAL_COLORS = [
    '#4E79A7', '#59a14F',  '#4E79A7', '#59a14F',  '#4E79A7', '#59a14F'
]
GROUPS = ['Reference', 'Group1', 'Group2', 'Group3', 'Group4', 'Group5', 'Group6']
X = np.arange(len(GROUPS))
BAR_WIDTH = 0.55

# -------------------------- Data Preparation --------------------------
# Format: [Subfigure 1 data, Subfigure 2 data, ..., Subfigure 6 data], where each subfigure data set = (f1 mean list, f1 standard deviation list)
# Each list must contain nine elements (corresponding to nine groups), with the order strictly matching.
plot_datas = [
    # Subfigure 1
    ([92.64, 53.19, 62.80, 31.76, 89.67, 56.83, 67.17], [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]),
    # Subfigure 2
    ([93.23, 53.81, 61.80, 25.12, 91.31, 61.04, 63.78], [0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]),
    # Subfigure 3
    ([93.12, 53.19, 62.80, 37.03, 89.67, 57.25, 68.43], [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]),
    # Subfigure 4
    ([93.81, 53.81, 61.80, 32.77, 91.28, 61.46, 64.93], [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]),
    # Subfigure 5
    ([93.46, 53.19, 62.80, 39.43, 89.67, 57.45, 68.42], [0.30, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]),
    # Subfigure 6
    ([93.67, 53.81, 61.80, 32.38, 91.31, 61.90, 64.14], [0.01,0.00, 0.00, 0.00, 0.00, 0.00, 0.00]),
]
# Titles of 6 Subfigures
plot_titles = ['Exp1', 'Exp2', 'Exp3', 'Exp4', 'Exp5', 'Exp6']

# Create a 3-row, 2-column subplot; tight_layout automatically adjusts spacing between subplots.
fig, axes = plt.subplots(3, 2, figsize=(15, 12), tight_layout=False)
# Convert the 2D axes to 1D for convenient iteration through the six subgraphs.
axes = axes.flatten()

# Draw 6 subplots in a loop
for idx, (ax, (means, stds), title) in enumerate(zip(axes, plot_datas, plot_titles)):
    color = TOP_JOURNAL_COLORS[idx]
    bars = ax.bar(X, means, BAR_WIDTH, color=color, edgecolor='black', alpha=0.85, linewidth=0.8)
    
    # Add error bars: mean ± standard deviation. Capsize controls the length of the horizontal lines at both ends of the error bars.
    ax.errorbar(X, means, yerr=stds, fmt='none', ecolor='#333333', 
                linewidth=1.2, capsize=4, elinewidth=1.2)
    
    # ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    ax.set_xticks(X)
    ax.set_xticklabels(GROUPS, fontsize=12)
    ax.set_xlabel('Groups', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score (Mean ± SD, %)', fontsize=12, fontweight='bold')
    ax.set_ylim(20, 100)
    ax.set_yticks(np.arange(20, 100, 30))
    ax.tick_params(axis='y', labelsize=12)
    
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        # ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                # f'{mean:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        # Mean ± standard deviation, rounded to two decimal places
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.008,
                f'{mean:.2f}±{std:.2f}', 
                ha='center', va='bottom', fontsize=10)
    
    ax.grid(axis='y', linestyle='--', color='#dddddd', alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# fig.supxlabel('Groups', fontsize=14, fontweight='bold', y=0.02)
# fig.supylabel('F1 Score (Mean ± SD, %)', fontsize=14, fontweight='bold', x=0.01)

# Save as PNG
plt.savefig('result.png', dpi=600, bbox_inches='tight', facecolor='white')
plt.show()
