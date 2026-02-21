import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ====================== 1. Global configuration ======================
top_journal_colors = [
    '#1f77b4',  
    '#ff7f0e',  
    '#2ca02c',  
    '#d62728',  
    '#9467bd',  
    '#8c564b',  
    '#e377c2'   
]

ref_line_color = "#4B545F"  
ref_line_width = 2.5        
ref_line_style = (0, (3, 1))

# Global font configuration: Times New Roman, maintain enlarged font
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12  # Base Font Size

# ====================== 2. Data preparation ======================
feature_labels = ['All features', 'Group1', 'Group2', 'Group3', 'Group4', 'Group5', 'Group6']

f1_data = [
    [0.82, 0.80, 0.83, 0.79, 0.81],  # Group1
    [0.85, 0.87, 0.84, 0.86, 0.85],  # Group2
    [0.83, 0.81, 0.82, 0.80, 0.82],  # Group3
    [0.90, 0.92, 0.89, 0.91, 0.90],  # Group4
    [0.88, 0.89, 0.87, 0.88, 0.89],  # Group5
    [0.92, 0.93, 0.91, 0.94, 0.92],  # Group6
    [0.86, 0.85, 0.87, 0.86, 0.85]   # Group7
]

# Predefine reference group indices
ref_group_idx = 0
ref_group_x = ref_group_idx + 1  # X-axis position corresponding to Group 1
ref_group_label = feature_labels[ref_group_idx]

data_matrix = f1_data
group_means = [np.mean(data) for data in data_matrix]  # Mean of each group
group_stds = [np.std(data, ddof=1) for data in data_matrix]  # Standard deviation of each group of samples
flattened_data = [val for sublist in data_matrix for val in sublist]  # Flatten data

# ====================== 3. Statistical Test (Group 1 as reference, general significance results) ======================
# Generate group labels (for statistical testing)
group_labels_stat = []
for idx in range(len(feature_labels)):
    group_labels_stat.extend([idx] * len(data_matrix[idx]))

# One-way ANOVA
f_stat, anova_p = stats.f_oneway(*data_matrix)
print(f"One-way ANOVA Result: F-statistic={f_stat.round(4)}, p-value={anova_p.round(6)}")
if anova_p < 0.05:
    print("Conclusion: Significant overall differences among groups, performing Tukey HSD test.")
else:
    print("Conclusion: No significant overall differences among groups.")
print("-" * 60)

# Tukey's post hoc HSD test
tukey_result = pairwise_tukeyhsd(np.array(flattened_data), np.array(group_labels_stat), alpha=0.05)
print(f"Tukey HSD Post-hoc Test Result:")
print(tukey_result)

# Organize significant results: (Group Index 1, Group Index 2) -> (p-value, Significant)
sig_results = {}
for row in tukey_result.summary().data[1:]:
    g1 = int(row[0])
    g2 = int(row[1])
    p_val = float(row[3])
    is_sig = row[4] == 'reject'
    sig_results[(g1, g2)] = (p_val, is_sig)
    sig_results[(g2, g1)] = (p_val, is_sig)

print(f"\nReference Group: {ref_group_label} (Mean F1={group_means[ref_group_idx].round(4)})")

# ====================== 4. Significance symbol mapping function ======================
def get_sig_symbol(p_val):
    if p_val < 0.001:
        return '***'
    elif p_val < 0.01:
        return '**'
    elif p_val < 0.05:
        return '*'
    else:
        return 'ns'

# ====================== 5. Drawing a horizontal, flattened subgraph ======================
# Flat layout
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 4), sharey=True)
axes = [ax1, ax2]
split_ratios = ["7:3 (Train:Test)", "6:4 (Train:Test)"] 

# Uniform parameter annotation
y_label_offset = 0.012
y_mean_std_offset = 0.018
y_min_global = min(flattened_data) - 0.08
y_max_global = max(flattened_data) + 0.08

for idx, (ax, split_ratio) in enumerate(zip(axes, split_ratios)):
    
    ax.axvline(
        x=ref_group_x,
        ymin=0,
        ymax=1,
        color=ref_line_color,
        linewidth=ref_line_width,
        alpha=0.9,  
        linestyle=ref_line_style,
        zorder=2
    )

    # Step2: plot a box plot
    box_plot = ax.boxplot(
        data_matrix,
        patch_artist=True,
        labels=feature_labels,
        medianprops={'color': 'white', 'linewidth': 2}, 
        whiskerprops={'color': 'black', 'linewidth': 1.2},
        capprops={'color': 'black', 'linewidth': 1.2},
        flierprops={'marker': 'None'},
        widths=0.6
    )

    for i, patch in enumerate(box_plot['boxes']):
        patch.set_facecolor(top_journal_colors[i])
        patch.set_alpha(0.7)  # Fill transparency

    # Step3: Overlay Transparent scatter points
    for group_idx, (data, color) in enumerate(zip(data_matrix, top_journal_colors)):
        x_jitter = np.random.normal(group_idx + 1, 0.04, size=len(data))
        ax.scatter(
            x_jitter, data,
            color=color,
            s=40,
            alpha=0.6,
            edgecolors='black',
            linewidth=0.8,
            zorder=3  # Scatter points above the vertical line are not obscured
        )

    # legend
    ax.plot([], [], color=ref_line_color, linewidth=ref_line_width, linestyle=ref_line_style,
            alpha=0.9, label=f'Reference Group ({ref_group_label})')
    # Set a borderless legend
    ax.legend(loc='upper right', fontsize=12, framealpha=0)

    # Significance symbols are labeled above each group
    for group_idx in range(len(feature_labels)):
        if group_idx == ref_group_idx:
            continue
        p_val, is_sig = sig_results[(ref_group_idx, group_idx)]
        sig_symbol = get_sig_symbol(p_val)
        current_group_max = max(data_matrix[group_idx])
        label_y_pos = current_group_max + y_label_offset
        ax.text(group_idx + 1, label_y_pos, sig_symbol, ha='center', va='bottom',
                fontsize=14, fontweight='bold', zorder=4)

    # The mean ± standard deviation is indicated below each group
    for group_idx in range(len(feature_labels)):
        current_group_min = min(data_matrix[group_idx])
        label_y_pos = current_group_min - y_mean_std_offset
        mean_std_text = f"{group_means[group_idx]:.3f} ± {group_stds[group_idx]:.3f}"
        ax.text(group_idx + 1, label_y_pos, mean_std_text, ha='center', va='top',
                fontsize=13, color='black', zorder=4)

    # Subfigure captions and X-axis labels
    # ax.set_title(split_ratio, fontsize=14, fontweight='bold', pad=8)
    ax.set_xlabel('Groups', fontsize=12, fontweight='bold', labelpad=12)

    # X-axis scale labels
    ax.set_xticks(range(1, len(feature_labels)+1))
    ax.set_xticklabels(feature_labels, fontsize=13)

    # Y-axis range and gridlines
    ax.set_ylim(y_min_global, y_max_global)
    ax.grid(axis='y', alpha=0.2, linestyle='-', linewidth=0.8, zorder=1)  # Grid lines below vertical lines
    ax.set_axisbelow(True)

# ====================== 6. Global configuration ======================
# Y-axis label for the left subplot
ax1.set_ylabel('Macro Average F1-score', fontsize=12, fontweight='bold', labelpad=10)
# Y-axis scale labels
y_ticks = ax1.get_yticks()
plt.yticks(ticks=y_ticks, fontsize=10)

# Global description of the left subfigure
sig_annotation = "Note: *p<0.05, **p<0.01, ***p<0.001, ns=not significant (vs All features)"
ax1.text(0.02, 0.95, sig_annotation, ha='left', va='top',
         fontsize=12, style='italic', transform=ax1.transAxes,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7))

# Adjust subgraph spacing
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.subplots_adjust(wspace=0.08)

# ====================== 7. Display image ======================
plt.show()
