import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from matplotlib.ticker import StrMethodFormatter, FuncFormatter

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

levels = ["SO", "HS", "LC", "MC", "SC"]
academic_colors = ["#20c70e", "#88af06", "#ff0e0e", "#273ed6", "#3d0a6d"]
level_color = dict(zip(levels, academic_colors))

# ---------------------- 1. Unify data (ensure consistency across all subgraphs) ----------------------
dtProcessed = pd.read_csv(r'zigong_data\\20251227_final_processed_data.csv')
# # # First, remove the -1 entries for language, motion, and open_one's_eyes from dtProcessed.
# dtProcessed = dtProcessed[dtProcessed['language'] != -1]
# dtProcessed = dtProcessed[dtProcessed['motion'] != -1]
# dtProcessed = dtProcessed[dtProcessed['open_one\'s_eyes'] != -1]
# # Calculate the total score for language, motion, and open one's eyes.
# dtProcessed['GCS_Total'] = dtProcessed['language'] + dtProcessed['motion'] + dtProcessed['open_one\'s_eyes']
# # Convert the total score to an integer
# dtProcessed['GCS_Total'] = dtProcessed['GCS_Total'].astype(int)
# # Statistically analyze the correlation between GCS_Total and consciousness
# gcs_consciousness = dtProcessed.groupby('consciousness')['GCS_Total'].value_counts().unstack(fill_value=0)
# plt.figure(figsize=(10, 4))
# # Custom annotation content (add thousand separators)
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
# Iterate through each entry in `dtProcessed` and organize the data in the format `{‘Verbal’:1, ‘Motor’:1, ‘Eye’:1, ‘Consciousness’:'Sober'}`.
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
    """Add slight jitter to prevent scattered points from overlapping."""
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
ax_top = fig.add_subplot(gs[0, 1])    # Top: Verbal Score Distribution
ax_right = fig.add_subplot(gs[1, 2])  # Right Side: Open-Eye Score Distribution
ax_left = fig.add_subplot(gs[1, 0])   # Left: Motor Score Distribution
ax_main = fig.add_subplot(gs[1, 1])   # Center: Scatter Plot

# -------------------------- 4. Main Scatter Plot --------------------------
ax_main.set_xlabel('GCS Verbal Score', fontsize=13)
ax_main.set_ylabel('GCS Motor Score', fontsize=13)
ax_main.set_xlim(-2, 6)
ax_main.set_ylim(-2, 7)
ax_main.tick_params(axis='x', labelsize=13)
ax_main.tick_params(axis='y', labelsize=13)
ax_main.grid(alpha=0.3, linestyle='--', color='#cccccc')

ax_main_twin = ax_main.twinx()
ax_main_twin.set_ylabel('GCS Eye Opening Score', fontsize=13, rotation=-90, labelpad=15)
ax_main_twin.set_ylim(0.5, 4.5)
ax_main_twin.tick_params(axis='y', labelsize=13)

# Plot scatter points (fixed size)
for level in levels:
    subset = df[df['Consciousness'] == level]
    ax_main.scatter(
        subset['Verbal_jitter'], subset['Motor_jitter'],
        s=25,  # Fixed size, unlink from Eye score
        c=level_color[level], alpha=0.5,
        label=level, edgecolors='white', linewidth=0.6
    )
    ax_main_twin.scatter(
        subset['Verbal_jitter'], subset['Eye_jitter'],
        s=25,  # Secondary axis labels are also fixed in size.
        c=level_color[level], alpha=0.5, edgecolors='white', linewidth=0.6
    )

# Add a red dashed line at the point where the y-axis equals 0 on the scatter plot.
ax_main.axhline(0, color='gray', linestyle=':', linewidth=1.5, zorder=10)
# Add a red dashed line at x = 0 on the scatter plot.
ax_main.axvline(0, color='gray', linestyle=':', linewidth=1.5, zorder=10)
# The area to the left and above the two dotted lines is set to a different color.
ax_main.fill_betweenx(
    ax_main.get_ylim(), ax_main.get_xlim()[0], 0,
    color='gray', alpha=0.1, zorder=9
)
ax_main.fill_between(
    ax_main.get_xlim(), ax_main.get_ylim()[0], 0,
    color='gray', alpha=0.1, zorder=9
)

# ax_main.legend(loc='upper right', bbox_to_anchor=(1, 1), frameon=True, fancybox=True, shadow=True)
# Use the ncol parameter to arrange the legend in a single row, and position it at the bottom-right corner using loc and bbox_to_anchor.
ax_main.legend(loc='lower right', ncol=len(levels), 
               frameon=False, fontsize=13)

def plot_horizontal_density(ax, data, data_jitter, title, color, linecolor):
    """Horizontal Normalized Density Histogram (top)"""
    # Retrieve unique values and sort them (including -1 values)
    unique_values = np.sort(np.unique(data))
    unique_values = np.insert(unique_values, 1, 0)
    
    # Calculate bins: Centered on each unique value, with bar width set to 1 (matching the scatter plot coordinate step size)
    bin_edges = np.array([val - 0.5 for val in unique_values] + [unique_values[-1] + 0.5])
    # Force the bin width to 1, corresponding one-to-one with the x-axis tick marks of the scatter plot.
    bins = bin_edges
    
    # Compute normalized histogram (density=True)
    counts, bins, patches = ax.hist(
        data_jitter, bins=bins,  
        density=True,
        color=color, alpha=0.6, edgecolor='gray', linewidth=1,
        width=1.0
    )
    
    # Labeled values (normalized density values)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])  # Center position of the bin (exactly aligned with the x-axis of the scatter plot)
    for i, (patch, count) in enumerate(zip(patches, counts)):
        # Label the values above the columns (to two decimal places).
        ax.text(
            bin_centers[i], count + 0.01, f'{count:.2f}',
            ha='center', va='bottom', fontsize=13
        )
    
    # Plot a kernel density curve (normalized)
    if len(data) > 1:  # Ensure there are sufficient data points to calculate the KDE.
        kde = stats.gaussian_kde(data)
        x_range = np.linspace(-2, 6, 200)  # Match the scatter plot with xlim(-1.5,5.5)
        ax.plot(x_range, kde(x_range), color=linecolor, linewidth=2, alpha=0.7)

    
    # Style optimization: Move the title upward.
    ax.set_title(title, fontsize=13, pad=20)
    ax.set_ylabel('Density', fontsize=13)  # Change the vertical axis to density
    ax.set_xlim(-2, 6)
    # ax.set_xlim(-2, 6)  # Fully matches the x-axis range of the scatter plot
    ax.tick_params(
        axis='x',        # Only affects the X-axis
        which='both',    # Simultaneously control primary and secondary scales (to prevent skipping minor scale lines)
        bottom=False,    # Turn off bottom scale lines (dashed lines)
        labelbottom=False
    )
    ax.tick_params(axis='both', labelsize=13)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.grid(alpha=0.2, linestyle='--', color='#cccccc')
    
def plot_vertical_density(ax, data, title, color, is_left=True, linecolor='gray'):
    """Vertical Density Histogram"""
    ax.clear()
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    # if is_left:
    y_min, y_max = -1.5, 6.5  # Fully matches the Y-axis range of the main scatter plot
    # Calculate bins: Ensure the entire range is covered, with each bin having a width of 1.0.
    bins = np.arange(y_min, y_max+1, 1)
    bin_width = bins[1] - bins[0]  # Bin width is 1.0
    
    counts, bins = np.histogram(data, bins=bins, density=True)
    density_max = np.ceil(counts.max() * 10) / 10 if counts.size > 0 else 0.1
    density_min = 0
    y_centers = 0.5 * (bins[1:] + bins[:-1])

    # Generate scale
    density_max = density_max + 0.5 if is_left else density_max + 1
    xticks = np.linspace(density_min, density_max, 4)
    xtick_labels = [f'{x:.2f}' for x in xticks]

    if is_left:
        patches = ax.barh(
            y_centers, 
            width=counts,
            height=bin_width,  # Use the calculated fixed width
            color=color, alpha=0.7, edgecolor='gray', linewidth=1,
            left=np.zeros_like(counts)
        )

        ax.set_xlim(density_max, density_min)
        ax.set_ylim(y_min-0.5, y_max+0.5)  

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
            y_range = np.linspace(y_min - 0.5, y_max + 0.5, 300)  # Use ranges that match the main scatter plot
            kde_values = kde(y_range)
            ax.plot(kde_values, y_range, color=linecolor, linewidth=2, zorder=12, alpha=0.7)

    else:
        # Draw the column to the right (starting from the leftmost position 0 and drawing to the right).
        y_min = 0.5
        y_max = 4.5
        y_centers = [1, 2, 3, 4]
        counts = counts[2:6]
        patches = ax.barh(
            y_centers, 
            width=counts,  # Width as a density value
            height=bin_width,
            color=color, alpha=0.7, edgecolor='gray', linewidth=1,
            left=np.zeros_like(counts)  # Draw starting from the 0 position
        )

        ax.set_xlim(density_min, density_max + 0.1)
        ax.set_ylim(y_min, y_max)

        #  Scale displayed at the bottom
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, fontsize=13)
        ax.set_xlabel('Density', fontsize=13)
        ax.spines['bottom'].set_visible(True)
        ax.spines['right'].set_visible(False)

        # Density values marked inside the columns
        for patch, count in zip(patches, counts):
            # The text is positioned on the right side inside the column.
            text_x = count + 0.3
            text_y = patch.get_y() + patch.get_height()/2
            ax.text(text_x, text_y, f'{count:.2f}',
                    ha='right', va='center', fontsize=13, color='black')

        # Nuclear Density Curve (y-axis range expanded to ensure complete display)
        kde = stats.gaussian_kde(data)
        y_range = np.linspace(y_min, y_max, 300)  # Increase the number of sampling points to obtain a smoother curve.
        kde_values = kde(y_range)
        ax.plot(kde_values, y_range, color=linecolor, linewidth=2, zorder=12, alpha=0.7)

    ax.set_title(title, fontsize=13, pad=10)

# Top: Language Score Normalized Density Distribution
plot_horizontal_density(ax_top, df['Verbal'], df['Verbal_jitter'], 
                        'GCS Verbal Score Density Distribution', 'gray', 'gray')

# Left: Normalized density distribution of motion scores
plot_vertical_density(
    ax_left, df['Motor'],
    'GCS Motor Score Density Distribution', 'gray',
    is_left=True, linecolor='gray'
)

# Right side: Normalized density distribution of eye-open scores (rightward + numerical annotation)
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
# Set the DPI for saving images
plt.tight_layout()
plt.savefig(r'exp_image\\20251230_gcs_density_distribution.jpg', dpi=500, bbox_inches='tight')
