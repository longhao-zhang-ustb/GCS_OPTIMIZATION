import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker

# -------------------------- 1. Prepare 5 rows and 13 columns of data. --------------------------
# Method 1: Manual Definition (Example: 5 rows, 13 columns; Row = Level of Consciousness, Column = GCS 3-15)
data = np.array([
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 12, 5, 420, 9852],
    [0, 2, 0, 1, 35, 284, 2279, 2155, 2448, 2908, 5402, 26401, 349],
    [39, 629, 3218, 6038, 6673, 1695, 400, 22, 2, 8, 0, 1, 0],
    [176, 8714, 9549, 2088, 917, 115, 0, 0, 0, 0, 0, 0, 0],
    [2695, 25, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

# -------------------------- 2. Define the x-axis and y-axis labels --------------------------
x_labels = list(range(3, 16))  # Horizontal axis: GCS score 3–15 (13 points total)
y_labels = ['SO', 'HS', 'LC', 'MC', 'SC']  # Vertical axis: 5 levels of consciousness

# -------------------------- 3. Global Font Settings: Times New Roman (applies to all images) --------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 14,
    'axes.unicode_minus': False
})

def format_num(x):
    """Format the number as an integer with a thousand separator, leaving zeros unchanged."""
    return f"{int(x):,.0f}"

# Generate labeled text with thousand separators (5 rows, 13 columns, corresponding one-to-one with the data)
annot_text = np.vectorize(format_num)(data)

# -------------------------- 4. Core: Plotting a Seaborn heatmap (without any additional logic) --------------------------
plt.figure(figsize=(16, 5))  # Canvas size (width 16, height 5, accommodating 13 columns horizontally)
ax = sns.heatmap(
    data,
    annot=True,        
    fmt=',',          
    cmap='Blues',     
    linewidths=0.2,    # Width of cubicle partition lines
    linecolor="#1A1212",# Separator line color (light gray, clear yet unobtrusive)
    xticklabels=x_labels,  # Horizontal Axis Scale Label (GCS3-15)
    yticklabels=y_labels,  # Vertical Axis Label (5 Levels of Consciousness)
    cbar_kws={'label': 'Count', 'shrink': 0.8, 'format': mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")}  # Color Bar: Label + Length Adjustment
)

# -------------------------- 5. Simple annotations --------------------------
plt.xlabel('GCS Total Score', fontsize=14, labelpad=10)
plt.ylabel('Arousal Level', fontsize=14, labelpad=10)
plt.xticks(rotation=0)  # Horizontal axis scale displayed horizontally
plt.yticks(rotation=0)  # Vertical axis scale displayed horizontally

# -------------------------- 6. Display/Save Image --------------------------
plt.tight_layout()  # Prevent Tag Clipping
# plt.savefig('heatmap_GCS_AROUSAL.jpg', dpi=1000, bbox_inches='tight', facecolor='white')  # Save as a 300dpi high-resolution image
plt.show()
