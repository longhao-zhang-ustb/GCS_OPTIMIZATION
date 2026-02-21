import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import pandas as pd

plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

# ---------------------- 1. Unify data (maintain consistency across all subgraph data) ----------------------
dtBaseline = pd.read_csv(r'zigong_data\\dtBaseline.csv')
dtProcessed = pd.read_csv(r'zigong_data\\20251227_final_processed_data.csv')
dtProcessed = dtProcessed.drop_duplicates(subset='INP_NO', keep='first')

# Identify the age of male and female patients and their corresponding levels of consciousness.
male_age = dtProcessed[dtProcessed['SEX'] == 'Male']['Age'].values
male_levels = dtProcessed[dtProcessed['SEX'] == 'Male']['consciousness'].values
female_age = dtProcessed[dtProcessed['SEX'] == 'Female']['Age'].values
female_levels = dtProcessed[dtProcessed['SEX'] == 'Female']['consciousness'].values

# Number of patients printed
# print(f"Number of male patients: {len(male_age)}")
# print(f"Number of female patients: {len(female_age)}")

# dtProcessed[['INP_NO', 'Age', 'SEX', 'consciousness']].to_csv(r'zigong_data\\dtProcessed_age_sex_consciousness.csv', index=False)

# Consciousness Level Label Mapping
consciousness_levels = ["SO", "HS", "LC", "MC", "SC"]
level_to_num = {level: i for i, level in enumerate(consciousness_levels)}
# Convert male_levels and female_levels to strings based on consciousness_levels
male_levels = [consciousness_levels[int(level)] for level in male_levels]
female_levels = [consciousness_levels[int(level)] for level in female_levels]

# Convert consciousness levels to numerical values (Key: Convert text labels to numbers to add jitter)
male_levels_num = np.array([level_to_num[level] for level in male_levels])
female_levels_num = np.array([level_to_num[level] for level in female_levels])

# Add Y-axis jitter (using a normal distribution random disturbance; 0.15 is the adjustable jitter amplitude)
JA = 0.1  # Control the amount of jitter; the higher the value, the more pronounced the dispersion.
male_levels_jittered = male_levels_num + np.random.normal(0, JA, size=len(male_levels_num))
female_levels_jittered = female_levels_num + np.random.normal(0, JA, size=len(female_levels_num))

# Merge data for edge map
all_age = np.concatenate([male_age, female_age])
all_levels = np.concatenate([male_levels_num, female_levels_num])  # Using a digital level

# Age Group Configuration
age_bins = np.arange(0, 140, 10)
age_labels = [f'{i}-{i+10}' for i in age_bins[:-1]]
male_counts, _ = np.histogram(male_age, bins=age_bins)
female_counts, _ = np.histogram(female_age, bins=age_bins)

# ---------------------- 2. Canvas Layout ----------------------
fig = plt.figure(figsize=(8, 8))
gs = fig.add_gridspec(
    nrows=5, ncols=5,
    hspace=0.5, wspace=0.1,
    height_ratios=[2.5, 0.2, 6, 0.2, 0.1],
    width_ratios=[0.1, 5, 0.05, 1, 0.1]
)

ax_age_gender = fig.add_subplot(gs[0, 1])          # Bar Chart Above
ax_scatter = fig.add_subplot(gs[2, 1])             # Scatter Plot in the Middle
ax_hist_y = fig.add_subplot(gs[2, 3])              # Right-side histogram

# ---------------------- 3. Primary Scatter Plot (Using Jittered Y Values) ----------------------
ax_scatter.scatter(
    male_age, male_levels_jittered,  # Male: Use the y-value after jittering
    color='blue', marker='o', label='Male', s=5, edgecolors='blue'
)
ax_scatter.scatter(
    female_age, female_levels_jittered,  # Female: Use the y-value after jittering
    color='red', marker='s', label='Female', s=5, edgecolors='red', linestyle='--'
)

ax_scatter.set_xticks(np.arange(0, 130, 10))
ax_scatter.set_xticklabels(np.arange(0, 130, 10), fontsize=8)
ax_scatter.set_yticks(range(len(consciousness_levels)))
ax_scatter.set_yticklabels(consciousness_levels, fontsize=8, rotation=15)
ax_scatter.set_xlabel('Age (years)', fontsize=8)
ax_scatter.set_ylabel('Arousal Level', fontsize=8)
ax_scatter.legend(fontsize=8, loc='upper right')
ax_scatter.grid(axis='y', linestyle='--', alpha=0.7)

# ---------------------- 4. Age Distribution Bar Chart Above (Remain unchanged) ----------------------
x = np.arange(len(age_labels))
width = 0.3
ax_age_gender.bar(x - width/2, male_counts, width, label='Male', color='blue', alpha=0.7, edgecolor='blue')
ax_age_gender.bar(x + width/2, female_counts, width, label='Female', color='red', alpha=0.7, edgecolor='red', linestyle='--')

kde_male_age = gaussian_kde(male_age)
kde_female_age = gaussian_kde(female_age)
x_range_hist = np.linspace(0, 130, 500)
kde_male_scaled = kde_male_age(x_range_hist) * len(male_age) * 10
kde_female_scaled = kde_female_age(x_range_hist) * len(female_age) * 10
ax_age_gender.plot(x_range_hist/10, kde_male_scaled, color='darkblue', linestyle='-', linewidth=2, alpha=0.5)
ax_age_gender.plot(x_range_hist/10, kde_female_scaled, color='darkred', linestyle='--', linewidth=2, alpha=0.5)

ax_age_gender.set_xlabel('Age Interval (years)', fontsize=8)
ax_age_gender.set_ylabel('Number of Patients', fontsize=8)
ax_age_gender.set_title('Age Distribution (10-year Intervals, by Gender)', fontsize=8, pad=10)
ax_age_gender.set_xticks(x)
ax_age_gender.set_xticklabels(age_labels, fontsize=8, rotation=15, ha='right')
ax_age_gender.set_yticklabels([f'{int(y)}' for y in ax_age_gender.get_yticks()], fontsize=8)
ax_age_gender.legend(fontsize=8, loc='upper right')
ax_age_gender.grid(axis='y', linestyle='--', alpha=0.5)
ax_age_gender.set_xlim(-0.5, len(age_labels)-0.5)

for i, (m, f) in enumerate(zip(male_counts, female_counts)):
    if m > 0:
        ax_age_gender.text(i - width/2, m + 40, str(m), ha='center', va='top', fontsize=8)
    if f > 0:
        ax_age_gender.text(i + width/2, f + 40, str(f), ha='center', va='top', fontsize=8)

# ---------------------- 5. Right-side consciousness level distribution histogram (remains unchanged) ----------------------
all_y_counts = np.bincount(all_levels, minlength=len(consciousness_levels))
all_y_density = all_y_counts / len(all_levels)
y_bins = range(len(consciousness_levels))

ax_hist_y.barh(y_bins, all_y_density, color='#444444', alpha=0.6)
kde_all_y = gaussian_kde(all_levels)
y_range = np.linspace(0, len(consciousness_levels)-1, 100)
ax_hist_y.plot(kde_all_y(y_range), y_range, color='#222222', linestyle='-', linewidth=2, alpha=0.5)

for i, v in enumerate(all_y_density):
    ax_hist_y.text(v + 0.05, i, f'{v:.2f}', fontsize=8, va='center')

x_ticks = ax_hist_y.get_xticks()
ax_hist_y.set_xticklabels([f'{x}' for x in x_ticks], fontsize=8)
ax_hist_y.set_yticks([])
ax_hist_y.set_xlabel('Density', fontsize=8)
ax_hist_y.set_title('Overall Arousal Level Distribution', fontsize=8, pad=10)
ax_hist_y.grid(axis='x', linestyle='--', alpha=0.5)

plt.show()
