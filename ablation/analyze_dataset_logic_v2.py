import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set global aesthetic params
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs('ablation/current/analysis', exist_ok=True)

df = pd.read_excel('ablation/current/datasets/hea-composition-performance.xlsx')

# 1. Properties mapping
mapping = {
    '室温延伸率(%)': 'RT_EL',
    '高温延伸率(%)': 'HT_EL',
    '室温屈服强度(MPa)': 'RT_YS',
    '高温屈服强度(MPa)': 'HT_YS',
    '室温抗拉强度(MPa)': 'RT_UTS',
    '高温抗拉强度(MPa)': 'HT_UTS'
}

# 2. Current R2 scores (from previous experimental data)
r2_scores = {
    'RT_EL': 0.838, 'HT_EL': 0.801,
    'RT_YS': 0.451, 'HT_YS': 0.384,
    'RT_UTS': 0.333, 'HT_UTS': 0.163
}

# 3. Data Collection
data_meta = []
for zh, en in mapping.items():
    valid_n = pd.to_numeric(df[zh], errors='coerce').dropna().shape[0]
    data_meta.append({
        'Property': en,
        'Samples': valid_n,
        'R2': r2_scores[en]
    })

res = pd.DataFrame(data_meta)
# Sort by property categories for cleaner logic (RT followed by HT)
res = res.sort_values('Property', ascending=True)

# 4. Starting the Dual-Panel Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8), sharey=True)
plt.subplots_adjust(wspace=0.15) 

y_pos = np.arange(len(res))
height = 0.65

# --- Plot 1: DATA SUPPLY (How much the model saw) ---
# Color logic: RT properties in Grey, HT in Bronze, HT_UTS in Red
colors = []
for p in res['Property']:
    if p == 'HT_UTS': colors.append('#E74C3C') # Warning Red
    elif 'HT' in p: colors.append('#D35400')  # Bronze/Orange
    else: colors.append('#34495E')            # Dark Grey

bars1 = ax1.barh(y_pos, res['Samples'], height, color=colors, alpha=0.9, edgecolor='white')
ax1.set_xlabel('Sample Count (Input Knowledge)', fontsize=12, fontweight='bold')
ax1.set_xlim(0, 80)
ax1.invert_xaxis() 
ax1.set_title('Data Supply (Resource)', fontsize=14, pad=20, fontweight='black', color='#2C3E50')

# Labeling Sample Counts
for i, v in enumerate(res['Samples']):
    ax1.text(v + 2, i, f"N={v}", va='center', ha='right', fontsize=11, fontweight='bold', color='black')

# --- Plot 2: PERFORMANCE (R2 Prediction Results) ---
bars2 = ax2.barh(y_pos, res['R2'], height, color=colors, alpha=0.9, edgecolor='white')
ax2.set_xlabel('R-Squared Score (Accuracy)', fontsize=12, fontweight='bold')
ax2.set_xlim(0, 1.0)
ax2.set_title('Model Performance (Output)', fontsize=14, pad=20, fontweight='black', color='#2C3E50')

# Labeling R2 values
for i, v in enumerate(res['R2']):
    ax2.text(v + 0.02, i, f"R² = {v:.3f}", va='center', ha='left', fontsize=11, fontweight='bold', color='black')

# --- Shared Y-Axis Labeling (The Center) ---
ax2.set_yticks(y_pos)
ax2.set_yticklabels(res['Property'], fontsize=12, fontweight='bold', color='#2C3E50')

# --- Title and Logical Annotations ---
plt.suptitle('THE ROOT CAUSE: WHY HT_UTS PERFORMANCE IS DISAPPOINTING?', fontsize=18, fontweight='black', y=1.02)

# Specific Annotation for the failure point
ax1.annotate('Significant Data Shortage!', xy=(32, 2), xytext=(55, 4),
             arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5), 
             fontsize=11, fontweight='bold', color='#E74C3C', ha='center')

plt.tight_layout()

# Save final logic chart
save_path = 'ablation/current/analysis/root_cause_logic_dual.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Created Root Cause Logic Chart: {save_path}")
