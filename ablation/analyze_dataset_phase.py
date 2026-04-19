import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set premium academic style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs('ablation/current/analysis', exist_ok=True)

# Load data
df = pd.read_excel('ablation/current/datasets/hea-composition-performance.xlsx')

# Clean columns
df['HT_UTS'] = pd.to_numeric(df['高温抗拉强度(MPa)'], errors='coerce')
df['Temperature'] = pd.to_numeric(df['高温温度(°C)'], errors='coerce')
df['Phase'] = df['相结构'].fillna('Unknown').astype(str)

# Filter valid data for HT_UTS
plot_df = df[['Temperature', 'HT_UTS', 'Phase']].dropna(subset=['Temperature', 'HT_UTS'])

# Simplify Phase names if they are too long (optional but helps readability)
def simplify_phase(p):
    if 'FCC' in p and 'L1' in p: return 'FCC+L12'
    if 'FCC' in p and 'BCC' in p: return 'Dual Phase'
    if 'FCC' in p: return 'FCC'
    if 'BCC' in p: return 'BCC'
    return 'Other'

plot_df['Phase_Simple'] = plot_df['Phase'].apply(simplify_phase)

# Create the Scatter Plot
plt.figure(figsize=(12, 7))

# Use a distinct color palette for different phases
# BCC typically stronger at high temp, FCC more ductile
palette = {'BCC': '#E74C3C', 'FCC': '#3498DB', 'FCC+L12': '#2ECC71', 'Dual Phase': '#9B59B6', 'Other': '#95A5A6'}

sns.scatterplot(data=plot_df, 
                x='Temperature', 
                y='HT_UTS', 
                hue='Phase_Simple', 
                style='Phase_Simple',
                palette=palette,
                s=120, 
                alpha=0.8, 
                edgecolor='black', 
                linewidth=1)

# Add titles and labels
plt.title('HT_UTS vs Temperature: Color-Coded by Phase Structure', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Test Temperature (°C)', fontsize=13, fontweight='bold')
plt.ylabel('Ultimate Tensile Strength (MPa)', fontsize=13, fontweight='bold')

# Customize legend
plt.legend(title='Crystal Structure (Phase)', title_fontsize='11', fontsize='10', loc='upper right', frameon=True)

plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Save
save_path = 'ablation/current/analysis/ht_uts_vs_temp_phase.png'
plt.savefig(save_path, dpi=300)
plt.close()

print(f"Created redesigned Phase-based scatter plot at: {save_path}")
