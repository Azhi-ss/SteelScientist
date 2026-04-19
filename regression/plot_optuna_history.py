#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def plot_optuna(prop):
    csv_file = f'./outputs/optuna/csvs/optuna_{prop}.csv'
    if not os.path.exists(csv_file):
        return

    df = pd.read_csv(csv_file)
    # 只要已完成 (COMPLETE) 且值不为NaN的 trial
    df = df[df['state'] == 'COMPLETE'].dropna(subset=['value']).copy()
    if df.empty: return

    # 绘制两张图: 1. 搜索历史趋势; 2. 学习率与层数的影响散点
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # --- 图 1: Optimization History ---
    # 算累计最佳值
    df['rolling_max'] = df['value'].cummax()
    
    axes[0].plot(df['number'], df['value'], marker='o', alpha=0.5, label='Actual Val_R2')
    axes[0].plot(df['number'], df['rolling_max'], marker='s', color='red', linewidth=2, label='Best Val_R2')
    axes[0].set_title(f'{prop} - Optuna Search History')
    axes[0].set_xlabel('Trial Number')
    axes[0].set_ylabel('Validation R2')
    axes[0].set_ylim(bottom=-0.1, top=0.8)  # ✂️ 切掉无用负分，强行放大 0~0.8 的有效提分区
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()

    # --- 图 2: Hyperparameter Scatter ---
    if 'params_lr' in df.columns and 'params_n_layers' in df.columns:
        scatter = axes[1].scatter(df['params_lr'], df['value'], 
                                  c=df['params_n_layers'], cmap='viridis', 
                                  s=100, alpha=0.8, edgecolors='k')
        axes[1].set_xscale('log')
        axes[1].set_ylim(bottom=-0.1, top=0.8) # 同理，限制右侧散点图的 Y 轴
        axes[1].set_title(f'{prop} - Learning Rate vs Performance')
        axes[1].set_xlabel('Learning Rate (log scale)')
        axes[1].set_ylabel('Validation R2')
        axes[1].grid(True, linestyle='--', alpha=0.6)
        cbar = plt.colorbar(scatter, ax=axes[1])
        cbar.set_label('Number of MLP Layers')

    plt.tight_layout()
    out_path = f'./outputs/optuna/figs/optuna_history_{prop}.png'
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == '__main__':
    os.makedirs('./outputs/optuna/figs', exist_ok=True)
    for p in ['Tensile_value', 'Yield_value', 'Elongation_value']:
        plot_optuna(p)
