import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_summary(base_dir, output_name):
    targets = ['RT_YS', 'RT_UTS', 'RT_EL', 'HT_YS', 'HT_UTS', 'HT_EL']
    labels = ['RT YS', 'RT UTS', 'RT EL', 'HT YS', 'HT UTS', 'HT EL']
    
    train_means = []
    val_means = []
    val_bests = []

    print(f"Scanning results in {base_dir}...")
    for t in targets:
        path = os.path.join(base_dir, t, 'seed_summary.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
            train_means.append(df['train_r2'].mean())
            val_means.append(df['val_r2'].mean())
            val_bests.append(df['val_r2'].max())
        else:
            print(f"  Warning: {t}/seed_summary.csv not found, filling with zeros.")
            train_means.append(0)
            val_means.append(0)
            val_bests.append(0)

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    rects1 = ax.bar(x - width, train_means, width, label='Train $R^2$ (Mean)', color='#1F4788', alpha=0.6)
    rects2 = ax.bar(x, val_means, width, label='Val $R^2$ (Mean)', color='#A2555A', alpha=0.6)
    rects3 = ax.bar(x + width, val_bests, width, label='Val $R^2$ (Best Seed)', color='#CD7F32', alpha=0.9, hatch='//')

    ax.set_ylabel('$R^2$ Score')
    ax.set_title(f'HEA Ablation Study Summary - {output_name.upper()}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='best')
    ax.set_ylim(-max(0.2, abs(min(val_means+train_means))*1.1), 1.0)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    def autolabel(rects, fontsize=8):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords='offset points',
                        ha='center', va='bottom', fontsize=fontsize)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3, fontsize=9)

    fig.tight_layout()
    out_path = os.path.join(base_dir, f'overall_performance_{output_name}_best.png')
    plt.savefig(out_path)
    print(f"\n✅ Summary plot saved as: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate summary performance plot for HEA ablation study.")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing the target result folders (RT_YS, etc.)")
    parser.add_argument("--name", type=str, default="v1", help="Short name for the output image suffix (e.g., v1, v2)")
    
    args = parser.parse_args()
    plot_summary(args.dir, args.name)
