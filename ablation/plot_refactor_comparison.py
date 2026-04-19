from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent

LEGACY_SEED_SUMMARY = REPO_ROOT / "ablation" / "back" / "legacy_20260414_120737" / "results" / "v2_academic" / "HT_YS" / "seed_summary.csv"
CURRENT_SEED_SUMMARY = REPO_ROOT / "ablation" / "current" / "results" / "hea_refactor" / "v2_academic" / "HT_YS" / "seed_summary.csv"
CURRENT_GATE_SUMMARY = REPO_ROOT / "ablation" / "current" / "results" / "hea_refactor" / "v2_academic" / "HT_YS" / "gate_summary.csv"

OUTPUT_DIR = REPO_ROOT / "ablation" / "current" / "analysis"
OUTPUT_PNG = OUTPUT_DIR / "ht_ys_refactor_vs_legacy.png"
OUTPUT_CSV = OUTPUT_DIR / "ht_ys_refactor_vs_legacy_summary.csv"


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def summarize(df: pd.DataFrame) -> dict:
    best_row = df.loc[df["val_r2"].idxmax()]
    return {
        "train_r2_mean": df["train_r2"].mean(),
        "val_r2_mean": df["val_r2"].mean(),
        "val_r2_best": df["val_r2"].max(),
        "val_rmse_best": best_row["val_rmse"],
        "val_mae_best": best_row["val_mae"],
        "best_seed": int(best_row["seed"]),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    legacy_df = load_csv(LEGACY_SEED_SUMMARY)
    current_df = load_csv(CURRENT_SEED_SUMMARY)
    gate_df = load_csv(CURRENT_GATE_SUMMARY)

    legacy_stats = summarize(legacy_df)
    current_stats = summarize(current_df)

    summary_df = pd.DataFrame(
        [
            {"version": "legacy_v2_academic", **legacy_stats},
            {"version": "current_hea_refactor", **current_stats},
        ]
    )
    summary_df["best_val_r2_gain"] = summary_df["val_r2_best"] - legacy_stats["val_r2_best"]
    summary_df.to_csv(OUTPUT_CSV, index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=160)
    fig.suptitle("HT_YS: Legacy v2_academic vs Current Refactor", fontsize=16, fontweight="bold")

    # Panel 1: summary metrics
    metrics = ["train_r2_mean", "val_r2_mean", "val_r2_best"]
    labels = ["Train R² Mean", "Val R² Mean", "Val R² Best"]
    x = np.arange(len(metrics))
    width = 0.34

    legacy_vals = [legacy_stats[m] for m in metrics]
    current_vals = [current_stats[m] for m in metrics]

    bars1 = axes[0].bar(x - width / 2, legacy_vals, width, label="Legacy v2", color="#A2555A", alpha=0.8)
    bars2 = axes[0].bar(x + width / 2, current_vals, width, label="Current refactor", color="#1F4B73", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=8)
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(min(0, min(legacy_vals + current_vals) - 0.05), 1.0)
    axes[0].set_title("Summary Metrics")
    axes[0].legend(loc="upper left")

    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            axes[0].annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Panel 2: seed-wise validation R2
    merged = legacy_df[["seed", "val_r2"]].rename(columns={"val_r2": "legacy_val_r2"}).merge(
        current_df[["seed", "val_r2"]].rename(columns={"val_r2": "current_val_r2"}),
        on="seed",
        how="outer",
    ).sort_values("seed")

    axes[1].plot(merged["seed"], merged["legacy_val_r2"], marker="o", linewidth=2, color="#A2555A", label="Legacy v2")
    axes[1].plot(merged["seed"], merged["current_val_r2"], marker="o", linewidth=2, color="#1F4B73", label="Current refactor")
    axes[1].set_xticks(merged["seed"].tolist())
    axes[1].set_xlabel("Seed")
    axes[1].set_ylabel("Validation R²")
    axes[1].set_title("Seed-wise Validation Performance")
    axes[1].legend(loc="upper left")

    delta = current_stats["val_r2_best"] - legacy_stats["val_r2_best"]
    gate_val = gate_df.loc[gate_df["split"] == "val"].iloc[0]
    info_text = (
        f"Best Val R²: {legacy_stats['val_r2_best']:.3f} -> {current_stats['val_r2_best']:.3f}\n"
        f"Gain: +{delta:.3f}\n"
        f"Best Val RMSE: {legacy_stats['val_rmse_best']:.1f} -> {current_stats['val_rmse_best']:.1f}\n"
        f"Best Val MAE: {legacy_stats['val_mae_best']:.1f} -> {current_stats['val_mae_best']:.1f}\n"
        f"Current gate (val): text={gate_val['text_weight']:.2f}, "
        f"ele={gate_val['ele_weight']:.2f}, raw={gate_val['raw_weight']:.2f}"
    )
    axes[1].text(
        0.03,
        0.03,
        info_text,
        transform=axes[1].transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved comparison plot: {OUTPUT_PNG}")
    print(f"Saved comparison summary: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
