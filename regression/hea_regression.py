"""
HEA 回归管线：SteelBERT 特征提取 + 三路独立编码器 + 门控融合回归头。

用法:
    python regression/hea_regression.py
    python regression/hea_regression.py --target HT_YS
    python regression/hea_regression.py --data ablation/current/datasets/v2_academic/hea_data.csv --target HT_YS
"""

import copy
import random
import warnings
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
PRIMARY_CSV_PATH = REPO_ROOT / "datasets" / "hea_with_text.csv"
FALLBACK_CSV_PATH = REPO_ROOT / "ablation" / "current" / "datasets" / "v2_academic" / "hea_data.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "regression" / "outputs" / "hea_refactor"

MODEL_NAME = "/internfs/Zy/Steelllm/ckpt/SteelBERT"

ELEMENT_COLS = [
    "Co", "Cr", "Ni", "Al", "Ti", "Ta",
    "B", "C", "Cu", "Fe", "Hf", "Mn", "Mo", "Nb", "V", "W", "Zr",
]

TARGET_COLS = ["RT_YS", "RT_UTS", "RT_EL", "HT_YS", "HT_UTS", "HT_EL"]

TARGET_DISPLAY = {
    "RT_YS": "Room-Temp Yield Strength (MPa)",
    "RT_UTS": "Room-Temp Tensile Strength (MPa)",
    "RT_EL": "Room-Temp Elongation (%)",
    "HT_YS": "High-Temp Yield Strength (MPa)",
    "HT_UTS": "High-Temp Tensile Strength (MPa)",
    "HT_EL": "High-Temp Elongation (%)",
}

DEFAULT_CONFIG = {
    "latent_dim": 16,
    "text_hidden_dims": [64],
    "ele_hidden_dims": [64],
    "raw_hidden_dims": [16],
    "head_hidden_dims": [64, 32],
    "gate_hidden_dim": 16,
    "dropout": 0.2,
    "lr": 3e-4,
    "batch_size": 16,
    "epochs": 300,
    "patience": 30,
    "split_ratio": 0.8,
    "embed_batch_size": 32,
}


def resolve_default_csv_path() -> Path:
    if PRIMARY_CSV_PATH.exists():
        return PRIMARY_CSV_PATH
    return FALLBACK_CSV_PATH


# ---------------------------------------------------------------------------
#  SteelBERT 特征提取 (完全冻结)
# ---------------------------------------------------------------------------

def load_steelbert(model_name: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return tokenizer, model


def get_cls_embedding(texts: list[str], tokenizer, model, device, batch_size: int = 32) -> np.ndarray:
    """对一组文本提取 [CLS] 向量，返回 (N, 768)。"""
    all_embeds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, padding="max_length", max_length=512, truncation=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        cls_vec = out.last_hidden_state[:, 0].cpu().numpy()
        all_embeds.append(cls_vec)
        torch.cuda.empty_cache()
    return np.vstack(all_embeds).astype(np.float32)


def extract_element_embeddings(
    element_names: list[str],
    composition: np.ndarray,
    tokenizer,
    model,
    device,
) -> np.ndarray:
    """按 at% 加权平均元素 embedding，返回 (N, 768)。"""
    ele_embeds = get_cls_embedding(element_names, tokenizer, model, device)
    result = np.zeros((composition.shape[0], ele_embeds.shape[1]), dtype=np.float32)
    for i in range(composition.shape[0]):
        weights = composition[i]
        if weights.sum() > 0:
            result[i] = np.average(ele_embeds, axis=0, weights=weights)
    return result


def extract_all_features(df: pd.DataFrame, tokenizer, model, device, batch_size: int = 32) -> dict:
    """提取三路特征：text_embed + ele_embed + raw_composition。"""
    print("Extracting text embeddings ...")
    text_embeds = get_cls_embedding(df["Text"].fillna("").astype(str).tolist(), tokenizer, model, device, batch_size)
    print(f"  text_embeds shape: {text_embeds.shape}")

    raw_composition = df[ELEMENT_COLS].values.astype(np.float32)
    print(f"  raw_composition shape: {raw_composition.shape}")

    print("Extracting element embeddings ...")
    ele_embeds = extract_element_embeddings(ELEMENT_COLS, raw_composition, tokenizer, model, device)
    print(f"  ele_embeds shape: {ele_embeds.shape}")

    return {
        "text_embeds": text_embeds,
        "ele_embeds": ele_embeds,
        "raw_composition": raw_composition,
    }


# ---------------------------------------------------------------------------
#  三路编码器 + 门控融合回归头
# ---------------------------------------------------------------------------

def build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int, dropout: float) -> nn.Sequential:
    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend([
            nn.Linear(prev_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
        ])
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class TowerEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int, dropout: float):
        super().__init__()
        self.net = build_mlp(input_dim, hidden_dims, latent_dim, dropout)

    def forward(self, x):
        return self.net(x)


class GatedFusionRegressor(nn.Module):
    def __init__(
        self,
        text_input_dim: int,
        ele_input_dim: int,
        raw_input_dim: int,
        latent_dim: int,
        text_hidden_dims: list[int],
        ele_hidden_dims: list[int],
        raw_hidden_dims: list[int],
        gate_hidden_dim: int,
        head_hidden_dims: list[int],
        dropout: float,
    ):
        super().__init__()
        self.text_encoder = TowerEncoder(text_input_dim, text_hidden_dims, latent_dim, dropout)
        self.ele_encoder = TowerEncoder(ele_input_dim, ele_hidden_dims, latent_dim, dropout)
        self.raw_encoder = TowerEncoder(raw_input_dim, raw_hidden_dims, latent_dim, dropout)

        self.gate = nn.Sequential(
            nn.Linear(latent_dim * 3, gate_hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, 3),
        )
        self.head = build_mlp(latent_dim * 4, head_hidden_dims, 1, dropout)

    def forward(self, text_embeds, ele_embeds, raw_composition, return_aux: bool = False):
        text_latent = self.text_encoder(text_embeds)
        ele_latent = self.ele_encoder(ele_embeds)
        raw_latent = self.raw_encoder(raw_composition)

        stacked = torch.cat([text_latent, ele_latent, raw_latent], dim=1)
        gate_weights = torch.softmax(self.gate(stacked), dim=1)
        fused = (
            gate_weights[:, 0:1] * text_latent
            + gate_weights[:, 1:2] * ele_latent
            + gate_weights[:, 2:3] * raw_latent
        )
        head_input = torch.cat([text_latent, ele_latent, raw_latent, fused], dim=1)
        pred = self.head(head_input)
        if return_aux:
            return pred, {
                "text_latent": text_latent,
                "ele_latent": ele_latent,
                "raw_latent": raw_latent,
                "gate_weights": gate_weights,
            }
        return pred


# ---------------------------------------------------------------------------
#  训练 / 评估
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "r2": round(r2_score(y_true, y_pred), 4),
        "rmse": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "mae": round(mean_absolute_error(y_true, y_pred), 4),
    }


def fit_standardizer(array: np.ndarray) -> dict:
    mean = array.mean(axis=0, keepdims=True).astype(np.float32)
    std = array.std(axis=0, keepdims=True).astype(np.float32)
    std[std < 1e-6] = 1.0
    return {"mean": mean, "std": std}


def transform_with_standardizer(array: np.ndarray, stats: dict) -> np.ndarray:
    return ((array - stats["mean"]) / stats["std"]).astype(np.float32)


def build_tensors(features: dict, targets: np.ndarray):
    return (
        torch.tensor(features["text_embeds"], dtype=torch.float32),
        torch.tensor(features["ele_embeds"], dtype=torch.float32),
        torch.tensor(features["raw_composition"], dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32).unsqueeze(1),
    )


def predict_split(model, features: dict, device: torch.device):
    with torch.no_grad():
        preds, aux = model(
            torch.tensor(features["text_embeds"], dtype=torch.float32, device=device),
            torch.tensor(features["ele_embeds"], dtype=torch.float32, device=device),
            torch.tensor(features["raw_composition"], dtype=torch.float32, device=device),
            return_aux=True,
        )
    return preds.cpu().numpy().flatten(), aux["gate_weights"].cpu().numpy()


def train_one_seed(feature_bank: dict, targets: np.ndarray, config: dict, seed: int, device: torch.device) -> dict:
    set_seed(seed)
    n = len(targets)
    indices = np.random.permutation(n)
    split = int(n * config["split_ratio"])
    train_idx, val_idx = indices[:split], indices[split:]

    train_features_raw = {key: value[train_idx] for key, value in feature_bank.items()}
    val_features_raw = {key: value[val_idx] for key, value in feature_bank.items()}

    feature_stats = {key: fit_standardizer(value) for key, value in train_features_raw.items()}
    train_features = {
        key: transform_with_standardizer(train_features_raw[key], feature_stats[key])
        for key in train_features_raw
    }
    val_features = {
        key: transform_with_standardizer(val_features_raw[key], feature_stats[key])
        for key in val_features_raw
    }

    train_tensors = build_tensors(train_features, targets[train_idx])
    val_tensors = build_tensors(val_features, targets[val_idx])
    train_loader = DataLoader(
        TensorDataset(*train_tensors),
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=False,
    )

    model = GatedFusionRegressor(
        text_input_dim=train_features["text_embeds"].shape[1],
        ele_input_dim=train_features["ele_embeds"].shape[1],
        raw_input_dim=train_features["raw_composition"].shape[1],
        latent_dim=config["latent_dim"],
        text_hidden_dims=config["text_hidden_dims"],
        ele_hidden_dims=config["ele_hidden_dims"],
        raw_hidden_dims=config["raw_hidden_dims"],
        gate_hidden_dim=config["gate_hidden_dim"],
        head_hidden_dims=config["head_hidden_dims"],
        dropout=config["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.01,
        total_iters=config["epochs"],
    )
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(config["epochs"]):
        model.train()
        for text_batch, ele_batch, raw_batch, y_batch in train_loader:
            text_batch = text_batch.to(device)
            ele_batch = ele_batch.to(device)
            raw_batch = raw_batch.to(device)
            y_batch = y_batch.to(device)

            pred = model(text_batch, ele_batch, raw_batch)
            loss = loss_fn(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(
                val_tensors[0].to(device),
                val_tensors[1].to(device),
                val_tensors[2].to(device),
            )
            val_loss = loss_fn(val_pred, val_tensors[3].to(device)).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= config["patience"]:
            break

    model.load_state_dict(best_state)
    model.eval()
    train_pred, train_gate_weights = predict_split(model, train_features, device)
    val_pred, val_gate_weights = predict_split(model, val_features, device)

    train_metrics = eval_metrics(targets[train_idx], train_pred)
    val_metrics = eval_metrics(targets[val_idx], val_pred)

    return {
        "seed": seed,
        "stopped_epoch": epoch + 1,
        "train": train_metrics,
        "val": val_metrics,
        "model_state": best_state,
        "model_config": {
            "text_input_dim": train_features["text_embeds"].shape[1],
            "ele_input_dim": train_features["ele_embeds"].shape[1],
            "raw_input_dim": train_features["raw_composition"].shape[1],
            "latent_dim": config["latent_dim"],
            "text_hidden_dims": config["text_hidden_dims"],
            "ele_hidden_dims": config["ele_hidden_dims"],
            "raw_hidden_dims": config["raw_hidden_dims"],
            "gate_hidden_dim": config["gate_hidden_dim"],
            "head_hidden_dims": config["head_hidden_dims"],
            "dropout": config["dropout"],
        },
        "feature_stats": feature_stats,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "train_pred": train_pred,
        "val_pred": val_pred,
        "train_gate_mean": train_gate_weights.mean(axis=0),
        "val_gate_mean": val_gate_weights.mean(axis=0),
    }


def plot_parity(y_true, y_pred, metrics, title, save_path):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
    ax.scatter(y_true, y_pred, s=20, color="#1F4B73", alpha=0.8)
    lo = min(y_true.min(), y_pred.min()) * 0.9
    hi = max(y_true.max(), y_pred.max()) * 1.1
    ax.plot([lo, hi], [lo, hi], "--", color="#A2555A", linewidth=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True", fontsize=12)
    ax.set_ylabel("Predicted", fontsize=12)
    ax.set_title(f"{title}\nR²={metrics['r2']}  RMSE={metrics['rmse']}  MAE={metrics['mae']}", fontsize=10)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
#  主流程
# ---------------------------------------------------------------------------

def run_pipeline(df: pd.DataFrame, feature_bank: dict, target: str, seeds: list[int], config: dict, device: torch.device, out_dir_base: str):
    mask = df[target].notna()
    valid_indices = np.flatnonzero(mask.to_numpy())
    n_valid = len(valid_indices)
    print(f"\n{'='*60}")
    print(f"Target: {TARGET_DISPLAY.get(target, target)}  ({n_valid} valid samples)")
    print(f"{'='*60}")

    if n_valid < 10:
        print(f"  SKIP: too few samples ({n_valid})")
        return None

    target_feature_bank = {key: value[valid_indices] for key, value in feature_bank.items()}
    targets_arr = df.loc[mask, target].to_numpy(dtype=np.float32)

    out_dir = Path(out_dir_base) / target
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for seed in tqdm(seeds, desc=f"Seeds ({target})"):
        res = train_one_seed(target_feature_bank, targets_arr, config, seed, device)
        results.append(res)
        print(
            f"  seed={seed}  train R²={res['train']['r2']:.3f}  "
            f"val R²={res['val']['r2']:.3f}  gap={res['train']['r2'] - res['val']['r2']:.3f}  "
            f"stopped@{res['stopped_epoch']}"
        )

    best = max(results, key=lambda result: result["val"]["r2"])
    print(f"\n  Best seed={best['seed']}  val R²={best['val']['r2']:.4f}")

    checkpoint = {
        "model_state": best["model_state"],
        "model_config": best["model_config"],
        "feature_stats": best["feature_stats"],
        "seed": best["seed"],
        "target": target,
    }
    torch.save(checkpoint, out_dir / "best_model.pt")

    plot_parity(
        targets_arr[best["train_idx"]],
        best["train_pred"],
        best["train"],
        f"{target} — Train (seed={best['seed']})",
        out_dir / "parity_train.png",
    )
    plot_parity(
        targets_arr[best["val_idx"]],
        best["val_pred"],
        best["val"],
        f"{target} — Validation (seed={best['seed']})",
        out_dir / "parity_val.png",
    )

    summary_rows = []
    for result in results:
        summary_rows.append({
            "seed": result["seed"],
            "stopped_epoch": result["stopped_epoch"],
            "train_r2": result["train"]["r2"],
            "train_rmse": result["train"]["rmse"],
            "train_mae": result["train"]["mae"],
            "val_r2": result["val"]["r2"],
            "val_rmse": result["val"]["rmse"],
            "val_mae": result["val"]["mae"],
            "gap": round(result["train"]["r2"] - result["val"]["r2"], 4),
            "train_gate_text": round(float(result["train_gate_mean"][0]), 4),
            "train_gate_ele": round(float(result["train_gate_mean"][1]), 4),
            "train_gate_raw": round(float(result["train_gate_mean"][2]), 4),
            "val_gate_text": round(float(result["val_gate_mean"][0]), 4),
            "val_gate_ele": round(float(result["val_gate_mean"][1]), 4),
            "val_gate_raw": round(float(result["val_gate_mean"][2]), 4),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "seed_summary.csv", index=False)

    gate_summary = pd.DataFrame([
        {
            "split": "train",
            "text_weight": round(float(best["train_gate_mean"][0]), 4),
            "ele_weight": round(float(best["train_gate_mean"][1]), 4),
            "raw_weight": round(float(best["train_gate_mean"][2]), 4),
        },
        {
            "split": "val",
            "text_weight": round(float(best["val_gate_mean"][0]), 4),
            "ele_weight": round(float(best["val_gate_mean"][1]), 4),
            "raw_weight": round(float(best["val_gate_mean"][2]), 4),
        },
    ])
    gate_summary.to_csv(out_dir / "gate_summary.csv", index=False)

    print(f"\n  Results saved to {out_dir}/")
    print(summary_df.to_string(index=False))
    return best


def main():
    parser = ArgumentParser(description="HEA SteelBERT Regression Pipeline")
    parser.add_argument("--data", type=str, default=str(resolve_default_csv_path()), help="Path to CSV dataset")
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--target", nargs="+", default=TARGET_COLS, help="Target columns to predict (default: all)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--patience", type=int, default=DEFAULT_CONFIG["patience"])
    parser.add_argument("--dropout", type=float, default=DEFAULT_CONFIG["dropout"])
    parser.add_argument("--latent_dim", type=int, default=DEFAULT_CONFIG["latent_dim"])
    parser.add_argument("--gate_hidden_dim", type=int, default=DEFAULT_CONFIG["gate_hidden_dim"])
    parser.add_argument("--text_hidden_dims", nargs="*", type=int, default=DEFAULT_CONFIG["text_hidden_dims"])
    parser.add_argument("--ele_hidden_dims", nargs="*", type=int, default=DEFAULT_CONFIG["ele_hidden_dims"])
    parser.add_argument("--raw_hidden_dims", nargs="*", type=int, default=DEFAULT_CONFIG["raw_hidden_dims"])
    parser.add_argument("--head_hidden_dims", nargs="*", type=int, default=DEFAULT_CONFIG["head_hidden_dims"])
    parser.add_argument("--embed_batch_size", type=int, default=DEFAULT_CONFIG["embed_batch_size"])
    args = parser.parse_args()

    csv_path = Path(args.data)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing_cols = [col for col in ["Text", *ELEMENT_COLS, *TARGET_COLS] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {csv_path}")

    config = {
        **DEFAULT_CONFIG,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "dropout": args.dropout,
        "latent_dim": args.latent_dim,
        "gate_hidden_dim": args.gate_hidden_dim,
        "text_hidden_dims": args.text_hidden_dims,
        "ele_hidden_dims": args.ele_hidden_dims,
        "raw_hidden_dims": args.raw_hidden_dims,
        "head_hidden_dims": args.head_hidden_dims,
        "embed_batch_size": args.embed_batch_size,
    }

    tokenizer, bert = load_steelbert(MODEL_NAME, device)
    feature_bank = extract_all_features(df, tokenizer, bert, device, batch_size=config["embed_batch_size"])

    for target in args.target:
        if target not in TARGET_COLS:
            print(f"Unknown target '{target}', skipping. Valid: {TARGET_COLS}")
            continue
        run_pipeline(df, feature_bank, target, args.seeds, config, device, args.out_dir)


if __name__ == "__main__":
    main()
