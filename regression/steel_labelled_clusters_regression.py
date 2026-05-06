#!/usr/bin/env python
# coding: utf-8
"""Run the latest gated-fusion regression model on labelled-cluster steel data.

This script reuses the refactored SteelBERT regression architecture from
``hea_regression.py`` while using the fixed deduplicated 8:2 split produced by
``scripts/prepare_labelled_clusters_data.py``.
"""

from __future__ import annotations

import copy
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent))
    from hea_regression import (  # noqa: E402
        DEFAULT_CONFIG,
        MODEL_NAME,
        GatedFusionRegressor,
        TowerEncoder,
        build_mlp,
        build_tensors,
        eval_metrics,
        extract_element_embeddings,
        fit_standardizer,
        get_cls_embedding,
        load_steelbert,
        plot_parity,
        predict_split,
        set_seed,
        transform_with_standardizer,
    )
else:
    from .hea_regression import (  # noqa: E402
        DEFAULT_CONFIG,
        MODEL_NAME,
        GatedFusionRegressor,
        TowerEncoder,
        build_mlp,
        build_tensors,
        eval_metrics,
        extract_element_embeddings,
        fit_standardizer,
        get_cls_embedding,
        load_steelbert,
        plot_parity,
        predict_split,
        set_seed,
        transform_with_standardizer,
    )


REPO_ROOT = Path(__file__).resolve().parent.parent
LABELLED_DATASET_DIR = REPO_ROOT / "datasets" / "steel_labelled_clusters"
DEFAULT_TRAIN_CSV = LABELLED_DATASET_DIR / "train.csv"
DEFAULT_VAL_CSV = LABELLED_DATASET_DIR / "val.csv"
DEFAULT_LABELLED_OUTPUT_DIR = REPO_ROOT / "regression" / "outputs" / "steel_labelled_clusters"

ELEMENT_COLS = [
    "H", "B", "C", "N", "O", "F", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ca",
    "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "As", "Y", "Zr",
    "Nb", "Mo", "Sn", "Sb", "La", "Ce", "Ta", "W", "Pb", "Bi",
]

TARGET_COLS = ["Tensile_value", "Yield_value", "Elongation_value"]

TARGET_DISPLAY = {
    "Tensile_value": "Ultimate Tensile Strength (MPa)",
    "Yield_value": "Yield Strength (MPa)",
    "Elongation_value": "Elongation / Ductility (%)",
}


class TextElementGatedFusionRegressor(nn.Module):
    """Two-tower gated fusion model without the raw-composition branch."""

    def __init__(
        self,
        text_input_dim: int,
        ele_input_dim: int,
        latent_dim: int,
        text_hidden_dims: list[int],
        ele_hidden_dims: list[int],
        gate_hidden_dim: int,
        head_hidden_dims: list[int],
        dropout: float,
    ):
        super().__init__()
        self.text_encoder = TowerEncoder(text_input_dim, text_hidden_dims, latent_dim, dropout)
        self.ele_encoder = TowerEncoder(ele_input_dim, ele_hidden_dims, latent_dim, dropout)
        self.gate = nn.Sequential(
            nn.Linear(latent_dim * 2, gate_hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, 2),
        )
        self.head = build_mlp(latent_dim * 3, head_hidden_dims, 1, dropout)

    def forward(self, text_embeds, ele_embeds, return_aux: bool = False):
        text_latent = self.text_encoder(text_embeds)
        ele_latent = self.ele_encoder(ele_embeds)
        stacked = torch.cat([text_latent, ele_latent], dim=1)
        gate_weights = torch.softmax(self.gate(stacked), dim=1)
        fused = gate_weights[:, 0:1] * text_latent + gate_weights[:, 1:2] * ele_latent
        pred = self.head(torch.cat([text_latent, ele_latent, fused], dim=1))
        if return_aux:
            return pred, {
                "text_latent": text_latent,
                "ele_latent": ele_latent,
                "gate_weights": gate_weights,
            }
        return pred


def validate_dataset(df: pd.DataFrame, name: str) -> None:
    required = ["Text", *ELEMENT_COLS, *TARGET_COLS]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def build_feature_bank(df: pd.DataFrame, text_embeds, ele_embeds) -> dict[str, np.ndarray]:
    return {
        "text_embeds": np.asarray(text_embeds, dtype=np.float32),
        "ele_embeds": np.asarray(ele_embeds, dtype=np.float32),
        "raw_composition": df[ELEMENT_COLS].to_numpy(dtype=np.float32),
    }


def extract_split_features(train_df, val_df, tokenizer, model, device, batch_size: int) -> tuple[dict, dict]:
    combined = pd.concat([train_df, val_df], ignore_index=True)
    texts = combined["Text"].fillna("").astype(str).tolist()

    print("Extracting text embeddings for labelled-cluster steel data ...")
    text_embeds = get_cls_embedding(texts, tokenizer, model, device, batch_size=batch_size)

    composition = combined[ELEMENT_COLS].to_numpy(dtype=np.float32)
    print("Extracting element embeddings for 36 steel element columns ...")
    ele_embeds = extract_element_embeddings(ELEMENT_COLS, composition, tokenizer, model, device)

    feature_bank = build_feature_bank(combined, text_embeds, ele_embeds)
    split = len(train_df)
    train_features = {key: value[:split] for key, value in feature_bank.items()}
    val_features = {key: value[split:] for key, value in feature_bank.items()}
    return train_features, val_features


def train_fixed_split(
    train_features_raw: dict,
    val_features_raw: dict,
    train_targets: np.ndarray,
    val_targets: np.ndarray,
    config: dict,
    seed: int,
    device: torch.device,
    feature_mode: str,
) -> dict:
    set_seed(seed)

    feature_stats = {key: fit_standardizer(value) for key, value in train_features_raw.items()}
    train_features = {
        key: transform_with_standardizer(train_features_raw[key], feature_stats[key])
        for key in train_features_raw
    }
    val_features = {
        key: transform_with_standardizer(val_features_raw[key], feature_stats[key])
        for key in val_features_raw
    }

    if feature_mode == "full":
        train_tensors = build_tensors(train_features, train_targets)
        val_tensors = build_tensors(val_features, val_targets)
    elif feature_mode == "text_ele":
        train_tensors = (
            torch.tensor(train_features["text_embeds"], dtype=torch.float32),
            torch.tensor(train_features["ele_embeds"], dtype=torch.float32),
            torch.tensor(train_targets, dtype=torch.float32).unsqueeze(1),
        )
        val_tensors = (
            torch.tensor(val_features["text_embeds"], dtype=torch.float32),
            torch.tensor(val_features["ele_embeds"], dtype=torch.float32),
            torch.tensor(val_targets, dtype=torch.float32).unsqueeze(1),
        )
    else:
        raise ValueError(f"unknown feature_mode: {feature_mode}")
    train_loader = DataLoader(
        TensorDataset(*train_tensors),
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=False,
    )

    if feature_mode == "full":
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
    else:
        model = TextElementGatedFusionRegressor(
            text_input_dim=train_features["text_embeds"].shape[1],
            ele_input_dim=train_features["ele_embeds"].shape[1],
            latent_dim=config["latent_dim"],
            text_hidden_dims=config["text_hidden_dims"],
            ele_hidden_dims=config["ele_hidden_dims"],
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
        for batch in train_loader:
            if feature_mode == "full":
                text_batch, ele_batch, raw_batch, y_batch = batch
                pred = model(text_batch.to(device), ele_batch.to(device), raw_batch.to(device))
            else:
                text_batch, ele_batch, y_batch = batch
                pred = model(text_batch.to(device), ele_batch.to(device))
            loss = loss_fn(pred, y_batch.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            if feature_mode == "full":
                val_pred = model(
                    val_tensors[0].to(device),
                    val_tensors[1].to(device),
                    val_tensors[2].to(device),
                )
                val_loss = loss_fn(val_pred, val_tensors[3].to(device)).item()
            else:
                val_pred = model(val_tensors[0].to(device), val_tensors[1].to(device))
                val_loss = loss_fn(val_pred, val_tensors[2].to(device)).item()

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
    if feature_mode == "full":
        train_pred, train_gate_weights = predict_split(model, train_features, device)
        val_pred, val_gate_weights = predict_split(model, val_features, device)
    else:
        with torch.no_grad():
            train_pred_tensor, train_aux = model(
                torch.tensor(train_features["text_embeds"], dtype=torch.float32, device=device),
                torch.tensor(train_features["ele_embeds"], dtype=torch.float32, device=device),
                return_aux=True,
            )
            val_pred_tensor, val_aux = model(
                torch.tensor(val_features["text_embeds"], dtype=torch.float32, device=device),
                torch.tensor(val_features["ele_embeds"], dtype=torch.float32, device=device),
                return_aux=True,
            )
        train_pred = train_pred_tensor.cpu().numpy().flatten()
        val_pred = val_pred_tensor.cpu().numpy().flatten()
        train_gate_weights = train_aux["gate_weights"].cpu().numpy()
        val_gate_weights = val_aux["gate_weights"].cpu().numpy()

    return {
        "seed": seed,
        "stopped_epoch": epoch + 1,
        "train": eval_metrics(train_targets, train_pred),
        "val": eval_metrics(val_targets, val_pred),
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
        "train_pred": train_pred,
        "val_pred": val_pred,
        "train_gate_mean": train_gate_weights.mean(axis=0),
        "val_gate_mean": val_gate_weights.mean(axis=0),
    }


def run_target(train_df, val_df, train_features, val_features, target, seeds, config, device, out_dir_base):
    print(f"\n{'=' * 60}")
    print(f"Target: {TARGET_DISPLAY.get(target, target)}")
    print(f"Train={len(train_df)}  Val={len(val_df)}")
    print(f"{'=' * 60}")

    train_targets = train_df[target].to_numpy(dtype=np.float32)
    val_targets = val_df[target].to_numpy(dtype=np.float32)
    feature_mode = config["feature_mode"]
    out_dir = Path(out_dir_base) / feature_mode / target
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for seed in tqdm(seeds, desc=f"Seeds ({target})"):
        res = train_fixed_split(
            train_features,
            val_features,
            train_targets,
            val_targets,
            config,
            seed,
            device,
            feature_mode=feature_mode,
        )
        results.append(res)
        print(
            f"  seed={seed}  train R2={res['train']['r2']:.3f}  "
            f"val R2={res['val']['r2']:.3f}  gap={res['train']['r2'] - res['val']['r2']:.3f}  "
            f"stopped@{res['stopped_epoch']}"
        )

    best = max(results, key=lambda result: result["val"]["r2"])
    torch.save(
        {
            "model_state": best["model_state"],
            "model_config": best["model_config"],
            "feature_stats": best["feature_stats"],
            "seed": best["seed"],
            "target": target,
            "feature_mode": feature_mode,
            "train_csv": str(DEFAULT_TRAIN_CSV),
            "val_csv": str(DEFAULT_VAL_CSV),
        },
        out_dir / "best_model.pt",
    )

    plot_parity(
        train_targets,
        best["train_pred"],
        best["train"],
        f"{target} - Train (seed={best['seed']})",
        out_dir / "parity_train.png",
    )
    plot_parity(
        val_targets,
        best["val_pred"],
        best["val"],
        f"{target} - Validation (seed={best['seed']})",
        out_dir / "parity_val.png",
    )

    summary_df = pd.DataFrame([
        {
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
            "train_gate_raw": round(float(result["train_gate_mean"][2]), 4) if feature_mode == "full" else np.nan,
            "val_gate_text": round(float(result["val_gate_mean"][0]), 4),
            "val_gate_ele": round(float(result["val_gate_mean"][1]), 4),
            "val_gate_raw": round(float(result["val_gate_mean"][2]), 4) if feature_mode == "full" else np.nan,
        }
        for result in results
    ])
    summary_df.to_csv(out_dir / "seed_summary.csv", index=False)
    print(f"\nBest seed={best['seed']}  val R2={best['val']['r2']:.4f}")
    print(f"Results saved to {out_dir}")
    print(summary_df.to_string(index=False))


def main():
    parser = ArgumentParser(description="Labelled-cluster steel regression with the latest gated-fusion model")
    parser.add_argument("--train", type=str, default=str(DEFAULT_TRAIN_CSV))
    parser.add_argument("--val", type=str, default=str(DEFAULT_VAL_CSV))
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_LABELLED_OUTPUT_DIR))
    parser.add_argument("--target", nargs="+", default=TARGET_COLS, choices=TARGET_COLS)
    parser.add_argument("--feature_mode", choices=["full", "text_ele"], default="full")
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

    train_df = pd.read_csv(args.train)
    val_df = pd.read_csv(args.val)
    validate_dataset(train_df, "train")
    validate_dataset(val_df, "val")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Train: {args.train} ({len(train_df)} rows)")
    print(f"Val:   {args.val} ({len(val_df)} rows)")

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
        "feature_mode": args.feature_mode,
    }

    tokenizer, bert = load_steelbert(MODEL_NAME, device)
    train_features, val_features = extract_split_features(
        train_df,
        val_df,
        tokenizer,
        bert,
        device,
        batch_size=config["embed_batch_size"],
    )

    for target in args.target:
        run_target(train_df, val_df, train_features, val_features, target, args.seeds, config, device, args.out_dir)


if __name__ == "__main__":
    main()
