#!/usr/bin/env python
# coding: utf-8
"""Inference for labelled-cluster steel regression checkpoints."""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent))
    from hea_regression import (  # noqa: E402
        MODEL_NAME,
        GatedFusionRegressor,
        extract_element_embeddings,
        get_cls_embedding,
        load_steelbert,
        transform_with_standardizer,
    )
    from steel_labelled_clusters_regression import ELEMENT_COLS, TARGET_COLS  # noqa: E402
else:
    from .hea_regression import (  # noqa: E402
        MODEL_NAME,
        GatedFusionRegressor,
        extract_element_embeddings,
        get_cls_embedding,
        load_steelbert,
        transform_with_standardizer,
    )
    from .steel_labelled_clusters_regression import ELEMENT_COLS, TARGET_COLS  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = Path("/internfs/Zy/Steelllm/steel_labelled_clusters_inference_sample.csv")
DEFAULT_OUTPUT = REPO_ROOT / "regression" / "outputs" / "steel_labelled_clusters" / "inference_predictions.csv"
DEFAULT_CKPT_DIR = Path("/internfs/Zy/Steelllm/ckpt/steel_labelled_clusters_regression")


def validate_inference_dataset(df: pd.DataFrame) -> None:
    required = ["Text", *ELEMENT_COLS]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"inference input missing required columns: {missing}")


def checkpoint_path_for_target(ckpt_dir: Path, target: str) -> Path:
    return ckpt_dir / f"{target}_best_model.pt"


def build_model_from_checkpoint(checkpoint: dict, device: torch.device) -> GatedFusionRegressor:
    config = checkpoint["model_config"]
    model = GatedFusionRegressor(
        text_input_dim=config["text_input_dim"],
        ele_input_dim=config["ele_input_dim"],
        raw_input_dim=config["raw_input_dim"],
        latent_dim=config["latent_dim"],
        text_hidden_dims=config["text_hidden_dims"],
        ele_hidden_dims=config["ele_hidden_dims"],
        raw_hidden_dims=config["raw_hidden_dims"],
        gate_hidden_dim=config["gate_hidden_dim"],
        head_hidden_dims=config["head_hidden_dims"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def extract_features(df: pd.DataFrame, tokenizer, steelbert, device, batch_size: int) -> dict[str, np.ndarray]:
    text_embeds = get_cls_embedding(
        df["Text"].fillna("").astype(str).tolist(),
        tokenizer,
        steelbert,
        device,
        batch_size=batch_size,
    )
    raw_composition = df[ELEMENT_COLS].to_numpy(dtype=np.float32)
    ele_embeds = extract_element_embeddings(ELEMENT_COLS, raw_composition, tokenizer, steelbert, device)
    return {
        "text_embeds": text_embeds,
        "ele_embeds": ele_embeds,
        "raw_composition": raw_composition,
    }


def predict_one_target(
    features: dict[str, np.ndarray],
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[str, np.ndarray, np.ndarray]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    target = checkpoint["target"]
    feature_stats = checkpoint["feature_stats"]
    normalized = {
        key: transform_with_standardizer(features[key], feature_stats[key])
        for key in ["text_embeds", "ele_embeds", "raw_composition"]
    }
    model = build_model_from_checkpoint(checkpoint, device)
    with torch.no_grad():
        preds, aux = model(
            torch.tensor(normalized["text_embeds"], dtype=torch.float32, device=device),
            torch.tensor(normalized["ele_embeds"], dtype=torch.float32, device=device),
            torch.tensor(normalized["raw_composition"], dtype=torch.float32, device=device),
            return_aux=True,
        )
    return target, preds.cpu().numpy().flatten(), aux["gate_weights"].cpu().numpy()


def run_inference(args) -> pd.DataFrame:
    input_path = Path(args.input)
    output_path = Path(args.output)
    ckpt_dir = Path(args.ckpt_dir)

    df = pd.read_csv(input_path)
    validate_inference_dataset(df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Input: {input_path} ({len(df)} rows)")
    print(f"Checkpoint dir: {ckpt_dir}")

    tokenizer, steelbert = load_steelbert(args.model_name, device)
    features = extract_features(df, tokenizer, steelbert, device, batch_size=args.batch_size)

    output = df.copy()
    for target in args.targets:
        ckpt_path = checkpoint_path_for_target(ckpt_dir, target)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
        target_name, preds, gate_weights = predict_one_target(features, ckpt_path, device)
        output[f"pred_{target_name}"] = preds
        output[f"{target_name}_gate_text"] = gate_weights[:, 0]
        output[f"{target_name}_gate_ele"] = gate_weights[:, 1]
        output[f"{target_name}_gate_raw"] = gate_weights[:, 2]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Saved predictions -> {output_path}")
    return output


def main() -> None:
    parser = ArgumentParser(description="Run labelled-cluster steel regression inference")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="CSV with Text and 36 element columns")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output CSV path")
    parser.add_argument("--ckpt_dir", type=str, default=str(DEFAULT_CKPT_DIR), help="Directory containing *_best_model.pt")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME, help="SteelBERT checkpoint path")
    parser.add_argument("--targets", nargs="+", choices=TARGET_COLS, default=TARGET_COLS)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
