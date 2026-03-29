#!/bin/bash

set -e

MODEL_SAVE_DIR="${MODEL_SAVE_DIR:-./model}"
PREDS_SAVE_DIR="${PREDS_SAVE_DIR:-./preds}"
CACHE_DIR="${CACHE_DIR:-./.cache}"

MODELS="${MODELS:-steelberta matscibert bert scibert}"

echo "=== SteelBERTa Classification ==="
echo "Model save dir: $MODEL_SAVE_DIR"
echo "Preds save dir: $PREDS_SAVE_DIR"
echo "Cache dir: $CACHE_DIR"

for model_name in $MODELS; do
    echo "[Training] Model: $model_name"
    python -u cls.py \
        --model_name "$model_name" \
        --model_save_dir "$MODEL_SAVE_DIR" \
        --preds_save_dir "$PREDS_SAVE_DIR" \
        --cache_dir "$CACHE_DIR"
done

echo "=== Classification Complete ==="
