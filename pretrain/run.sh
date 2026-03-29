#!/bin/bash

set -e

CACHE_DIR="${CACHE_DIR:-./cache}"
SAVE_DIR="${SAVE_DIR:-./output}"
DATA_DIR="${DATA_DIR:-./data}"

TRAIN_CORPUS="${DATA_DIR}/train_corpus.json"
VAL_CORPUS="${DATA_DIR}/val_corpus.json"
TRAIN_NORM="${DATA_DIR}/train_corpus_norm.txt"
VAL_NORM="${DATA_DIR}/val_corpus_norm.txt"
CORPUS_RANGE="${CORPUS_RANGE:-material}"

mkdir -p "$SAVE_DIR" "$DATA_DIR"

echo "=== SteelBERTa Pre-training Pipeline ==="
echo "Cache dir: $CACHE_DIR"
echo "Save dir: $SAVE_DIR"
echo "Data dir: $DATA_DIR"

echo "[1/4] Combining corpus..."
python -u json_combination.py \
    --abstract_file "${DATA_DIR}/steel_abstract.json" \
    --fulltext_file "${DATA_DIR}/steel_full_text_v3.json" \
    --train_output "${DATA_DIR}/train_corpus.json" \
    --val_output "${DATA_DIR}/val_corpus.json"

echo "[2/4] Normalizing corpus..."
python -u corpus_normalize.py \
    --train_corpus_file "$TRAIN_CORPUS" \
    --val_corpus_file "$VAL_CORPUS" \
    --train_norm_file "$TRAIN_NORM" \
    --val_norm_file "$VAL_NORM" \
    --corpus_range "$CORPUS_RANGE"

echo "[3/4] Training tokenizer..."
python -u tokenizer_train.py \
    --train_norm_file "$TRAIN_NORM" \
    --val_norm_file "$VAL_NORM" \
    --save_dir "$SAVE_DIR" \
    --cache_dir "$CACHE_DIR"

echo "[4/4] Training model..."
python -u model_train.py \
    --save_dir "$SAVE_DIR" \
    --cache_dir "$CACHE_DIR"

echo "=== Pre-training Complete ==="
