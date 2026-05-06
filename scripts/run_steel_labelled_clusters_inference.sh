#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

INPUT_CSV="${REPO_ROOT}/datasets/steel_labelled_clusters/inference_sample.csv"
OUTPUT_CSV="${REPO_ROOT}/regression/outputs/steel_labelled_clusters/inference_predictions.csv"
CKPT_DIR="/internfs/Zy/Steelllm/ckpt/steel_labelled_clusters_regression"
MODEL_NAME="/internfs/Zy/Steelllm/ckpt/SteelBERT"
BATCH_SIZE="32"
TARGETS=("Tensile_value" "Yield_value" "Elongation_value")

usage() {
  cat <<'EOF'
Run labelled-cluster steel regression inference.

Usage:
  scripts/run_steel_labelled_clusters_inference.sh [options]

Options:
  --input PATH          Input CSV with Text and 36 element columns.
  --output PATH         Output prediction CSV.
  --ckpt-dir PATH       Directory containing *_best_model.pt checkpoints.
  --model-name PATH     SteelBERT checkpoint directory.
  --batch-size N        Embedding batch size.
  --targets LIST        Space-separated target names, quoted as one argument.
                        Example: --targets "Tensile_value Yield_value"
  -h, --help            Show this help message.

Defaults:
  input      datasets/steel_labelled_clusters/inference_sample.csv
  output     regression/outputs/steel_labelled_clusters/inference_predictions.csv
  ckpt-dir   /internfs/Zy/Steelllm/ckpt/steel_labelled_clusters_regression
  model-name /internfs/Zy/Steelllm/ckpt/SteelBERT
  batch-size 32
  targets    Tensile_value Yield_value Elongation_value
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      INPUT_CSV="$2"
      shift 2
      ;;
    --output)
      OUTPUT_CSV="$2"
      shift 2
      ;;
    --ckpt-dir)
      CKPT_DIR="$2"
      shift 2
      ;;
    --model-name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --targets)
      read -r -a TARGETS <<< "$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

python "${REPO_ROOT}/regression/steel_labelled_clusters_inference.py" \
  --input "${INPUT_CSV}" \
  --output "${OUTPUT_CSV}" \
  --ckpt_dir "${CKPT_DIR}" \
  --model_name "${MODEL_NAME}" \
  --batch_size "${BATCH_SIZE}" \
  --targets "${TARGETS[@]}"
