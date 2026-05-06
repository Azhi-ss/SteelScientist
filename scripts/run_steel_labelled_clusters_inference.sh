#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

INPUT_CSV="${1:-${REPO_ROOT}/datasets/steel_labelled_clusters/inference_sample.csv}"
OUTPUT_CSV="${2:-${REPO_ROOT}/regression/outputs/steel_labelled_clusters/inference_predictions.csv}"
CKPT_DIR="${3:-/internfs/Zy/Steelllm/ckpt/steel_labelled_clusters_regression}"

python "${REPO_ROOT}/regression/steel_labelled_clusters_inference.py" \
  --input "${INPUT_CSV}" \
  --output "${OUTPUT_CSV}" \
  --ckpt_dir "${CKPT_DIR}" \
  --targets Tensile_value Yield_value Elongation_value
