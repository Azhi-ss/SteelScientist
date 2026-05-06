# Steel Labelled Clusters Dataset

This directory contains the cleaned steel labelled-clusters dataset used by
`regression/steel_labelled_clusters_regression.py`.

## Files

| File | Rows | Description |
|------|------|-------------|
| `raw/Steel database with labelled clusters.xlsx` | 3234 | Original spreadsheet. |
| `full_with_duplicates.csv` | 3234 | Raw spreadsheet mapped to the project regression schema. |
| `clean.csv` | 1943 | Strictly deduplicated regression mother table. |
| `train.csv` | 1554 | Fixed 8:2 training split generated from `clean.csv`. |
| `val.csv` | 389 | Fixed 8:2 validation split generated from `clean.csv`. |
| `duplicate_rows.csv` | 2560 | All rows that belong to strict duplicate groups. |
| `duplicate_groups.csv` | 1269 | One row per strict duplicate group. |
| `reg_v1_train_data.xlsx` | 1943 | Excel export for legacy `reg_v1.py` compatibility. |
| `inference_sample.csv` | 20 | Small inference example copied from the validation split; also copied to `/internfs/Zy/Steelllm/steel_labelled_clusters_inference_sample.csv`. |

## Regeneration

Run from the repository root:

```bash
python scripts/prepare_labelled_clusters_data.py
```

The split uses `seed=42` and a fixed 8:2 train/validation ratio.

## Deduplication Key

Strict duplicates are identified by:

```text
Material + Text + 36 element columns + Tensile_value + Yield_value + Elongation_value + cluster_number + cluster_label
```

`actions` is retained only as a legacy schema compatibility field and is copied
from `Text`; it is not used as an effective independent feature in the current
regression model.

## Inference Example

The default inference script uses the full-model weights copied to:

```text
/internfs/Zy/Steelllm/ckpt/steel_labelled_clusters_regression/
```

Run from the repository root:

```bash
./scripts/run_steel_labelled_clusters_inference.sh
```

Defaults:

```text
input : /internfs/Zy/Steelllm/steel_labelled_clusters_inference_sample.csv
output: regression/outputs/steel_labelled_clusters/inference_predictions.csv
ckpt  : /internfs/Zy/Steelllm/ckpt/steel_labelled_clusters_regression
```

You can also pass custom options:

```bash
./scripts/run_steel_labelled_clusters_inference.sh \
  --input /internfs/Zy/Steelllm/steel_labelled_clusters_inference_sample.csv \
  --output regression/outputs/steel_labelled_clusters/inference_predictions.csv \
  --ckpt-dir /internfs/Zy/Steelllm/ckpt/steel_labelled_clusters_regression \
  --model-name /internfs/Zy/Steelllm/ckpt/SteelBERT \
  --batch-size 32 \
  --targets "Tensile_value Yield_value Elongation_value"
```

The input CSV must contain `Text` and the 36 standard element columns. If target
columns are present, they are preserved in the output for quick comparison.
