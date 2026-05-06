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
