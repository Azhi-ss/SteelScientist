#!/usr/bin/env python
# coding: utf-8
"""Prepare the labelled steel-cluster Excel file as clean regression datasets.

The pipeline maps the raw spreadsheet to the project regression schema, audits
strict duplicate records, removes those duplicates, and writes a fixed 8:2
train/validation split for downstream regression experiments.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "datasets" / "steel_labelled_clusters"
DEFAULT_INPUT = DATASET_DIR / "raw" / "Steel database with labelled clusters.xlsx"
DEFAULT_OUTPUT = DATASET_DIR / "clean.csv"
FULL_OUTPUT = DATASET_DIR / "full_with_duplicates.csv"
DUPLICATE_ROWS_OUTPUT = DATASET_DIR / "duplicate_rows.csv"
DUPLICATE_GROUPS_OUTPUT = DATASET_DIR / "duplicate_groups.csv"
TRAIN_OUTPUT = DATASET_DIR / "train.csv"
VAL_OUTPUT = DATASET_DIR / "val.csv"
REGRESSION_XLSX_OUTPUT = DATASET_DIR / "reg_v1_train_data.xlsx"
DEFAULT_SEED = 42
DEFAULT_TRAIN_RATIO = 0.8

ELEMENT_COLS = [
    "H", "B", "C", "N", "O", "F", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ca",
    "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "As", "Y", "Zr",
    "Nb", "Mo", "Sn", "Sb", "La", "Ce", "Ta", "W", "Pb", "Bi",
]

META_COLS = [
    "DOIs", "Files", "problem", "status", "Table_topic", "title", "abstract", "Material",
    "Tensile_name", "Tensile_value", "Tensile_unit",
    "Yield_name", "Yield_value", "Yield_unit",
    "Elongation_name", "Elongation_value", "Elongation_unit",
]

TAIL_COLS = ["Other_ele", "Text_addition", "Text", "actions"]
CLUSTER_COLS = ["source_entry", "cluster_number", "cluster_label", "cluster_number_0_to_11"]
TARGET_COLS = ["Tensile_value", "Yield_value", "Elongation_value"]
DUPLICATE_KEY_COLS = ["Material", "Text", *ELEMENT_COLS, *TARGET_COLS, "cluster_number", "cluster_label"]

REQUIRED_RAW_COLS = [
    "Entry",
    "Name",
    "Processing condition",
    "(Ultimate) Tensile strength (MPa)",
    "Yield strength (MPa)",
    "Ductility (%)",
    "Cluster Number",
    "Cluster label",
]


def _clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def _numeric_series(raw: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in raw.columns:
        return pd.Series(default, index=raw.index, dtype="float64")
    return pd.to_numeric(raw[column], errors="coerce").fillna(default).astype("float64")


def clean_labelled_clusters(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert the raw labelled-cluster spreadsheet to the project CSV schema.

    This function deliberately does not deduplicate rows.
    """
    missing = [col for col in REQUIRED_RAW_COLS if col not in raw.columns]
    if missing:
        raise ValueError(f"Missing required raw columns: {missing}")

    cleaned = pd.DataFrame(index=raw.index)
    cleaned["DOIs"] = ""
    cleaned["Files"] = ""
    cleaned["problem"] = ""
    cleaned["status"] = 1
    cleaned["Table_topic"] = "Steel database with labelled clusters"
    cleaned["title"] = ""
    cleaned["abstract"] = ""
    cleaned["Material"] = raw["Name"].map(_clean_text)
    cleaned["Tensile_name"] = "Ultimate tensile strength"
    cleaned["Tensile_value"] = _numeric_series(raw, "(Ultimate) Tensile strength (MPa)")
    cleaned["Tensile_unit"] = "MPa"
    cleaned["Yield_name"] = "Yield strength"
    cleaned["Yield_value"] = _numeric_series(raw, "Yield strength (MPa)")
    cleaned["Yield_unit"] = "MPa"
    cleaned["Elongation_name"] = "Ductility"
    cleaned["Elongation_value"] = _numeric_series(raw, "Ductility (%)")
    cleaned["Elongation_unit"] = "%"

    for element in ELEMENT_COLS:
        cleaned[element] = _numeric_series(raw, element)

    processing_text = raw["Processing condition"].map(_clean_text)
    cleaned["Other_ele"] = ""
    cleaned["Text_addition"] = ""
    cleaned["Text"] = processing_text
    cleaned["actions"] = processing_text

    cleaned["source_entry"] = pd.to_numeric(raw["Entry"], errors="coerce").astype("Int64")
    cleaned["cluster_number"] = pd.to_numeric(raw["Cluster Number"], errors="coerce").astype("Int64")
    cleaned["cluster_label"] = raw["Cluster label"].map(_clean_text)
    if "Cluster Number (0 to 11)" in raw.columns:
        cleaned["cluster_number_0_to_11"] = pd.to_numeric(
            raw["Cluster Number (0 to 11)"], errors="coerce"
        ).astype("Int64")
    else:
        cleaned["cluster_number_0_to_11"] = cleaned["cluster_number"]

    return cleaned[META_COLS + ELEMENT_COLS + TAIL_COLS + CLUSTER_COLS].reset_index(drop=True)


def drop_strict_duplicates(cleaned: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Remove exact duplicate training records while preserving an audit trail."""
    duplicate_mask = cleaned.duplicated(DUPLICATE_KEY_COLS, keep=False)
    duplicate_rows = cleaned.loc[duplicate_mask].copy()

    if duplicate_rows.empty:
        duplicate_rows.insert(0, "duplicate_group_id", pd.Series(dtype="int64"))
        duplicate_groups = pd.DataFrame(
            columns=[
                "duplicate_group_id", "duplicate_count", "source_entries", "Material", "Text",
                *TARGET_COLS, "cluster_number", "cluster_label",
            ]
        )
    else:
        group_keys = duplicate_rows[DUPLICATE_KEY_COLS].drop_duplicates().reset_index(drop=True)
        group_keys.insert(0, "duplicate_group_id", range(1, len(group_keys) + 1))
        duplicate_rows = duplicate_rows.merge(group_keys, on=DUPLICATE_KEY_COLS, how="left")

        front_cols = [
            "duplicate_group_id", "source_entry", "Material", "Text",
            *TARGET_COLS, "cluster_number", "cluster_label",
        ]
        duplicate_rows = duplicate_rows[
            front_cols + [col for col in duplicate_rows.columns if col not in front_cols]
        ].sort_values(["duplicate_group_id", "source_entry"]).reset_index(drop=True)

        duplicate_groups = (
            duplicate_rows.groupby("duplicate_group_id", as_index=False)
            .agg(
                duplicate_count=("source_entry", "size"),
                source_entries=("source_entry", lambda values: ",".join(map(str, values.tolist()))),
                Material=("Material", "first"),
                Text=("Text", "first"),
                Tensile_value=("Tensile_value", "first"),
                Yield_value=("Yield_value", "first"),
                Elongation_value=("Elongation_value", "first"),
                cluster_number=("cluster_number", "first"),
                cluster_label=("cluster_label", "first"),
            )
            .sort_values(["duplicate_count", "duplicate_group_id"], ascending=[False, True])
            .reset_index(drop=True)
        )

    deduped = cleaned.drop_duplicates(DUPLICATE_KEY_COLS, keep="first").reset_index(drop=True)
    return deduped, duplicate_rows, duplicate_groups


def split_regression_dataset(
    df: pd.DataFrame,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a deterministic random 8:2 split for regression experiments."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")

    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    split_idx = int(len(shuffled) * train_ratio)
    train_df = shuffled.iloc[:split_idx].reset_index(drop=True)
    val_df = shuffled.iloc[split_idx:].reset_index(drop=True)
    return train_df, val_df


def prepare(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_excel(input_path)
    full = clean_labelled_clusters(raw)
    cleaned, duplicate_rows, duplicate_groups = drop_strict_duplicates(full)
    train_df, val_df = split_regression_dataset(cleaned, train_ratio=train_ratio, seed=seed)

    FULL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(FULL_OUTPUT, index=False)
    duplicate_rows.to_csv(DUPLICATE_ROWS_OUTPUT, index=False)
    duplicate_groups.to_csv(DUPLICATE_GROUPS_OUTPUT, index=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    train_df.to_csv(TRAIN_OUTPUT, index=False)
    val_df.to_csv(VAL_OUTPUT, index=False)
    REGRESSION_XLSX_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_excel(REGRESSION_XLSX_OUTPUT, index=False)
    return cleaned, train_df, val_df


def main() -> None:
    cleaned, train_df, val_df = prepare()
    print(f"Saved deduplicated clean data: {len(cleaned)} rows -> {DEFAULT_OUTPUT}")
    print(f"Saved fixed 8:2 split: train={len(train_df)} -> {TRAIN_OUTPUT}")
    print(f"Saved fixed 8:2 split: val={len(val_df)} -> {VAL_OUTPUT}")
    print(f"Saved full duplicate audit -> {DUPLICATE_GROUPS_OUTPUT}, {DUPLICATE_ROWS_OUTPUT}")


if __name__ == "__main__":
    main()
