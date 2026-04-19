"""
生成消融实验 V4 数据（不含成分信息）
此脚本读取原始数据，生成纯文本描述（仅包含相结构、热处理和测试温度），
旨在避免 Text 列和数值特征列包含重复信息（成分重叠），从而测试纯加工工艺文本的增益。
"""
import re
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
XLSX_PATH = REPO_ROOT / "datasets" / "hea-composition-performance.xlsx"

OUTPUT_DIR = REPO_ROOT / "ablation" / "datasets" / "v4_text_no_comp"
OUTPUT_PATH = OUTPUT_DIR / "hea_data.csv"

MAIN_ELEMENTS = ["Co", "Cr", "Ni", "Al", "Ti", "Ta"]

TARGET_MAP = {
    "RT_YS":   "室温屈服强度(MPa)",
    "RT_UTS":  "室温抗拉强度(MPa)",
    "RT_EL":   "室温延伸率(%)",
    "HT_YS":   "高温屈服强度(MPa)",
    "HT_UTS":  "高温抗拉强度(MPa)",
    "HT_EL":   "高温延伸率(%)",
}

def parse_other_elements(s: str) -> dict:
    if pd.isna(s) or str(s).strip() == "-":
        return {}
    pairs = re.findall(r"([A-Z][a-z]?)\s*:\s*([\d.]+)", str(s))
    return {elem: float(val) for elem, val in pairs}

def to_float(x):
    if pd.isna(x) or str(x).strip() == "-":
        return 0.0
    try:
        return float(x)
    except (ValueError, TypeError):
        return 0.0

def to_float_or_nan(x):
    if pd.isna(x) or str(x).strip() == "-":
        return float("nan")
    try:
        return float(x)
    except (ValueError, TypeError):
        return float("nan")

PHASE_SYNONYMS = {
    "L12": "ordered L1-2 intermetallic precipitates",
    "η": "eta phase",
    "B2": "B2 ordered phase",
    "Laves": "Laves intermetallic phase",
}

def _expand_phase(phase_raw: str) -> str:
    if pd.isna(phase_raw) or str(phase_raw).strip() == "-":
        return "unknown phase structure"
    text = str(phase_raw)
    for short, full in PHASE_SYNONYMS.items():
        text = text.replace(short, full)
    return text

def template_text_no_comp(row: pd.Series) -> str:
    """仅拼接相结构 + 热处理 + 测试温度，完全去除成分描述。"""
    phase = _expand_phase(row.get("相结构", "-"))
    ht = row.get("热处理状态", "-")
    ht = ht if (not pd.isna(ht) and str(ht).strip() != "-") else "as-received"
    temp = row.get("高温温度(°C)", "room temperature")
    temp = temp if (not pd.isna(temp) and str(temp).strip() != "-") else "room temperature"

    return (
        f"This high-entropy alloy with {phase} "
        f"was heat treated at {ht} and tested at {temp} degrees Celsius."
    )

def preprocess() -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_excel(XLSX_PATH)

    for el in MAIN_ELEMENTS:
        raw[el] = raw[f"{el}(at%)"].apply(to_float)

    other_parsed = raw["其他元素"].apply(parse_other_elements)
    other_df = pd.DataFrame(other_parsed.tolist()).fillna(0.0)
    all_other_elements = sorted(other_df.columns.tolist())

    composition = pd.concat([raw[MAIN_ELEMENTS], other_df[all_other_elements]], axis=1)
    composition = composition.astype(float)

    meta = raw[["合金名称", "相结构", "热处理状态", "高温温度(°C)"]].copy()
    
    targets = pd.DataFrame()
    for short_name, cn_col in TARGET_MAP.items():
        targets[short_name] = raw[cn_col].apply(to_float_or_nan)

    combined = pd.concat([meta, composition], axis=1)
    combined["Text"] = combined.apply(template_text_no_comp, axis=1)

    result = pd.DataFrame()
    result["alloy_name"] = meta["合金名称"]
    result["Text"] = combined["Text"]
    for col in composition.columns:
        result[col] = composition[col]
    for col in targets.columns:
        result[col] = targets[col]

    result.to_csv(OUTPUT_PATH, index=False)
    print(f"[{OUTPUT_PATH.name}] generation complete.")
    print(f"Saved {len(result)} rows -> {OUTPUT_PATH}")
    print(f"\nSample Text:\n  {result['Text'].iloc[0]}")
    print(f"  {result['Text'].iloc[1]}")
    return result

if __name__ == "__main__":
    preprocess()
