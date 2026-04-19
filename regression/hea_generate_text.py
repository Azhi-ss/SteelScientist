"""
调用火山引擎豆包 API 为 HEA 数据生成学术英文描述，输出到 datasets/hea_llm_text.csv。

运行前只需设置一个环境变量：

    export ARK_API_KEY="your-api-key"

用法：
    python regression/hea_generate_text.py
    python regression/hea_generate_text.py --dry-run   # 只打印 prompt 不调用 API
    python regression/hea_generate_text.py --resume     # 断点续跑
"""

import os
import sys
import time
from argparse import ArgumentParser
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parent.parent
XLSX_PATH = REPO_ROOT / "datasets" / "hea-composition-performance.xlsx"
OUTPUT_PATH = REPO_ROOT / "datasets" / "hea_llm_text.csv"

MAIN_ELEMENTS = ["Co", "Cr", "Ni", "Al", "Ti", "Ta"]

SYSTEM_PROMPT = """\
You are a materials scientist writing for a metallurgy journal. \
Convert the alloy metadata below into exactly ONE academic English sentence.

Rules:
1. Only describe facts given — do NOT infer or predict any properties.
2. Use terminology common in steel and structural alloy literature: \
"face-centered cubic (FCC)", "ordered L1_2 precipitates" (not "L12"), \
"eta (η) phase", "B2 ordered phase", "Laves phase", "body-centered cubic (BCC)".
3. Describe composition, phase structure, heat treatment process, and test temperature.
4. Write in past tense, third person, similar to a methods section in Acta Materialia.
5. Output must be in English only. No Chinese characters. No bullet points.

Examples:
INPUT: Alloy: Co: 31.97 at%, Cr: 31.97 at%, Ni: 31.97 at%, Al: 2.04 at%, Ti: 2.04 at%
Phase structure: FCC+L12 | Heat treatment: 850°C×10h | Test temperature: 700°C
OUTPUT: A Co-31.97Cr-31.97Ni-2.04Al-2.04Ti (at.%) high-entropy alloy exhibiting a dual-phase \
microstructure of face-centered cubic (FCC) matrix with ordered L1_2 precipitates was aged \
at 850 °C for 10 h and mechanically tested at 700 °C.

INPUT: Alloy: Co: 33.3 at%, Cr: 33.3 at%, Ni: 33.3 at%
Phase structure: FCC | Heat treatment: 850°C×500h | Test temperature: 700°C
OUTPUT: An equiatomic CoCrNi medium-entropy alloy with a single face-centered cubic (FCC) \
phase was subjected to prolonged aging at 850 °C for 500 h and tested at 700 °C."""


def build_user_prompt(row: pd.Series) -> str:
    elements = []
    for el in MAIN_ELEMENTS:
        val = row.get(f"{el}(at%)", 0)
        if val and str(val).strip() != "-":
            try:
                if float(val) > 0:
                    elements.append(f"{el}: {val} at%")
            except (ValueError, TypeError):
                pass

    other = row.get("其他元素", "-")
    if other and str(other).strip() != "-":
        elements.append(str(other))

    phase = row.get("相结构", "-")
    phase = phase if (phase and str(phase).strip() != "-") else "unknown"

    ht = row.get("热处理状态", "-")
    ht = ht if (ht and str(ht).strip() != "-") else "as-received"

    temp = row.get("高温温度(°C)", "Room temperature")
    temp = temp if (temp and str(temp).strip() != "-") else "Room temperature"

    return (
        f"Alloy: {', '.join(elements)}\n"
        f"Phase structure: {phase}\n"
        f"Heat treatment: {ht}\n"
        f"Test temperature: {temp}°C"
    )


def generate_one(client: OpenAI, model: str, row: pd.Series) -> str:
    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        max_tokens=256,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(row)},
        ],
    )
    return resp.choices[0].message.content.strip()


def main():
    parser = ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="只打印 prompt，不调用 API")
    parser.add_argument("--resume", action="store_true", help="断点续跑：跳过已有 Text 的行")
    args = parser.parse_args()

    api_key = os.environ.get("ARK_API_KEY", "")
    base_url = "https://ark.cn-beijing.volces.com/api/v3"
    model = "glm-4-7-251222"

    if not args.dry_run and not api_key:
        print("ERROR: 请设置环境变量 ARK_API_KEY", flush=True)
        print("  export ARK_API_KEY='your-api-key'", flush=True)
        return

    raw = pd.read_excel(XLSX_PATH)
    print(f"Loaded {len(raw)} rows from {XLSX_PATH}", flush=True)

    # 断点续跑：加载已有结果，跳过报错行
    existing_texts = {}
    if args.resume and OUTPUT_PATH.exists():
        prev = pd.read_csv(OUTPUT_PATH)
        for _, r in prev.iterrows():
            txt = r.get("Text")
            if pd.notna(txt) and not str(txt).startswith("ERROR:"):
                # 使用 index + alloy_name 确保唯一性
                existing_texts[int(r["index"])] = txt
        print(f"Resumed: {len(existing_texts)} rows already generated successfully", flush=True)

    if not args.dry_run:
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=120)

    results = []

    def process_row(idx, row_data):
        alloy = row_data["合金名称"]
        if args.dry_run:
            prompt = build_user_prompt(row_data)
            return {"alloy_name": alloy, "Text": "", "index": idx, "status": "dry"}
        
        try:
            # print(f"  [DEBUG] Starting API call for {alloy} (row {idx+1})", flush=True)
            text = generate_one(client, model, row_data)
            return {"alloy_name": alloy, "Text": text, "index": idx, "status": "ok"}
        except Exception as e:
            return {"alloy_name": alloy, "Text": f"ERROR: {e}", "index": idx, "status": "error"}

    # 筛选需要运行的行
    to_process = []
    for i, row in raw.iterrows():
        alloy = row["合金名称"]
        if args.resume and i in existing_texts:
            results.append({"alloy_name": alloy, "Text": existing_texts[i], "index": i})
            print(f"[{i+1}/{len(raw)}] {alloy}: (cached)", flush=True)
            continue
        to_process.append((i, row))

    if not to_process:
        print("No new rows to process.", flush=True)
    else:
        print(f"Starting generation for {len(to_process)} rows with 5 threads...", flush=True)
        # 使用线程池并发执行
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_row = {executor.submit(process_row, i, row): (i, row["合金名称"]) for i, row in to_process}
            
            completed_count = 0
            for future in as_completed(future_to_row):
                i, alloy = future_to_row[future]
                res = future.result()
                results.append({"alloy_name": res["alloy_name"], "Text": res["Text"], "index": res["index"]})
                completed_count += 1
                
                if res["status"] == "ok":
                    print(f"[{completed_count+len(existing_texts)}/{len(raw)}] {alloy}: {res['Text'][:60]}...", flush=True)
                elif res["status"] == "error":
                    print(f"[{completed_count+len(existing_texts)}/{len(raw)}] {alloy}: {res['Text']}", flush=True)
                
                # 每 5 条排序并保存一次进度，减少 IO 频率
                if completed_count % 5 == 0:
                    pd.DataFrame(results).sort_values("index").to_csv(OUTPUT_PATH, index=False)

    out_df = pd.DataFrame(results).sort_values("index")
    out_df.to_csv(OUTPUT_PATH, index=False)

    success = (out_df["Text"] != "").sum()
    print(f"\nDone: {success}/{len(raw)} texts generated → {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
