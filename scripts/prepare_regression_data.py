#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd

INPUT_CSV = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'data.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'regression', 'datasets')

def prepare():
    print("=== 准备回归任务数据 (论文 8:2 划分策略) ===")
    df = pd.read_csv(INPUT_CSV)
    
    for col in ['Files', 'problem', 'title', 'abstract', 'Other_ele', 'Text_addition']:
        if col not in df.columns:
            df[col] = ''

    if 'status' not in df.columns:
        df['status'] = 1

    if 'actions' not in df.columns:
        df['actions'] = df['Text']

    element_cols = [c for c in df.columns if c in [
        'H','B','C','N','O','F','Na','Mg','Al','Si','P','S','Cl','Ca',
        'Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','As','Y','Zr',
        'Nb','Mo','Sn','Sb','La','Ce','Ta','W','Pb','Bi'
    ]]

    meta_cols = ['DOIs', 'Files', 'problem', 'status', 'Table_topic',
                 'title', 'abstract', 'Material',
                 'Tensile_name', 'Tensile_value', 'Tensile_unit',
                 'Yield_name', 'Yield_value', 'Yield_unit',
                 'Elongation_name', 'Elongation_value', 'Elongation_unit']

    tail_cols = ['Other_ele', 'Text_addition', 'Text', 'actions']
    final_cols = meta_cols + element_cols + tail_cols
    final_cols = [c for c in final_cols if c in df.columns]
    df = df[final_cols]

    for col in element_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    for col in ['Tensile_value', 'Yield_value', 'Elongation_value']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_path = os.path.join(OUTPUT_DIR, 'train_data.xlsx')
    df.to_excel(train_path, index=False)
    print(f"✅ train_data.xlsx: {len(df)} 条 (全量数据，将由 reg_v1.py 自动做 Train/Val 切分)")

if __name__ == '__main__':
    prepare()
