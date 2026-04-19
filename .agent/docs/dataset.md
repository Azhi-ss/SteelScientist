# 数据集说明

> 最后更新: 2026-03-31

## 下游回归任务数据

### 核心文件

```
datasets/data.csv   # 677 条钢铁样本，即论文回归任务所用的完整数据集
```

### 数据来源（论文 §2.1）

> "we filtered out a dataset with high quality for carbon steels (677 records) with their mechanical properties (UTS, YS and EL), and their corresponding composition (tabular data) and processing information (text data)."

**677 行 × 49 列**，是从文献中自动抽取并人工过滤后的高质量钢铁数据。

### 列结构

| 列名 | 类型 | 说明 |
|------|------|------|
| `DOIs` | str | 文献 DOI |
| `Table_topic` | str | 数据来源表格标题 |
| `Material` | str | 钢铁牌号 |
| `Tensile_value` | float | **抗拉强度 UTS (MPa)** ← 预测目标 |
| `Yield_value` | float | **屈服强度 YS (MPa)** ← 预测目标 |
| `Elongation_value` | float | **延伸率 EL (%)** ← 预测目标 |
| `Tensile_name/unit` | str | 单位标注（可丢弃）|
| `Yield/Elongation_name/unit` | str | 单位标注（可丢弃）|
| `H, B, C, N, ...` (36列) | float | **元素成分比例**（wt%），共 36 种元素 |
| `Text` | str | **工艺描述文本**（热轧、冷轧、热处理等） |

### 与源代码的映射关系

源代码 `regression/reg_v1.py` 期望读取的 `train_data.xlsx` 比 `data.csv` 多出以下列：

| 缺失列 | 代码用途 | 处理方案 |
|--------|---------|---------|
| `status` | `filter = data['status'] == 1` 过滤有效数据 | **补列全填 `1`**（data.csv 已是过滤后数据） |
| `actions` | `gen_text_embed(data, col_embed='actions')` 生成 768 维工艺嵌入特征 | **复制 `Text` 列**（两者语义高度重叠） |
| `Files` | drop_cols → 直接丢弃 | 补空列或忽略 |
| `problem` | drop_cols → 直接丢弃 | 补空列或忽略 |
| `title` | drop_cols → 直接丢弃 | 补空列或忽略 |
| `abstract` | drop_cols → 直接丢弃 | 补空列或忽略 |

> [!IMPORTANT]
> `actions` 列是模型的**第四路特征**（action_embed，768 维），不可省略。使用 `Text` 替代是合理近似，因为 `actions` 本质是从工艺文本中精提取的动作子片段。

### 特征一览（模型四路输入）

| 特征名 | 构造方式 | 维度 |
|--------|---------|------|
| `com` | 元素成分列 (wt%) | ~35 维 |
| `text_embed` | SteelBERT 对 `Text` 列的 [CLS] 嵌入 | 768 维 |
| `com_embed` | SteelBERT 对各元素名称嵌入的加权平均 | 768 维 |
| `action_embed` | SteelBERT 对 `actions` 列的 [CLS] 嵌入 | 768 维 |

### 数据统计

| 指标 | Tensile (MPa) | Yield (MPa) | Elongation (%) |
|------|-------------|------------|---------------|
| 均值 | 974.8 | 704.0 | 27.8 |
| 标准差 | 355.8 | 335.7 | 17.5 |
| 最小值 | 328.5 | 120.0 | 5.0 |
| 最大值 | 2183.0 | 1972.0 | 94.0 |

### 数据预处理脚本

使用 `scripts/prepare_regression_data.py` 将 `data.csv` 转换为 `reg_v1.py` 兼容的格式：

```bash
python scripts/prepare_regression_data.py
```

### 数据划分与评估 (基于源码与原文 SI.md Note S3)

根据论文原文，回归任务采用 **8:2** 的比例划分为训练集和验证集。目前脚本的处理逻辑为：

1. **`train_data.xlsx`**：包含 `data.csv` 的 **全量 677 条数据**。
   - `reg_v1.py` 内部会加载此文件，并根据 `split_ratio=0.8` 自动切分为 **541 条训练集** 和 **136 条验证集**。这是模型核心性能的评估主体。
2. **`text_test.xlsx`** & **`exp_test.xlsx`**：**占位数据集**。
   - ⚠️ **重要提示**：目前这两个文件仅提取全量数据的头部 20 条，旨在防止代码因文件缺失报错。由于其并未包含论文中所述的“2022-2023 独立文献数据”和“真实物理实验数据”，在这两个文件上的预测 R² 分数 **不具备** 论文所属的泛化效果验证。

## 分类任务数据

分类任务（`classification/cls.py`）需要的数据：

```
classification/dataset/
├── train.csv   # 需要 'sentence' 和 'label' 两列
├── val.csv
└── test.csv
```

> [!WARNING]
> 分类任务的标注数据**未随仓库发布**，需联系论文作者（ShaohanTian）获取，或自行从 `datasets/abstracts_doi/` 中的语料构建标注数据集。
