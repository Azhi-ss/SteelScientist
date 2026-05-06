# 数据集说明

> 最后更新: 2026-05-06

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
| `actions` | 旧版 `reg_v1.py` 的兼容字段 | 若旧脚本要求该列，直接复制 `Text`；后续建模不再把它作为有效独立特征 |
| `Files` | drop_cols → 直接丢弃 | 补空列或忽略 |
| `problem` | drop_cols → 直接丢弃 | 补空列或忽略 |
| `title` | drop_cols → 直接丢弃 | 补空列或忽略 |
| `abstract` | drop_cols → 直接丢弃 | 补空列或忽略 |

> [!IMPORTANT]
> 相关消融已证实 `actions` / `action_embed` 没有带来有效增益。保留 `actions` 仅为兼容旧脚本输入 schema；推荐模型输入以 `Text` 和成分特征为主。

### 特征一览（推荐有效输入）

| 特征名 | 构造方式 | 维度 |
|--------|---------|------|
| `com` | 元素成分列 (wt%) | ~35 维 |
| `text_embed` | SteelBERT 对 `Text` 列的 [CLS] 嵌入 | 768 维 |
| `com_embed` | SteelBERT 对各元素名称嵌入的加权平均 | 768 维 |

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

### Labelled Clusters 新检索数据

新增原始文件：

```
datasets/steel_labelled_clusters/raw/Steel database with labelled clusters.xlsx
```

该文件包含 `3234` 条带工艺聚类标签的钢铁性能记录。使用以下脚本生成去重清洗版 CSV 和固定 8:2 回归划分：

```bash
python scripts/prepare_labelled_clusters_data.py
```

输出文件：

```
datasets/steel_labelled_clusters/full_with_duplicates.csv  # 原始映射后全量 3234 行
datasets/steel_labelled_clusters/clean.csv                 # 严格去重后 1943 行
datasets/steel_labelled_clusters/train.csv                 # 8:2 训练集，1554 行
datasets/steel_labelled_clusters/val.csv                   # 8:2 验证集，389 行
datasets/steel_labelled_clusters/duplicate_rows.csv        # 重复行审计明细
datasets/steel_labelled_clusters/duplicate_groups.csv      # 重复组审计汇总
datasets/steel_labelled_clusters/reg_v1_train_data.xlsx    # reg_v1 兼容 Excel 母表
datasets/steel_labelled_clusters/README.md                 # 数据目录说明
```

清洗规则：

- **严格去重**：按 `Material + Text + 36 元素成分 + Tensile/Yield/Elongation + cluster_number + cluster_label` 去重，`3234` 行变为 `1943` 行。
- `Name` → `Material`。
- `Processing condition` → `Text`；`actions` 仅作为兼容字段同步复制，不作为有效独立特征。
- `(Ultimate) Tensile strength (MPa)` → `Tensile_value`。
- `Yield strength (MPa)` → `Yield_value`。
- `Ductility (%)` → `Elongation_value`。
- 原始存在的元素列转为 `float64`；原始缺失的标准 36 元素列补 `0.0`。
- 保留 `source_entry`、`cluster_number`、`cluster_label`、`cluster_number_0_to_11` 作为聚类元数据。
- 使用固定 `seed=42` 随机划分为 8:2 训练/验证集；去重后 train/val 严格键无重叠。

最新回归训练入口：

```bash
python regression/steel_labelled_clusters_regression.py \
  --target Tensile_value Yield_value Elongation_value
```

该入口复用 `regression/hea_regression.py` 的 SteelBERT 冻结特征提取 + 三路门控融合回归模型，但适配为 36 个钢铁元素列和 `Tensile_value` / `Yield_value` / `Elongation_value` 三个目标。训练输出位于：

```
regression/outputs/steel_labelled_clusters/
```

首轮训练结果 (`seed=42..46`, 300 epochs, patience=30)：

| 目标 | Best seed | Train R² | Val R² | Val RMSE | Val MAE |
|------|-----------|----------|--------|----------|---------|
| `Tensile_value` | 42 | 0.9506 | 0.8931 | 106.5751 | 67.0210 |
| `Yield_value` | 46 | 0.9485 | 0.8785 | 111.4556 | 67.7179 |
| `Elongation_value` | 46 | 0.9482 | 0.8618 | 2.3673 | 1.6446 |

去除 `raw_composition` 的两路消融入口：

```bash
python regression/steel_labelled_clusters_regression.py \
  --feature_mode text_ele \
  --target Tensile_value Yield_value Elongation_value
```

消融结果 (`seed=42..46`, 300 epochs, patience=30)：

| 目标 | Full Val R² | Text+Element Val R² | 差值 |
|------|-------------|---------------------|------|
| `Tensile_value` | 0.8931 | 0.8872 | -0.0059 |
| `Yield_value` | 0.8785 | 0.8838 | +0.0053 |
| `Elongation_value` | 0.8618 | 0.8572 | -0.0046 |

结论：`raw_composition` 分支贡献很小；去掉后 YS 略升，UTS/EL 略降，整体差异小于 `0.006`。当前默认仍保留 full 模型，`text_ele` 可作为简化模型候选。

推理示例：

```bash
./scripts/run_steel_labelled_clusters_inference.sh
```

默认使用：

```
input : datasets/steel_labelled_clusters/inference_sample.csv
ckpt  : /internfs/Zy/Steelllm/ckpt/steel_labelled_clusters_regression/
output: regression/outputs/steel_labelled_clusters/inference_predictions.csv
```

推理输入需要包含 `Text` 和 36 个标准元素列；若包含真实目标列，会原样保留到输出中用于对比。

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
