# 项目架构

> 最后更新: 2026-04-07

## 目录结构

```
SteelScientist/
│
├── pretrain/                       # ── 预训练流水线 (4 步) ──
│   ├── run.sh                      #   统一入口脚本
│   ├── json_combination.py         #   1. 合并摘要+全文语料
│   ├── corpus_normalize.py         #   2. 语料规范化
│   ├── tokenizer_train.py          #   3. 训练分词器
│   ├── tokens_count.py             #   4. 分词落盘
│   ├── model_train.py              #   5. MLM 预训练 (DeBERTa-v3)
│   └── traing_test.ipynb           #   草稿 notebook
│
├── classification/                 # ── 分类任务 ──
│   ├── cls.py                      #   Wandb sweep 多模型对比
│   ├── run.sh                      #   批量运行
│   └── dataset/                    #   (需自备 train/val/test.csv)
│
├── regression/                     # ── 回归任务 ──
│   ├── reg_v1.py                   #   原始钢铁回归 (FC+CNN, Ray Tune)
│   ├── reg_optuna.py               #   Optuna 超参搜索版
│   ├── plot_optuna_history.py      #   Optuna 结果可视化
│   ├── plot_final_optimal.py       #   最优超参多种子重训练
│   │
│   ├── hea_preprocess.py           #   HEA 数据清洗: xlsx → 成分对齐 → CSV
│   ├── hea_generate_text.py        #   HEA 文本合成: 调用 LLM API 生成学术描述
│   ├── hea_regression.py           #   HEA 回归管线: SteelBERT 特征 → MLP
│   │
│   ├── datasets/                   #   回归专用数据
│   │   └── train_data.xlsx         #     原始钢铁训练数据 (677 条)
│   └── outputs/                    #   训练产出
│       ├── reg_v1/                 #     reg_v1.py 的模型/图/预测
│       ├── optuna/                 #     Optuna 搜索结果 (最新)
│       ├── optuna_v1_backup/       #     Optuna 旧版备份
│       ├── hea/                    #     HEA 回归结果
│       │   └── HT_YS/             #       高温屈服强度 baseline
│       └── reg_model.csv           #     汇总指标
│
├── evaluation/                     # ── 评估 Notebooks ──
│   ├── steelberta_actions.ipynb    #   工艺词聚类展示，不作为当前回归有效特征
│   ├── supplemet_table.ipynb       #   论文附表 S5 格式化
│   └── model_words_similarity.ipynb#   词相似度 / 开发草稿
│
├── datasets/                       # ── 共享数据集 ──
│   ├── data.csv                    #   主数据 (~727KB, 原始论文数据)
│   ├── full_text_doi.csv           #   全文 DOI 列表 (~1.5MB)
│   ├── archive.7z                  #   压缩数据包
│   ├── hea-composition-performance.xlsx  # HEA 原始数据 (64 条)
│   ├── hea_with_text.csv           #   HEA 清洗后 (模板文本版)
│   ├── hea_llm_text.csv            #   HEA 清洗后 (LLM 文本版, 待生成)
│   ├── abstracts_doi/              #   按类型分的摘要 DOI
│   │   ├── article.csv
│   │   ├── meeting.csv
│   │   └── patent.csv
│   └── figures/                    #   数据可视化
│       ├── properties_boxplot.png
│       └── properties_distribution.png
│
├── scripts/                        # ── 工具脚本 ──
│   └── prepare_regression_data.py  #   回归数据预处理 (类型修复)
│
├── tests/                          # ── 测试 ──
│   ├── test_pytorch_env.py
│   ├── test_cls.py
│   ├── test_reg_v1.py
│   ├── test_reg_v1_standalone.py
│   └── test_weights_load.py
│
├── .agent/                         # ── Agent 上下文文档 ──
│   ├── AGENT.md                    #   项目总览 + 踩坑记录
│   └── docs/
│       ├── architecture.md         #   本文件
│       ├── weights.md              #   模型权重路径
│       ├── dataset.md              #   数据集说明
│       ├── dependencies.md         #   依赖说明
│       ├── environment.md          #   环境信息
│       └── hea_data_pipeline.md    #   HEA 处理方案设计
│
├── paper/                          #   论文原文 (MinerU 提取)
├── requirements.txt                #   Python 依赖
├── verify_dependencies.py          #   依赖验证
└── README.md
```

## 四大核心模块

### 1. 预训练 (`pretrain/`)

- **模型架构**: `microsoft/deberta-v3-base` 的 MLM (Masked Language Modeling)
- **流程**: 语料合并 → 规范化 → 分词器训练 → 分词落盘 → 模型预训练
- **输出**: `{save_dir}/model_saved/checkpoint-*`, `{save_dir}/tokenizer_saved/`
- **关键超参**: batch=8, grad_accum=72, lr=1e-4, warmup_steps=10000

### 2. 分类 (`classification/`)

- **任务**: 钢铁文献的工艺句判别 (二分类)
- **框架**: HuggingFace Trainer + Wandb Sweep (grid search)
- **支持模型**: steelberta / scibert / matscibert / bert
- **评估指标**: accuracy, f1, precision, recall

### 3. 回归 — 原始钢铁 (`regression/reg_v1.py`, `reg_optuna.py`)

- **任务**: 预测钢铁力学性能 (UTS / YS / EL)
- **模型**: FC + 1D-CNN + 2D-CNN 混合网络 (`CustomSimpleModel`)
- **特征**: 成分 wt% + text_embed(768d) + ele_embed(768d) + t-SNE(3d)；`action_embed` 已证实无有效增益，仅保留旧脚本兼容说明
- **数据**: `regression/datasets/train_data.xlsx` (677 条, ~35 个元素列)
- **调参**: Ray Tune / Optuna + Wandb

### 4. 回归 — HEA 私有数据 (`regression/hea_*.py`)

- **任务**: 预测 CoCrNi 高熵合金力学性能 (6 个目标)
- **数据**: `datasets/hea-composition-performance.xlsx` (64 条)
- **管线**:

```
hea-composition-performance.xlsx
    │
    ├── hea_preprocess.py ──→ hea_with_text.csv     (成分对齐 + 模板文本)
    ├── hea_generate_text.py ──→ hea_llm_text.csv   (LLM API 生成文本)
    │
    └── hea_regression.py                            (SteelBERT 特征提取 → MLP 回归)
            ├── text_embed (768d)  ─┐
            └── ele_embed  (768d)  ─┴→ 1536d → MLP [128, 64] → 预测值
```

- **模型**: 轻量 2 层 MLP (SteelBERT 完全冻结)
- **特征**: text_embed(768d) + ele_embed(768d) = 1536d
- **配置**: lr=3e-4, dropout=0.2, batch=16, patience=30, 5 种子训练

## 数据流总览

```
钢铁文献语料 (JSON)
    │  pretrain/
    ▼
SteelBERT (DeBERTa-v3, 188M params, /internfs/Zy/Steelllm/ckpt/SteelBERT)
    │
    ├──────────────────────┬──────────────────────────┐
    │                      │                          │
    ▼                      ▼                          ▼
classification/        regression/               regression/hea_*.py
文本 → 工艺句判别      钢铁文本+成分 → YS/UTS/EL   HEA 成分+相结构 → YS/UTS/EL
(Wandb sweep)         (677 条, Optuna)           (64 条, 轻量 MLP)
```
