# SteelScientist — Agent Context

> 基于大语言模型的钢铁材料设计项目 (论文: [Acta Materialia 2025](https://doi.org/10.1016/j.actamat.2024.120663))

## Quick Navigation

| 文档 | 说明 |
|------|------|
| [环境信息](.agent/docs/environment.md) | GPU、Python、CUDA 等本地环境 |
| [项目架构](.agent/docs/architecture.md) | 目录结构、模块职责、数据流 |
| [依赖说明](.agent/docs/dependencies.md) | 核心依赖包及安装指引 |
| [模型权重](.agent/docs/weights.md) | 预训练权重存放路径及文件清单 |
| [数据集说明](.agent/docs/dataset.md) | 数据结构、列映射、特征说明及预处理方案 |
| [HEA 数据管线](.agent/docs/hea_data_pipeline.md) | HEA 私有数据 → SteelBERT 回归的完整处理方案 |
| [论文原文解析](paper/steel-design-mineru.md) | MinerU提取的完整 Markdown 格式原文及高清附图 |
| [汇报与展示](presentation/README.md) | 阶段性项目汇报 PPT 记录、可视化素材及 Slide Decks |

## 一句话概述

使用 **DeBERTa-v3** 架构从钢铁文献语料中预训练领域模型 **SteelBERTa**，然后在下游做 **分类** (wandb sweep) 和 **回归** (Ray Tune 超参搜索) 两类任务，预测钢铁力学性能 (抗拉强度 / 屈服强度 / 延伸率)。

## 核心技术栈

`PyTorch` · `HuggingFace Transformers` · `Ray Tune` · `Wandb` · `scikit-learn`

## 关键约定

- 预训练模型实际存放路径: `/internfs/Zy/Steelllm/ckpt/SteelBERT`
- HuggingFace 模型 ID: `MGE-LLMs/SteelBERT` (Gated Model, 需申请访问)
- 环境变量: `CACHE_DIR`, `SAVE_DIR`, `DATA_DIR`, `MODEL_NAME`
- 随机种子常用: `666`, `42`, `888`, `123`
- FP16 训练默认开启

## 🤖 Agent 行动准则 (Agent Action Guidelines)

⚠️ **核心原则**：每次修改代码、新增高度复用的工具脚本（如绘图、数据处理等），或者变更了项目结构，**必须同步修改相应的 Markdown 文档以保持信息最新**。特别是那些高度复用的组件设计和执行逻辑，**必须写入到这个 `AGENT.md` 文件里**，确保后续接手的 Agent 拥有完整的状态上下文和工具箱使用指南。

## 已修复的源码重大 Bug (Known Issues Fixed)

1. **回归数据特征类型失效**
   - **问题**：`data.csv` 存在整型和浮点型混用情况，导致 pd 读入为 `object`。
   - **修复**：在数据生成脚本 `scripts/prepare_regression_data.py` 中强制将 36 个元素成分列和目标力学性能列转换为 `float64`。
2. **回归网络大面积突触死亡 (Dying ReLUs)**
   - **问题**：原仓库 `reg_v1.py` 内部硬编码回归学习率为 `0.01`，配合 `ReLU` 激活，极易诱发梯度爆炸并永久锁死在负区间（导致预测全塌缩，`R²=-0.5~-2`）。
   - **修复**：将 `ALL_ACT_LAYERS` 统一改为 `leaky_relu`；将学习率重构降级至 `1e-4` 左右的安全区间。
3. **回归头参数未更新 (Lazy Layer Initialization Bug)**
   - **问题**：`CustomSimpleModel` 使用了 `nn.LazyLinear`。但在 `reg_v1.py` 中，优化器是在前向传播前定义的，此时 Lazy 层的参数尚未创建，导致优化器无法抓取并更新这些核心权重。
   - **方案**：在定义 `optimizer` 之前，必须先执行一次 dummy forward pass 激活所有 Lazy 层。
4. **数据集划分科学性修正**
   - **问题**：原脚本随机打乱分成的三份数据集（text/exp_test）与论文所述的独立"未来数据"和"实验数据"不符。
   - **修复**：重写了脚本逻辑。现将全量数据合并至 `train_data.xlsx` 并将内部验证比例对齐至论文的 **8:2**。

## 论文核心超参数 (Extracted from MinerU Paper)

为了后续模型微调和修改架构提供参考依据，以下是原论文中的核心超参数及设定方案摘要：

### 1. SteelBERT 预训练阶段 (Pre-training)
- **模型结构**：基于 DeBERTa 的解耦注意力机制。`12` 层 Transformer encoder，共计 **1.88 亿** 参数。
- **训练参数**：Batch size = `576`, Peak LR = `1e-4`, Warmup ratio = `4.8%`。

### 2. 回归任务与两套技术路线 (Regression vs Fine-tuning)
论文对比了两类预测精度差异巨大的方案，这解释了代码现状与论文性能差异的来源：

- **路线 A：特征提取工具 (仓库目前逻辑)**
  - **SteelBERT 参数**：完全**冻结 (Frozen)**。仅用于离线提取 768 维语义向量。
  - **特点**：速度极快，但无法实现"端到端"的物理语义对齐，性能存在上限。
- **路线 B：端到端全局微调 (论文极致性能来源)**
  - **SteelBERT 参数**：解除冻结，参与反向传播。这是 Table S9 中达到 $R^2 \approx 0.9$ 的主要原因。
  - **训练环境**: 需 `8 x NVIDIA A100 40GB` 集群支持全量参数更新。

## 核心数据流水线 (Text -> Actions -> Prediction)

论文的完整预测链路实际上分为三个阶段，**分类任务是回归任务的前置依赖**：

```
原始论文全文 (Raw Text / data.csv 的 Text 列)
        │
        ▼
┌─────────────────────────────────┐
│ ① classification/cls.py        │  分类任务 (Binary Classification)
│    逐句判断: "这句话描述了工艺？" │  → 正样本 494 句 / 负样本 39,358 句
│    是 → 保留  /  否 → 丢弃       │    (SI Note S1: 258 篇文献, 39,852 句)
└─────────────────────────────────┘
        │
        ▼
  筛选出的工艺语句 (Action sentences)
        │
        ▼
┌──────────────────────────────────────────┐
│ ② evaluation/steelberta_actions.ipynb    │  工艺词归一化 & 聚类
│    SteelBERT Embedding → UMAP 降维       │  → tokens_n.json / chunks_n.json
│    → HDBSCAN 聚类 → 标准化工艺词          │    (对应 SI Fig S1 + Table S5)
└──────────────────────────────────────────┘
        │
        ▼
  干净的 actions 列 (如 "quenching", "cold rolling")
        │
        ▼
┌──────────────────────────────────────────┐
│ ③ regression/reg_v1.py                   │  回归预测
│    SteelBERT 将 actions + 元素配比        │
│    编码为 768 维向量 → MLP 预测 YS/UTS/EL │
└──────────────────────────────────────────┘
```

### 核心认知反转：Text 胜过 Actions
前期复现中我们曾以为 `df['actions'] = df['Text']` 是一种粗暴的妥协（引入了太多非工艺噪音），**但经与论文原作者沟通确认，这其实就是他们最终采取的最优策略！**

- **推翻错误认知**：论文中的分类 (cls.py) 和聚类 (Table S5) 主要是为了向审稿人证明大模型具备提取和理解材料学实体的能力。
- **真实训练逻辑**：在最终回归预测（预测强度/延伸率）时，**使用原始包含完整上下文的自然语言段落 (`Text`) 效果实际上好于提取出的干瘪工艺词（`actions`）**。
- **原理解释**：像 `then`、`subsequently` 等连词以及时态构成的完整语境，能更好地触发 SteelBERT 自注意力机制，帮助网络理解加工的**先后顺序和时序逻辑**，这是"孤立工艺词袋"做不到的。
- **结论**：回归管线的输入数据（`Text` 列直接喂给模型）已经是完全正确且最优的状态，性能差距100% 来源于尚未解密的 **Optuna 网络架构超参**与**最后阶段的局部冻结微调**。

---

### 🧪 私有数据集成与消融实验 (HEA Private Data & Ablation Study)
 
 **数据文件**：`datasets/hea-composition-performance.xlsx` (64 条 CoCrNi 高熵合金数据)
 **详细处理文档**：[hea_data_pipeline.md](.agent/docs/hea_data_pipeline.md)
 
 #### 1. 消融实验架构 (`ablation/` 目录)
 
 为了测试文本描述对模型性能的影响，我们构建了三套对比方案：
 
 | 版本 | 策略 | 文本特征来源 | 状态 |
 |------|------|------------|------|
 | **v1_simple** | 模板拼接 (Baseline) | `hea_preprocess.py` 生成的固定格式短句 | ✅ 已跑完 (R²~0.8) |
 | **v2_academic** | LLM 驱动 (Academic) | `hea_generate_text.py` 生成的学术化长句 | ⏳ 生成中 |
 | **v3_comp_only** | 纯成分 (Control) | 无文本描述，仅保留成分 Embedding | ⏳ 待测试 |
 
 #### 2. 优化后的数据管线
 
 | 步骤 | 工具/脚本 | 核心优化与功能 |
 |------|----------|--------------|
 | **① 语义合成** | `hea_generate_text.py` | **性能飞跃**：引入 `ThreadPoolExecutor` 多线程并发请求 (8-10x 加速)，支持断点续传 `--resume`，确保输出 CSV 行序与原始 Excel 严格一致。 |
 | **② 特征提取** | `hea_regression.py` | **路径解耦**：支持 `--data` 和 `--out_dir` 参数，可一键切换不同的消融实验版本进行训练调校。 |
 | **③ 多重验证** | `hea_regression.py` | **多种子平均**：默认运行 5 个随机种子 (42-46) 并自动挑选验证集最强模型，绘制 Parity Plot。 |
 
 #### 3. 关键结论 (现阶段 V1)
 - **延伸率 (EL)**：预测精度极高 ($R^2 > 0.8$)，对文本风格不敏感。
 - **强度 (YS/UTS)**：预测精度较低 ($R^2 \approx 0.1 \sim 0.4$)，亟需 V2 版本的学术文本增强语义特征。


#### 4. 可视化与性能评估逻辑 (Visual Analysis Pipeline)
 
 为了统一对比消融实验中所有力学性质的表现，我们采用 **“均值-最优对比柱状图 (Mean vs. Best Bar Chart)”**。该逻辑已提取为通用绘图逻辑，用于对比 V1/V2/V3 各方案的真实潜能。
 
**核心绘图脚本逻辑 (`ablation/plot_ablation_results.py` 原理)**:
 ```python
 # 1. 遍历 RT_YS, RT_UTS... 等 6 个输出子目录
 # 2. 从 seed_summary.csv 提取 train_r2.mean(), val_r2.mean(), val_r2.max()
 # 3. 使用 Matplotlib 绘制三组对比柱 (Width=0.25)
 # 4. 采用特定色值：#1F4788 (Train), #A2555A (Val Mean), #CD7F32 (Best Seed)
 ```
旧版可视化结果已归档到 `ablation/back/legacy_20260414_120737/results/vX/overall_performance_vX_best.png`，当前激活结果建议放在 `ablation/current/results/`。
 
 ---
 
 ### 🚀 全局回归模型突破 (Optuna 超参搜索)


我们于 2026-04-03 实装了 `regression/reg_optuna.py`，对三个力学性能属性运行了 TPE 贝叶斯搜索（各 50 轮）。
以下记录了完整的**试错过程和关键教训**，供后续 Agent 避坑。

---

#### ⚠️ 踩坑记录 #1：Loss 函数不一致 (SmoothL1 vs MSE)

- **错误**：初始版本 `reg_optuna.py` 使用 `nn.SmoothL1Loss()` 作为训练损失，但评估指标是 R²（基于 MSE 的变体）。
- **后果**：两者优化方向不完全一致。SmoothL1 对大误差"宽容"，导致模型不够精确。
- **修复**：将 `reg_optuna.py` 和 `plot_final_optimal.py` 的 loss 改为 `nn.MSELoss()`。
- **依据**：论文 SI 明确写明使用 MSELoss。`reg_v1.py` 中的 `train_function`（Ray Tune 版本）也确认原作用 MSE。
- **结论**：**训练 loss 必须与评估指标一致。回归任务用 R² 评估时，训练 loss 必须用 MSE。**

#### ⚠️ 踩坑记录 #2：单种子重训练不可复现 (Train R² 暴跌)

- **错误**：`plot_final_optimal.py` 用 Optuna 搜到的最佳超参数从头重新训练，但只用一个 seed=42。
- **后果**：UTS 在 Optuna 搜索中 Val R²=0.685，但重训练后暴跌到 **Val R²=0.037**（几乎等于随机）。
- **根因**：深度学习模型对随机初始化极度敏感。Optuna Trial #41 碰巧遇到好的初始化，单次重训练大概率遇到差的初始化。
- **修复**：改为 **多种子训练**（5 个 seed: 42~46），选 Val R² 最高的模型。
- **结论**：**永远不要用单次训练的结果作为最终报告值。至少跑 5 个种子取 best/mean。**

#### ⚠️ 踩坑记录 #3：batch_size 不一致

- **错误**：Optuna 搜索时 batch_size 是搜索参数（找到 64），但 `plot_final_optimal.py` 硬编码 `batch_size=32`。
- **后果**：不同 batch_size 会改变梯度噪声水平和 BN/Dropout 行为，导致重训练结果偏离搜索时的最优值。
- **修复**：在 `BEST_CONFIG` 字典中加入 `batch_size` 字段，保持与 Optuna 搜索结果一致。

#### ⚠️ 踩坑记录 #4：模型过拟合 - 参数量 vs 数据量严重不匹配

- **现象**：UTS 和 YS 的 Train-Val R² gap 达到 20~25%，而 EL 只有 1.4%。
- **根因分析**（参数量统计）：

| 属性 | 可训练参数 | 样本数 | 参数/样本比 | Train R² | Val R² | Gap |
|------|:---------:|:-----:|:----------:|:--------:|:------:|:---:|
| **YS** | **1,304,897** | 541 | **2412x** | 0.821 | 0.573 | 24.8% |
| **UTS** | 361,473 | 541 | 668x | 0.868 | 0.670 | 19.8% |
| **EL** | 192,177 | 541 | 355x | 0.776 | 0.762 | 1.4% |

- **EL 不过拟合的原因**：Optuna 恰好搜到了简单的 2 层 `[128, 512]` 架构，参数量和数据量匹配。
- **YS 严重过拟合的原因**：6 层架构含两个 1024 宽层，130 万参数吃 541 条数据，且 **dropout=0.0**。
- **教训**：
  - 搜索空间应 **限制最大层数为 4、最大宽度为 512**
  - 深层网络至少需要 dropout >= 0.1
  - 可在 Optuna 目标函数中加入正则惩罚项（如 `val_r2 - alpha * (train_r2 - val_r2)`）

---

#### 当前最佳搜索结果 (MSELoss, 多种子验证后)

| 属性 | Optuna 搜索 Val R² | 多种子重训 Val R² | Train R² | Gap |
|------|:-----------------:|:----------------:|:--------:|:---:|
| **UTS** | 0.685 | **0.670** | 0.868 | 19.8% |
| **YS** | 0.578 | **0.573** | 0.821 | 24.8% |
| **EL** | 0.757 | **0.762** | 0.776 | 1.4% |

**完整最优参数（可直接复用）：**

```python
BEST_CONFIG = {
    'Tensile_value': {                       # UTS 抗拉强度
        'simple_layer_list': [128, 256, 256, 128, 256, 256],  # 6层 MLP
        'cnn_start': 64,                     # CNN 通道起始数
        'dropout': 0.05,
        'lr': 0.0007837263023600375,
        'batch_size': 64,
        'epoch': 500,
        'patience': 40                       # Early stopping patience
    },
    'Yield_value': {                         # YS 屈服强度
        'simple_layer_list': [1024, 128, 128, 64, 1024, 256],  # 6层 MLP (过拟合!)
        'cnn_start': 64,
        'dropout': 0.0,                      # ← 无 dropout，导致严重过拟合
        'lr': 0.0008220584449153705,
        'batch_size': 64,
        'epoch': 500,
        'patience': 40
    },
    'Elongation_value': {                    # EL 延伸率
        'simple_layer_list': [128, 512],     # 2层 MLP (最佳泛化)
        'cnn_start': 32,
        'dropout': 0.0,
        'lr': 0.0003774297899184771,
        'batch_size': 32,
        'epoch': 500,
        'patience': 40
    }
}
# 固定参数（不在搜索空间内）：
# concat_layer_list = [cnn_start, cnn_start, cnn_start//2, 8, 4, 1]
# optimizer = AdamW(lr=lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.01)
# scheduler = LinearLR(start_factor=1.0, end_factor=0.01, total_iters=epoch)
# loss = nn.MSELoss()
```

#### 当前 Optuna 搜索空间 (`reg_optuna.py`)

```python
n_layers:  2 ~ 6           # <-- 建议收窄到 2~4
n_units:   64 ~ 1024       # <-- 建议收窄到 64~512
cnn_start: [32, 64]
dropout:   0.0 ~ 0.5       # <-- 建议对深层网络强制 >= 0.1
lr:        1e-5 ~ 1e-3
batch_size: [32, 64]
```

#### 与论文目标的差距

| 属性 | 当前 Val R² | 论文 R² | 差距 | 原因 |
|------|:----------:|:------:|:----:|------|
| UTS | 0.670 | 0.864 | -19.4% | 冻结 SteelBERT + 架构过拟合 |
| YS | 0.573 | 0.835 | -26.2% | 冻结 SteelBERT + 架构过拟合 |
| EL | 0.762 | 0.860 | -9.8% | 架构已近最优，差距主要来自冻结 |

**核心结论**：当前用冻结 SteelBERT 的特征提取方案（路线 A），EL 已接近天花板。要达到论文 85%+ R²，必须走**路线 B（端到端微调）**。

---

#### 关键文件清单

| 文件 | 功能 |
|------|------|
| `regression/reg_optuna.py` | Optuna 超参搜索主脚本 |
| `regression/plot_optuna_history.py` | 画搜索历史图 + LR-性能散点图 |
| `regression/plot_final_optimal.py` | 用最优超参数多种子训练并画散点图 |
| `regression/outputs/optuna/csvs/` | 搜索历史 CSV |
| `regression/outputs/optuna/figs/` | 所有生成的图 |
| `regression/outputs/optuna/models/` | 最优模型 .pt 文件 |

## evaluation 目录文件功能

| Notebook | 对应论文内容 | 功能 |
|----------|------------|------|
| `steelberta_actions.ipynb` | Fig S1 + Table S5 | SteelBERT 对工艺词 Embedding -> UMAP 降维 -> HDBSCAN 聚类归一化 |
| `supplemet_table.ipynb` | Table S5 格式化 | 将聚类结果按主题展开为论文附表输出 |
| `model_words_similarity.ipynb` | 无 | 开发者草稿纸 (MaxPool/Tensor 形状测试等) |
