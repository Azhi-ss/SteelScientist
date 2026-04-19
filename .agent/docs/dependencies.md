# 依赖说明

> 最后更新: 2026-03-31 (版本已锁定为实际验证通过的精确版本)

## 安装

```bash
pip install -r requirements.txt
```

## 核心依赖一览

### 深度学习 & NLP

| 包 | 锁定版本 | 用途 | 使用位置 |
|----|---------|------|---------|
| torch | 2.1.0+cu118 | 神经网络训练 | 全局 |
| transformers | 4.39.3 | DeBERTa/BERT 模型加载 | 全局 |
| datasets | 4.8.4 | HF 数据集 | 全局 |
| tokenizers | 0.15.2 | BPE 分词器 | pretrain/, cls.py |
| huggingface-hub | 0.36.2 | 模型仓库 | 全局 |

### 数据处理

| 包 | 最低版本 | 用途 |
|----|---------|------|
| numpy | 1.26.4 | 数值计算 |
| pandas | 2.3.3 | 表格处理 |
| scikit-learn | 1.7.2 | 评估指标, t-SNE |
| scipy | 1.15.3 | 科学计算 |
| matplotlib | 3.10.8 | 绘图 |
| seaborn | 0.13.2 | 统计绘图 |
| openpyxl | 3.1.5 | 读写 .xlsx 文件 (回归模块必需) |

### 调参 & 日志

| 包 | 锁定版本 | 用途 |
|----|---------|------|
| ray | 2.54.1 | 分布式超参搜索 (回归) |
| wandb | 0.25.1 | 实验追踪 (分类) |
| tensorboardX | 2.6.4 | 训练可视化 (回归) |
| tqdm | 4.67.3 | 进度条 |

## 注意事项

> [!WARNING]
> **transformers ≥ 4.40 与 PyTorch 2.1.x 不兼容！** transformers 5.x 要求 PyTorch ≥ 2.4，4.40+ 会触发 `torch.utils._pytree` API 不匹配。当前已锁定 `transformers==4.39.3`。

- `torch 2.1.0+cu118` 为预装版本 (CUDA 11.8)，与 4090D 的驱动兼容
- `openpyxl` 已补装，回归模块读取 `.xlsx` 数据时必需
- `ray` 会拉取较多传递依赖 (click, colorama, stevedore, cmd2, cliff, prettytable 等)
- 验证安装: `python verify_dependencies.py`
- 验证权重加载: `python tests/test_weights_load.py`
