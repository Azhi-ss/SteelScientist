# 模型权重

> 最后更新: 2026-03-31

## 存放路径

```
/internfs/Zy/Steelllm/ckpt/SteelBERT/
```

## HuggingFace 来源

- **模型 ID**: `MGE-LLMs/SteelBERT`
- **页面**: https://huggingface.co/MGE-LLMs/SteelBERT
- **访问方式**: Gated Model（需登录 HF 账号并申请权限）

## 文件清单

| 文件 | 大小 | 说明 |
|------|------|------|
| `config.json` | 866 B | 模型架构配置 (DeBERTa-v3, 12层, 12头, 188M参数) |
| `pytorch_model.bin` | 738 MB | 核心预训练权重 |
| `optimizer.pt` | 1.48 GB | 优化器状态 (可用于断点续训) |
| `tokenizer.json` | 8.52 MB | 钢铁领域专用分词器 (词表 128,100) |
| `tokenizer_config.json` | 412 B | 分词器配置 |
| `special_tokens_map.json` | 173 B | 特殊 token 映射 |
| `scheduler.pt` | 627 B | 学习率调度器状态 |
| `rng_state.pth` | 14.6 KB | 随机数生成器状态 |
| `trainer_state.json` | 51.2 KB | Trainer 训练状态日志 |
| `training_args.bin` | 3.96 KB | 训练参数配置 |
| `README.md` | 4.74 KB | 模型说明文档 |

## 代码中的引用

代码中原本使用相对路径 `./../model_saved/checkpoint-140000`，已统一修改为绝对路径：

- `classification/cls.py` 第 54 行
- `regression/reg_v1.py` 第 42 行

## 模型规格摘要

- **架构**: DeBERTa-v3 (Disentangled Attention)
- **参数量**: 188M
- **Transformer 层数**: 12
- **注意力头数**: 12
- **最大序列长度**: 512 tokens
- **词表大小**: 128,100
- **预训练任务**: MLM (Masked Language Modeling, mask 比例 15%)
- **预训练语料**: 420万材料科学摘要 + 5.5万钢铁全文文章 ≈ 9.6亿词
- **原始训练硬件**: 8× NVIDIA A100 40GB, 训练 840 小时
