# 本地环境信息

> 最后更新: 2026-03-31

## 硬件

| 项目 | 详情 |
|------|------|
| GPU | NVIDIA GeForce RTX 4090 D (24GB VRAM) |
| GPU UUID | GPU-52e88ed2-c16f-2747-e062-4438d1910fb1 |

## 软件

| 项目 | 版本 |
|------|------|
| OS | Linux |
| Python | 3.10.12 (Miniconda) |
| pip | 23.1.2 |
| CUDA Driver | 已就绪 (nvidia-smi 可用) |

## 注意事项

- 4090D 的 24GB VRAM 足以进行 SteelBERTa (DeBERTa-v3-base, ~86M 参数) 的预训练和微调
- `pretrain/model_train.py` 默认 `per_device_train_batch_size=8`, `gradient_accumulation_steps=72`, FP16 训练
- 回归任务 `reg_v1.py` 的模型较轻量 (FC + CNN 混合)，显存不是瓶颈
