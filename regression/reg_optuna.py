#!/usr/bin/env python
# coding: utf-8
"""
reg_optuna.py — 基于 Optuna (TPE 贝叶斯) 的回归任务超参搜索

搜索空间 (v2 — 收窄版，防过拟合):
  - MLP 隐藏层数 (2~4 层) 及每层节点数 (64~512)
  - Dropout (0.1~0.5)
  - Batch Size (32, 64)
  - Learning Rate (1e-5 ~ 1e-3, log scale)
  - Weight Decay (1e-4 ~ 0.1, log scale)
  - 早停 patience

使用方法:
  cd regression && python reg_optuna.py --prop Tensile_value --n_trials 50
"""

import argparse
import csv
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import optuna
from optuna.trial import Trial

# 复用 reg_v1.py 中已有的组件
from reg_v1 import (
    device, load_data, gen_data_class, CustomSimpleDataset,
    CustomSimpleModel, eval_model, set_global_seed,
)


def build_model(trial: Trial):
    """让 Optuna 贝叶斯采样器自动决定网络架构 (v2 收窄版)"""

    # MLP 层数与每层宽度 (收窄: 最多4层, 最宽512)
    n_layers = trial.suggest_int('n_layers', 2, 4)
    simple_layer_list = []
    for i in range(n_layers):
        n_units = trial.suggest_categorical(f'n_units_l{i}', [64, 128, 256, 512])
        simple_layer_list.append(n_units)

    # CNN 通道数
    cnn_start = trial.suggest_categorical('cnn_start', [32, 64])
    concat_layer_list = [cnn_start, cnn_start, cnn_start // 2, 8, 4, 1]

    # Dropout (最低 0.1, 防止过拟合)
    dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.05)

    # 训练超参
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64])
    weight_decay = trial.suggest_float('weight_decay', 1e-4, 0.1, log=True)

    return simple_layer_list, concat_layer_list, dropout, lr, batch_size, weight_decay


def objective(trial: Trial, prop: str, train_data, test_data):
    """Optuna 目标函数: 最大化 val_r2"""

    simple_layer_list, concat_layer_list, dropout, lr, batch_size, weight_decay = build_model(trial)

    set_global_seed(seed=42)

    train_dataloader = DataLoader(
        CustomSimpleDataset(**gen_data_class(train_data)),
        batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_dataloader = DataLoader(
        CustomSimpleDataset(**gen_data_class(test_data)),
        batch_size=len(test_data)
    )

    model = CustomSimpleModel(
        simple_layer_list=simple_layer_list,
        concat_layer_list=concat_layer_list,
        seq_embed_con1d_list=[1, 1],
        seq_embed_fc_list=[32, 16],
        seq_embed_con2d_list=[1, 1],
        seq_embed_2d_fc_list=[32, 16],
        simple_layer_drop_prob=dropout,
        concat_layer_drop_prob=dropout,
    ).to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  betas=(0.9, 0.98), eps=1e-6, weight_decay=weight_decay)
    max_epochs = 500
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                   start_factor=1.0, end_factor=0.01,
                                                   total_iters=max_epochs)

    # Early stopping
    patience = 40
    no_improve = 0
    best_val_loss = float('inf')
    best_val_r2 = -1e5

    for epoch in range(max_epochs):
        model.train()
        for batch, inputs in enumerate(train_dataloader):
            y = inputs['labels'].unsqueeze(1)
            preds = model(**inputs)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            for batch, inputs in enumerate(val_dataloader):
                y = inputs['labels'].unsqueeze(1)
                preds = model(**inputs)
                val_loss = loss_fn(preds, y).item()
                val_r2 = eval_model(y, preds)['r2']

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

        # 让 Optuna 可以提前剪枝不值得跑下去的 trial
        trial.report(best_val_r2, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val_r2


def main():
    parser = argparse.ArgumentParser(description='Optuna Hyperparameter Search for Regression')
    parser.add_argument('--prop', type=str, default='Tensile_value',
                        choices=['Tensile_value', 'Yield_value', 'Elongation_value'],
                        help='Target property to optimize')
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Number of Optuna trials (default: 50)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f" Optuna TPE Bayesian Search")
    print(f" Property: {args.prop}")
    print(f" Trials:   {args.n_trials}")
    print(f" Device:   {device}")
    print(f"{'='*60}")

    # 加载数据（只做一次，所有 trial 共享）
    fes = ['com', 'com_embed', 'text_embed', 'action_embed']
    train_data, test_data = load_data(
        label='train_data', pred_prop=args.prop,
        fes=fes, split_ratio=0.8, seed=args.seed, perplexity=3
    )
    print(f"Data loaded: train={train_data.shape}, val={test_data.shape}")

    # 创建 Optuna study (最大化 val_r2)
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=50),
        study_name=f'steel_reg_{args.prop}'
    )

    study.optimize(
        lambda trial: objective(trial, args.prop, train_data, test_data),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    # 输出最优结果
    best = study.best_trial
    print(f"\n{'='*60}")
    print(f" Best Trial #{best.number}")
    print(f" Val R²: {best.value:.4f} ({best.value*100:.1f}%)")
    print(f" Params:")
    for k, v in best.params.items():
        print(f"   {k}: {v}")
    print(f"{'='*60}")

    # 保存结果到 CSV
    os.makedirs('./outputs/optuna/csvs', exist_ok=True)
    os.makedirs('./outputs/optuna/models', exist_ok=True)
    result_file = f'./outputs/optuna/csvs/optuna_{args.prop}.csv'
    df = study.trials_dataframe()
    df.to_csv(result_file, index=False)
    print(f"\nAll trials saved to: {result_file}")

    # 追加最优结果到主记录
    with open('./outputs/reg_model.csv', 'a+') as f:
        w = csv.writer(f)
        w.writerow([
            f'{args.prop}_optuna_best',
            f"val_r2={best.value:.4f}",
            f"layers={best.params.get('n_layers')}",
            f"dropout={best.params.get('dropout')}",
            f"lr={best.params.get('lr'):.6f}",
            f"batch={best.params.get('batch_size')}",
        ])


if __name__ == '__main__':
    main()
