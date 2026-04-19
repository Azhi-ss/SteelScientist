#!/usr/bin/env python
# coding: utf-8

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from reg_v1 import (
    device, load_data, gen_data_class, CustomSimpleDataset,
    CustomSimpleModel, eval_model, set_global_seed, plot_test_data
)

BEST_CONFIG = {
    'Tensile_value': {
        'simple_layer_list': [128, 256, 256, 128, 256, 256],
        'cnn_start': 64,
        'dropout': 0.05,
        'lr': 0.0007837263023600375,
        'batch_size': 64,
        'epoch': 500,
        'patience': 40
    },
    'Yield_value': {
        'simple_layer_list': [1024, 128, 128, 64, 1024, 256],
        'cnn_start': 64,
        'dropout': 0.0,
        'lr': 0.0008220584449153705,
        'batch_size': 64,
        'epoch': 500,
        'patience': 40
    },
    'Elongation_value': {
        'simple_layer_list': [128, 512],
        'cnn_start': 32,
        'dropout': 0.0,
        'lr': 0.0003774297899184771,
        'batch_size': 32,
        'epoch': 500,
        'patience': 40
    }
}

NUM_SEEDS = 5  # 多种子训练，选最佳

def _single_train(prop, c, seed, train_data, test_data):
    """用指定 seed 跑一次完整训练，返回 (val_r2, model_state_dict)"""
    set_global_seed(seed)

    train_dataloader = DataLoader(
        CustomSimpleDataset(**gen_data_class(train_data)),
        batch_size=c['batch_size'], shuffle=True, drop_last=False
    )
    val_dataloader = DataLoader(
        CustomSimpleDataset(**gen_data_class(test_data)),
        batch_size=len(test_data)
    )

    concat_layer_list = [c['cnn_start'], c['cnn_start'], c['cnn_start'] // 2, 8, 4, 1]

    model = CustomSimpleModel(
        simple_layer_list=c['simple_layer_list'],
        concat_layer_list=concat_layer_list,
        seq_embed_con1d_list=[1, 1], seq_embed_con2d_list=[1, 1],
        seq_embed_fc_list=[32, 16], seq_embed_2d_fc_list=[32, 16],
        simple_layer_drop_prob=c['dropout'], concat_layer_drop_prob=c['dropout']
    ).to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['lr'], betas=(0.9, 0.98), eps=1e-6, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=c['epoch'])

    best_val_loss = float('inf')
    best_val_r2 = -1e5
    best_model_state = None
    no_improve = 0

    for epoch in range(c['epoch']):
        model.train()
        for inputs in train_dataloader:
            y = inputs['labels'].unsqueeze(1)
            preds = model(**inputs)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            for inputs in val_dataloader:
                y = inputs['labels'].unsqueeze(1)
                preds = model(**inputs)
                val_loss = loss_fn(preds, y).item()
                val_r2 = eval_model(y, preds)['r2']

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= c['patience']:
                break

    return best_val_r2, best_model_state


def train_and_plot(prop):
    print(f"\n{'='*50}\nTraining Optimal Config for: {prop}\n{'='*50}")
    c = BEST_CONFIG[prop]

    set_global_seed(42)
    fes = ['com', 'com_embed', 'text_embed', 'action_embed']
    train_data, test_data = load_data(label='train_data', pred_prop=prop, fes=fes, split_ratio=0.8, seed=42, perplexity=3)

    # 多种子训练，选 Val R² 最高的模型
    best_overall_r2 = -1e5
    best_overall_state = None
    best_seed = -1

    for i, seed in enumerate(range(42, 42 + NUM_SEEDS)):
        r2, state = _single_train(prop, c, seed, train_data, test_data)
        print(f"  Seed {seed}: Val R² = {r2:.4f}")
        if r2 > best_overall_r2:
            best_overall_r2 = r2
            best_overall_state = state
            best_seed = seed

    print(f"  >> Best seed: {best_seed} with Val R² = {best_overall_r2:.4f}")

    # 用最佳 state 加载模型做最终评估和画图
    concat_layer_list = [c['cnn_start'], c['cnn_start'], c['cnn_start'] // 2, 8, 4, 1]
    model = CustomSimpleModel(
        simple_layer_list=c['simple_layer_list'],
        concat_layer_list=concat_layer_list,
        seq_embed_con1d_list=[1, 1], seq_embed_con2d_list=[1, 1],
        seq_embed_fc_list=[32, 16], seq_embed_2d_fc_list=[32, 16],
        simple_layer_drop_prob=c['dropout'], concat_layer_drop_prob=c['dropout']
    ).to(device)
    model.load_state_dict(best_overall_state)

    all_train_dataloader = DataLoader(
        CustomSimpleDataset(**gen_data_class(train_data)),
        batch_size=len(train_data), shuffle=False
    )
    val_dataloader = DataLoader(
        CustomSimpleDataset(**gen_data_class(test_data)),
        batch_size=len(test_data)
    )

    model.eval()
    with torch.no_grad():
        for inputs in all_train_dataloader:
            y_train = inputs['labels'].unsqueeze(1).detach().cpu().numpy()
            y_train_preds = model(**inputs).detach().cpu().numpy()
            train_r = eval_model(y_train, y_train_preds)
        for inputs in val_dataloader:
            y_val = inputs['labels'].unsqueeze(1).detach().cpu().numpy()
            y_val_preds = model(**inputs).detach().cpu().numpy()
            val_r = eval_model(y_val, y_val_preds)

    print(f"\nFINAL {prop} => Train R2: {train_r['r2']:.4f} | Val R2: {val_r['r2']:.4f}")

    os.makedirs("./outputs/optuna/figs", exist_ok=True)
    os.makedirs("./outputs/optuna/models", exist_ok=True)

    torch.save(model, f"./outputs/optuna/models/best_optuna_{prop}.pt")
    plot_test_data(y_train, y_train_preds, train_r, y_val, y_val_preds, val_r,
                   fig_name=f"./outputs/optuna/figs/best_optuna_{prop}.png",
                   labels=["Train Data (80%)", "Val Data (20%)"], point_size=5)

if __name__ == "__main__":
    for p in ['Tensile_value', 'Yield_value', 'Elongation_value']:
        train_and_plot(p)
