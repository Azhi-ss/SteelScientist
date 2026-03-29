#!/usr/bin/env python
# coding: utf-8

import os
import warnings
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Optional

import copy
import csv
import pickle
import random
import time
import uuid

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, IterableDataset, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import matplotlib.pyplot as plt
from datasets import Dataset
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import ray
from ray import air, tune
from ray.air import session
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings('ignore')

ALL_ACT_LAYERS = {
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh
}

ALL_LOSS_FUNC = {
    "mseloss": nn.MSELoss,
    "l1loss": nn.L1Loss,
    "smoothl1loss": nn.SmoothL1Loss
}

ALL_OPTIM_FUNC = {
    "adamw": torch.optim.AdamW,
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD
}


def set_global_seed(seed=123):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.random.set_seed(seed)
        tf.experimental.numpy.random.seed(seed)
        tf.set_random_seed(seed)
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        os.environ['TF_DETERMINISTIC_OPS'] = '1'

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def range_data(temp_df, ts_range=[10, 3000], ys_range=[0, 3000], el_range=[5, 150]):
    temp_df = temp_df[(temp_df['Tensile_value']>=ts_range[0]) & (temp_df['Tensile_value']<=ts_range[1])]
    temp_df = temp_df[(temp_df['Yield_value']>=ys_range[0]) & (temp_df['Yield_value']<=ys_range[1])]
    temp_df = temp_df[(temp_df['Elongation_value']>=el_range[0]) & (temp_df['Elongation_value']<=el_range[1])]
    temp_df.reset_index(drop=True, inplace=True)
    return temp_df


def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]


class EmbeddingExtractor:
    def __init__(self, model_name: str, device: torch.device):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device

    def get_embeddings(self, text_list):
        encoded_input = self.tokenizer(
            text_list, padding='max_length', max_length=512, truncation=True, return_tensors="pt"
        )
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        model_output = self.model(**encoded_input)
        return cls_pooling(model_output).detach().cpu().numpy()[0].tolist()

    def get_element_embeddings(self, text_list):
        encoded_input = self.tokenizer(
            text_list, padding='max_length', max_length=512, truncation=True, return_tensors="pt"
        )
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        model_output = self.model(**encoded_input)
        return cls_pooling(model_output).detach().cpu().numpy()


def gen_text_embed(bert_df, col_embed='Text', extractor: Optional[EmbeddingExtractor] = None):
    tqdm.pandas(desc='Progress bar!')
    bert_df['process_emb'] = bert_df[col_embed].progress_apply(extractor.get_embeddings)

    temp_bert_df = pd.DataFrame(pd.Series(bert_df['process_emb'][0])).T
    for i in range(1, bert_df.shape[0]):
        new_row = pd.DataFrame(pd.Series(bert_df['process_emb'][i])).T
        temp_bert_df = pd.concat([temp_bert_df, new_row], ignore_index=True, axis=0)
    temp_bert_df.reset_index(drop=True, inplace=True)
    temp_bert_df.columns = [col_embed+str(i) for i in range(768)]
    df = pd.concat([bert_df, temp_bert_df], axis=1)
    df.drop(columns=['process_emb'], inplace=True)

    return df


def add_ele_embed(df, extractor: EmbeddingExtractor):
    coms = list(df.columns[3:-770])
    x = np.matrix(extractor.get_element_embeddings(coms))
    w1 = list(df.iloc[0, 3:-770])
    y = np.average(x, axis=0, weights=w1)

    for i in np.arange(1, df.shape[0]):
        w = list(df.iloc[i, 3:-770])
        temp_y = np.average(x, axis=0, weights=w)
        y = np.vstack((y, temp_y))

    ele_df = pd.DataFrame(y, columns=['ele'+str(i) for i in range(768)])

    return ele_df


def load_data(
    label='train_data',
    pred_prop='Yield_value',
    fes=['com', 'text_embed'],
    split_ratio=0.75,
    seed=42,
    perplexity=3,
    data_dir: Path = Path('./datasets'),
    extractor: Optional[EmbeddingExtractor] = None
):
    data_origin = pd.read_excel(data_dir / f'{label}.xlsx')
    com_cols = set(data_origin.columns[17:-4])

    if label == 'train_data':
        data_origin = range_data(data_origin, ts_range=[0, 3000], ys_range=[0, 3000], el_range=[5, 95])
        filter = data_origin['status'] == 1
        data = data_origin[filter].copy()
        data.reset_index(drop=True, inplace=True)
    else:
        data = data_origin.copy()

    for e_col in com_cols:
        data[e_col].fillna(0.0, inplace=True)
    data.fillna('', inplace=True)

    drop_cols = ['DOIs', 'Files', 'problem', 'status', 'Table_topic', 'title', 'abstract', 'Other_ele',
                 'Text_addition', 'Tensile_name', 'Tensile_unit', 'Yield_name', 'Yield_unit',
                 'Elongation_name', 'Elongation_unit', 'Material']
    data.drop(columns=drop_cols, inplace=True)

    data = gen_text_embed(data, extractor=extractor)

    com_drop = ['Fe']
    data.drop(columns=com_drop, inplace=True)

    data = pd.concat([data, add_ele_embed(data, extractor)], axis=1)

    prop_all = ['Tensile_value', 'Yield_value', 'Elongation_value']
    if pred_prop == 'all':
        drop_col = []
    else:
        drop_col = list(set(prop_all) - set([pred_prop]))

    data = gen_text_embed(data, col_embed='actions', extractor=extractor)

    if 'com' not in fes:
        drop_col += [col for col in list(data.columns) if col in com_cols]
    if 'text_embed' not in fes:
        drop_col += ['Text'+str(i) for i in range(768)]
    if 'com_embed' not in fes:
        drop_col += ['ele'+str(i) for i in range(768)]
    if 'action_embed' not in fes:
        drop_col += ['actions'+str(i) for i in range(768)]
    if 'text' not in fes:
        drop_col.append('Text')
    data.drop(columns=drop_col+['actions'], inplace=True)

    if label == 'train_data':
        for _ in range(50):
            data = data.sample(frac=1.0, random_state=seed)
        train_data, test_data = np.split(
            data.sample(frac=1, random_state=seed, ignore_index=True),
            [int(split_ratio*len(data))]
        )
        train_data.reset_index(drop=True, inplace=True)
        test_data.reset_index(drop=True, inplace=True)

        if perplexity and 'com_embed' in fes:
            tsne = TSNE(n_components=3, learning_rate='auto', init='random', perplexity=perplexity)
            train_tsne = tsne.fit_transform(train_data.iloc[:, -768:])
            test_tsne = tsne.transform(test_data.iloc[:, -768:])
            train_data = pd.concat([train_data, pd.DataFrame(train_tsne, columns=['tsne'+str(i) for i in range(3)])], axis=1)
            test_data = pd.concat([test_data, pd.DataFrame(test_tsne, columns=['tsne'+str(i) for i in range(3)])], axis=1)

        return train_data, test_data
    else:
        if perplexity and 'com_embed' in fes:
            data_com_embed = pd.DataFrame(
                TSNE(n_components=3, learning_rate='auto', init='random', perplexity=perplexity).fit_transform(
                    data.iloc[:, -768:]
                ),
                columns=['tsne'+str(i) for i in range(3)]
            )
            data = pd.concat([data, data_com_embed], axis=1)
        return data


def gen_data_class(df):
    eles_cols = list(df.columns[1:-2307])
    text_embeds = ['Text'+str(i) for i in range(768)]
    com_embeds = ['ele'+str(i) for i in range(768)]
    com_tsne = ['tsne'+str(i) for i in range(3)]
    action_embeds = ['actions'+str(i) for i in range(768)]

    dic = {
        "targets": df.iloc[:, 0],
        "eles": df.loc[:, eles_cols],
        "text_embeds": df.loc[:, text_embeds],
        "com_embeds": df.loc[:, com_embeds],
        "action_embeds": df.loc[:, action_embeds],
        "com_tsne_embeds": df.loc[:, com_tsne]
    }

    return dic


def eval_model(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy() if torch.is_tensor(y_true) else y_true
    y_pred = y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else y_pred

    r2 = round(r2_score(y_true, y_pred), 3)
    rmse = round(mean_squared_error(y_true, y_pred, squared=False), 3)
    mae = round(mean_absolute_error(y_true, y_pred), 3)
    return {"r2": r2, "rmse": rmse, "mae": mae}


def plot_test_data(
    best_new_text_y,
    best_new_text_preds,
    text_result,
    best_exp_y,
    best_exp_preds,
    exp_result,
    labels,
    fig_name: Path = Path("./outputs/test.png"),
    point_size=10
):
    text_val = [text_result['r2'], text_result['rmse'], text_result['mae']]
    exp_val = [exp_result['r2'], exp_result['rmse'], exp_result['mae']]

    fig = plt.figure(figsize=(10, 5), dpi=120)

    if isinstance(best_new_text_y, torch.Tensor):
        y_new_text = best_new_text_y.detach().cpu().numpy()
        y_new_text_pred = best_new_text_preds.detach().cpu().numpy()
        y_exp = best_exp_y.detach().cpu().numpy()
        y_exp_pred = best_exp_preds.detach().cpu().numpy()
    else:
        y_new_text, y_new_text_pred = best_new_text_y, best_new_text_preds
        y_exp, y_exp_pred = best_exp_y, best_exp_preds

    plt.subplot(121)
    plt.scatter(y_new_text, y_new_text_pred, s=point_size, color='#1F4B73')
    plt.plot(
        np.arange(int(max(y_new_text)*0.1), int(max(y_new_text)*1.1)),
        np.arange(int(max(y_new_text)*0.1), int(max(y_new_text)*1.1)),
        '-', color='#A2555A'
    )
    plt.title(
        f"$R^2$={text_val[0]}, RMSE={text_val[1]}, MAE={text_val[2]}",
        fontdict={'family': 'Times New Roman', 'size': 10}
    )
    plt.ylabel('prediction', fontdict={'family': 'Times New Roman', 'size': 14})
    plt.xlabel('True', fontdict={'family': 'Times New Roman', 'size': 14})
    plt.xlim(0, int(max(y_new_text)*1.2))
    plt.ylim(0, int(max(y_new_text)*1.2))
    plt.legend(labels=[labels[0], 'Y=X'])
    plt.grid()

    plt.subplot(122)
    plt.scatter(y_exp, y_exp_pred, s=point_size, color='#1F4B73')
    plt.plot(
        np.arange(int(max(y_exp)*0.1), int(max(y_exp)*1.1)),
        np.arange(int(max(y_exp)*0.1), int(max(y_exp)*1.1)),
        '-', color='#A2555A'
    )
    plt.title(
        f"$R^2$={exp_val[0]}, RMSE={exp_val[1]}, MAE={exp_val[2]}",
        fontdict={'family': 'Times New Roman', 'size': 10}
    )
    plt.ylabel('prediction', fontdict={'family': 'Times New Roman', 'size': 14})
    plt.xlabel('True', fontdict={'family': 'Times New Roman', 'size': 14})
    plt.xlim(0, int(max(y_exp)*1.2))
    plt.ylim(0, int(max(y_exp)*1.2))
    plt.legend(labels=[labels[1], 'Y=X'])
    plt.grid()

    plt.savefig(fig_name)


class CustomSimpleDataset(Dataset):
    def __init__(self, eles, text_embeds, com_embeds, com_tsne_embeds, action_embeds, targets, device: torch.device):
        self.eles = eles.values if isinstance(eles, pd.DataFrame) else eles
        self.text_embeds = text_embeds.values if isinstance(text_embeds, pd.DataFrame) else text_embeds
        self.com_embeds = com_embeds.values if isinstance(com_embeds, pd.DataFrame) else com_embeds
        self.com_tsne_embeds = com_tsne_embeds.values if isinstance(com_tsne_embeds, pd.DataFrame) else com_tsne_embeds
        self.action_embeds = action_embeds.values if isinstance(action_embeds, pd.DataFrame) else action_embeds
        self.targets = targets.values if isinstance(targets, pd.DataFrame) else targets
        self.device = device

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        ele = self.eles[idx]
        text_embed = self.text_embeds[idx]
        com_embed = self.com_embeds[idx]
        com_tsne_embed = self.com_tsne_embeds[idx]
        action_embed = self.action_embeds[idx]
        target = self.targets[idx]

        return {
            "labels": torch.tensor(target, dtype=torch.float32).to(self.device),
            "eles": torch.tensor(ele, dtype=torch.float32).squeeze(0).to(self.device),
            "text_embeds": torch.tensor(text_embed, dtype=torch.float32).squeeze(0).to(self.device),
            "com_embeds": torch.tensor(com_embed, dtype=torch.float32).squeeze(0).to(self.device),
            "action_embeds": torch.tensor(action_embed, dtype=torch.float32).squeeze(0).to(self.device),
            "com_tsne_embeds": torch.tensor(com_tsne_embed, dtype=torch.float32).squeeze(0).to(self.device),
        }


class Unit(nn.Module):
    def __init__(
        self,
        normalization: str,
        in_features: int,
        out_features: int,
        activation: str,
        dropout_prob: float,
    ):
        super().__init__()
        if normalization == "layer_norm":
            self.norm = nn.LayerNorm(in_features)
        elif normalization == "batch_norm":
            self.norm = nn.BatchNorm1d(in_features)
        elif normalization == "null_norm":
            self.norm = None
        else:
            raise ValueError(f"unknown normalization: {normalization}")
        self.fc = nn.Linear(in_features, out_features)
        if activation == "leaky_relu":
            self.act_fn = nn.LeakyReLU(negative_slope=-1)
        else:
            self.act_fn = ALL_ACT_LAYERS[activation]()

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        if self.norm is not None:
            x = self.norm(x)
        x = self.fc(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        return x


class FcUnit(nn.Module):
    def __init__(
        self,
        num_features: int,
        normalization: Optional[str] = "null_norm",
        activation: Optional[str] = 'relu',
        dropout_prob: Optional[float] = 0.1,
    ):
        super().__init__()
        if normalization == "layer_norm":
            self.norm = nn.LayerNorm(num_features)
        elif normalization == "batch_norm":
            self.norm = nn.BatchNorm1d(num_features)
        elif normalization == "null_norm":
            self.norm = None
        else:
            raise ValueError(f"unknown normalization: {normalization}")

        self.fc = nn.LazyLinear(num_features)
        if activation == "leaky_relu":
            self.act_fn = nn.LeakyReLU(negative_slope=-1)
        else:
            self.act_fn = ALL_ACT_LAYERS[activation]()

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        if self.norm is not None:
            x = self.norm(x)
        x = self.fc(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        return x


class Cnn1dUnit(nn.Module):
    def __init__(
        self,
        out_channels: int,
        kernel_size: Optional[int] = 3,
        activation: Optional[str] = 'relu',
        stride: Optional[int] = 1,
        padding: Optional[int] = 1,
        pooling: Optional[str] = 'max',
    ):
        super().__init__()
        if pooling == "max":
            self.pool = nn.MaxPool1d(kernel_size=kernel_size)
        elif pooling == "avg":
            self.pool = nn.AvgPool1d(kernel_size=kernel_size)
        else:
            raise ValueError(f"unknown pooling type: {pooling}")

        if activation == "leaky_relu":
            self.act_fn = nn.LeakyReLU(negative_slope=-1)
        else:
            self.act_fn = ALL_ACT_LAYERS[activation]()

        self.conv1d = nn.LazyConv1d(
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

    def forward(self, x):
        x = self.conv1d(x)
        x = self.act_fn(x)
        x = self.pool(x)
        return x


class Cnn2dUnit(nn.Module):
    def __init__(
        self,
        out_channels: int,
        kernel_size: Optional[int] = 2,
        activation: Optional[str] = 'relu',
        stride: Optional[int] = 1,
        padding: Optional[int] = 1,
        pooling: Optional[str] = 'max',
    ):
        super().__init__()
        if pooling == "max":
            self.pool = nn.MaxPool2d(kernel_size=kernel_size)
        elif pooling == "avg":
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)
        else:
            raise ValueError(f"unknown pooling type: {pooling}")

        if activation == "leaky_relu":
            self.act_fn = nn.LeakyReLU(negative_slope=-1)
        else:
            self.act_fn = ALL_ACT_LAYERS[activation]()

        self.conv2d = nn.LazyConv2d(
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

    def forward(self, x):
        x = self.conv2d(x)
        x = self.act_fn(x)
        x = self.pool(x)
        return x


class CustomSimpleModel(nn.Module):
    def __init__(
            self,
            simple_layer_list,
            concat_layer_list,
            seq_embed_con1d_list,
            seq_embed_fc_list,
            seq_embed_con2d_list,
            seq_embed_2d_fc_list,
            simple_layer_drop_prob=0.0,
            concat_layer_drop_prob=0.0,
    ):
        super(CustomSimpleModel, self).__init__()

        simple_layer = []
        for num_fes in simple_layer_list:
            per_unit = FcUnit(
                num_features=num_fes,
                normalization='null_norm',
                activation='relu',
                dropout_prob=simple_layer_drop_prob
            )
            simple_layer.append(per_unit)
        self.simple_layer = nn.Sequential(*simple_layer)

        seq_embed_layer = []
        for num_channels in seq_embed_con1d_list:
            per_unit = Cnn1dUnit(
                out_channels=num_channels,
                kernel_size=2,
                stride=1,
                padding=1
            )
            seq_embed_layer.append(per_unit)
        seq_embed_layer.append(nn.Flatten())
        for num_fes in seq_embed_fc_list:
            per_unit = FcUnit(
                num_features=num_fes,
                normalization='null_norm',
                activation='relu',
                dropout_prob=simple_layer_drop_prob
            )
            seq_embed_layer.append(per_unit)
        self.seq_embed_layer = nn.Sequential(*seq_embed_layer)

        common_embed_layer = []
        for num_channels in seq_embed_con2d_list:
            per_unit = Cnn2dUnit(
                out_channels=num_channels,
                kernel_size=2,
                stride=1,
                padding=2
            )
            common_embed_layer.append(per_unit)
        common_embed_layer.append(nn.Flatten())
        for num_fes in seq_embed_2d_fc_list:
            per_unit = FcUnit(
                num_features=num_fes,
                normalization='null_norm',
                activation='relu',
                dropout_prob=simple_layer_drop_prob
            )
            common_embed_layer.append(per_unit)
        self.common_embed_layer = nn.Sequential(*common_embed_layer)

        concat_layer = []
        for num_fes in concat_layer_list:
            per_unit = FcUnit(
                num_features=num_fes,
                normalization='null_norm',
                activation='relu',
                dropout_prob=concat_layer_drop_prob
            )
            concat_layer.append(per_unit)
        self.concat_layer = nn.Sequential(*concat_layer)

    def forward(self, eles, text_embeds, com_embeds, com_tsne_embeds, action_embeds, labels):
        inputs_addition = torch.concat(
            [com_embeds.unsqueeze(1), text_embeds.unsqueeze(1), action_embeds.unsqueeze(1)],
            dim=1
        )
        com_dense_output = self.simple_layer(com_embeds)
        text_dense_output = self.simple_layer(text_embeds)
        act_dense_output = self.simple_layer(action_embeds)

        com_embed_output = self.seq_embed_layer(com_dense_output.unsqueeze(1))
        text_embed_output = self.seq_embed_layer(text_dense_output.unsqueeze(1))
        action_embed_output = self.seq_embed_layer(act_dense_output.unsqueeze(1))

        output_addition = self.common_embed_layer(inputs_addition.unsqueeze(1))

        output = torch.concat(
            [output_addition, com_embed_output, text_embed_output, action_embed_output],
            dim=1
        )
        output = self.concat_layer(output)

        return output


def train_function(config):
    prop = config["prop"]
    seed = config["seed"]
    split_ratio = config["split_ratio"]
    train_batch = config["train_batch"]
    epoch = config["epoch"]
    step = config["step"]
    gamma_ratio = config["gamma_ratio"]
    lr = config["lr"]

    simple_layer_num_512 = config["simple_layer_num_512"]
    simple_layer_num_256 = config["simple_layer_num_256"]
    simple_layer_drop_prob = config["simple_layer_drop_prob"]

    concat_layer_num_64 = config["concat_layer_num_64"]
    concat_layer_num_32 = config["concat_layer_num_32"]
    concat_layer_drop_prob = config["concat_layer_drop_prob"]

    conv_channels_layer_1 = config["conv_channels_layer_1"]
    conv_channels_layer_2 = config["conv_channels_layer_2"]

    embed_layer_num_32 = config["embed_layer_num_32"]
    embed_layer_num_16 = config["embed_layer_num_16"]

    simple_layer_list = [512 for i in range(simple_layer_num_512)] + [256 for j in range(simple_layer_num_256)]
    concat_layer_list = [64 for i in range(concat_layer_num_64)] + [32 for j in range(concat_layer_num_32)] + [8, 4, 1]
    seq_embed_con1d_list = [conv_channels_layer_1, conv_channels_layer_2]
    seq_embed_con2d_list = [conv_channels_layer_1, conv_channels_layer_2]
    seq_embed_fc_list = [32 for i in range(embed_layer_num_32)] + [16 for j in range(embed_layer_num_16)]
    seq_embed_2d_fc_list = [32 for i in range(embed_layer_num_32)] + [16 for j in range(embed_layer_num_16)]

    paras_li = ['prop', 'seed', 'split_ratio', 'train_batch',
                'epoch', 'lr', 'step', 'gamma_ratio', 'simple_layer_num_512',
                'simple_layer_num_256', 'simple_layer_drop_prob', 'concat_layer_num_64',
                'concat_layer_num_32', 'concat_layer_drop_prob', 'conv_channels_layer_1',
                'conv_channels_layer_2', 'embed_layer_num_32', 'embed_layer_num_16']
    fes = ['com', 'com_embed', 'text_embed', 'action_embed']
    set_global_seed(seed=seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = './../model_saved/checkpoint-140000'
    extractor = EmbeddingExtractor(model_name, device)

    train_data, test_data = load_data(
        label='train_data',
        pred_prop=prop,
        fes=fes,
        split_ratio=split_ratio,
        seed=seed,
        perplexity=3,
        extractor=extractor
    )
    new_text_data = load_data(label='text_test', pred_prop=prop, fes=fes, perplexity=3, extractor=extractor)
    exp_data = load_data(label='exp_test', pred_prop=prop, fes=fes, perplexity=3, extractor=extractor)
    print(f"train_data.shape:{train_data.shape}, test_data.shape:{test_data.shape}")

    new_text_dataloader = DataLoader(
        CustomSimpleDataset(**gen_data_class(new_text_data), device=device),
        batch_size=len(new_text_data)
    )
    exp_dataloader = DataLoader(
        CustomSimpleDataset(**gen_data_class(exp_data), device=device),
        batch_size=len(exp_data)
    )
    all_train_dataloader = DataLoader(
        CustomSimpleDataset(**gen_data_class(train_data), device=device),
        batch_size=len(train_data),
        shuffle=False,
        drop_last=False
    )
    train_dataloader = DataLoader(
        CustomSimpleDataset(**gen_data_class(train_data), device=device),
        batch_size=train_batch,
        shuffle=True,
        drop_last=False
    )
    val_dataloader = DataLoader(
        CustomSimpleDataset(**gen_data_class(test_data), device=device),
        batch_size=len(test_data)
    )

    reg_model = CustomSimpleModel(
        simple_layer_list=simple_layer_list,
        concat_layer_list=concat_layer_list,
        seq_embed_con1d_list=seq_embed_con1d_list,
        seq_embed_fc_list=seq_embed_fc_list,
        seq_embed_con2d_list=seq_embed_con2d_list,
        seq_embed_2d_fc_list=seq_embed_2d_fc_list,
        simple_layer_drop_prob=simple_layer_drop_prob,
        concat_layer_drop_prob=concat_layer_drop_prob,
    ).to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(reg_model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma_ratio)

    dt_string = datetime.now().strftime("%d_%m_%H.%M.%S")
    writer = SummaryWriter('./outputs/runs', flush_secs=20)

    best_val_r2 = -1e5
    best_new_text_r2 = -1e5
    best_exp_r2 = -1e5
    patience = 20
    patience_counter = 0
    model_save_path = Path(f"./outputs/reg_model_saved/{paras_string}.pt")
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch_num in tqdm(range(epoch)):
        reg_model.train()
        for batch, inputs in enumerate(train_dataloader):
            y = inputs['labels'].unsqueeze(1)
            preds = reg_model(**inputs)
            train_loss = loss_fn(preds, y)
            train_r2 = eval_model(y, preds)['r2']

            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()

        reg_model.eval()
        with torch.no_grad():
            for batch, inputs in enumerate(val_dataloader):
                y = inputs['labels'].unsqueeze(1)
                preds = reg_model(**inputs)
                val_loss = loss_fn(preds, y).item()
                val_r2 = eval_model(y, preds)['r2']

            for batch, inputs in enumerate(new_text_dataloader):
                y = inputs['labels'].unsqueeze(1)
                preds = reg_model(**inputs)
                new_text_r2 = eval_model(y, preds)['r2']

                if new_text_r2 > best_new_text_r2:
                    best_new_text_r2 = new_text_r2
                    plot_best_text_y = y.detach().cpu().numpy()
                    plot_best_text_preds = preds.detach().cpu().numpy()
                    plot_best_text_result = eval_model(plot_best_text_y, plot_best_text_preds)

            for batch, inputs in enumerate(exp_dataloader):
                y = inputs['labels'].unsqueeze(1)
                preds = reg_model(**inputs)
                exp_r2 = eval_model(y, preds)['r2']
                if exp_r2 > best_exp_r2:
                    best_exp_r2 = exp_r2
                    plot_best_exp_y = y.detach().cpu().numpy()
                    plot_best_exp_preds = preds.detach().cpu().numpy()
                    plot_best_exp_result = eval_model(plot_best_exp_y, plot_best_exp_preds)

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            patience_counter = 0
            torch.save(reg_model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch_num+1}")
                break

        writer.add_scalars(
            'Reg_'+dt_string+'/Loss',
            tag_scalar_dict={'train_loss': train_loss, 'val_loss': val_loss},
            global_step=epoch_num+1
        )
        writer.add_scalars(
            'Reg_'+dt_string+'/R2',
            tag_scalar_dict={'train_r2': train_r2, 'val_r2': val_r2},
            global_step=epoch_num+1
        )
        writer.add_scalars(
            'Reg_'+dt_string+'/Test_R2',
            tag_scalar_dict={'new_text_r2': new_text_r2, 'exp_r2': exp_r2},
            global_step=epoch_num+1
        )

    writer.close()

    paras_string = '_'.join([str(eval(s)) + '_' for s in paras_li])
    model_save_path = Path(f"./outputs/reg_model_saved/{paras_string}.pt")
    if model_save_path.exists():
        best_model = CustomSimpleModel(
            simple_layer_list=simple_layer_list,
            concat_layer_list=concat_layer_list,
            seq_embed_con1d_list=seq_embed_con1d_list,
            seq_embed_fc_list=seq_embed_fc_list,
            seq_embed_con2d_list=seq_embed_con2d_list,
            seq_embed_2d_fc_list=seq_embed_2d_fc_list,
            simple_layer_drop_prob=simple_layer_drop_prob,
            concat_layer_drop_prob=concat_layer_drop_prob,
        ).to(device)
        best_model.load_state_dict(torch.load(model_save_path))
    else:
        best_model = reg_model

    best_model.eval()
    with torch.no_grad():
        for batch, inputs in enumerate(all_train_dataloader):
            y_train = inputs['labels'].unsqueeze(1).detach().cpu().numpy()
            y_train_preds = best_model(**inputs).detach().cpu().numpy()
            train_result = eval_model(y_train, y_train_preds)

        for batch, inputs in enumerate(val_dataloader):
            y_val = inputs['labels'].unsqueeze(1).detach().cpu().numpy()
            y_val_preds = best_model(**inputs).detach().cpu().numpy()
            val_result = eval_model(y_val, y_val_preds)

        for batch, inputs in enumerate(new_text_dataloader):
            best_new_text_y = inputs['labels'].unsqueeze(1)
            best_new_text_preds = best_model(**inputs)
            best_new_text_result = eval_model(best_new_text_y, best_new_text_preds)

        for batch, inputs in enumerate(exp_dataloader):
            best_exp_y = inputs['labels'].unsqueeze(1)
            best_exp_preds = best_model(**inputs)
            best_exp_result = eval_model(best_exp_y, best_exp_preds)

    plot_test_data(
        y_train, y_train_preds, train_result,
        y_val, y_val_preds, val_result,
        fig_name=Path(f"./outputs/figs/train_{paras_string}.png"),
        labels=["Model train data", "Model test data"],
        point_size=5
    )

    plot_test_data(
        plot_best_text_y, plot_best_text_preds, plot_best_text_result,
        plot_best_exp_y, plot_best_exp_preds, plot_best_exp_result,
        fig_name=Path(f"./outputs/figs/test_{paras_string}.png"),
        labels=["New literature data", "Experiment data"],
        point_size=15
    )

    print(y_train.shape)
    y_train = y_train.T.tolist()[0]
    y_train_preds = y_train_preds.T.tolist()[0]

    y_val = y_val.T.tolist()[0]
    y_val_preds = y_val_preds.T.tolist()[0]

    plot_best_text_y = plot_best_text_y.T.tolist()[0]
    plot_best_text_preds = plot_best_text_preds.T.tolist()[0]

    plot_best_exp_y = plot_best_exp_y.T.tolist()[0]
    plot_best_exp_preds = plot_best_exp_preds.T.tolist()[0]

    max_len = len(y_train)

    y_val += [6666.0 for i in range(max_len-len(y_val))]
    y_val_preds += [6666.0 for i in range(max_len-len(y_val_preds))]
    plot_best_text_y += [6666.0 for i in range(max_len-len(plot_best_text_y))]
    plot_best_text_preds += [6666.0 for i in range(max_len-len(plot_best_text_preds))]
    plot_best_exp_y += [6666.0 for i in range(max_len-len(plot_best_exp_y))]
    plot_best_exp_preds += [6666.0 for i in range(max_len-len(plot_best_exp_preds))]

    plot_result = pd.DataFrame({
        'y_train': y_train,
        'y_train_preds': y_train_preds,
        'y_val': y_val,
        'y_val_preds': y_val_preds,
        'text_y': plot_best_text_y,
        'text_preds': plot_best_text_preds,
        'exp_y': plot_best_exp_y,
        'exp_preds': plot_best_exp_preds
    })
    plot_result.to_excel(f"./outputs/preds/preds_{paras_string}.xlsx", index=None)

    with open('./outputs/reg_model.csv', 'a+') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([
            prop, train_result['r2'], val_result['r2'], best_new_text_result['r2'],
            best_exp_result['r2'], seed, split_ratio, train_batch, epoch, lr, step,
            gamma_ratio, simple_layer_num_512, simple_layer_num_256, simple_layer_drop_prob,
            concat_layer_num_64, concat_layer_num_32, concat_layer_drop_prob,
            conv_channels_layer_1, conv_channels_layer_2, embed_layer_num_32,
            embed_layer_num_16
        ])

    session.report({
        "train_r2": train_result['r2'],
        "train_rmse": train_result['rmse'],
        "train_mae": train_result['mae'],
        "val_r2": val_result['r2'],
        "val_rmse": val_result['rmse'],
        "val_mae": val_result['mae'],
        "new_text_r2": best_new_text_result['r2'],
        "new_text_rmse": best_new_text_result['rmse'],
        "new_text_mae": best_new_text_result['mae'],
        "exp_r2": best_exp_result['r2'],
        "exp_rmse": best_exp_result['rmse'],
        "exp_mae": best_exp_result['mae']
    })


def tune_with_callback():
    tuner = tune.Tuner(
        train_function,
        tune_config=tune.TuneConfig(
            metric="val_r2",
            mode="max",
            num_samples=1,
            max_concurrent_trials=1,
        ),
        run_config=air.RunConfig(
            name="tune_reg",
            local_dir="./outputs/logs",
            callbacks=[
                WandbLoggerCallback(
                    project="reg_model",
                    mode="offline",
                    tags=["1010"])
            ]
        ),
        param_space={
            'prop': tune.choice(['Tensile_value', 'Yield_value', 'Elongation_value']),
            'seed': tune.choice([42, 789, 666, 888]),
            'split_ratio': tune.choice([0.75, 0.7]),
            'train_batch': tune.choice([32]),
            'epoch': tune.choice([300]),
            'lr': tune.choice([0.01]),
            'step': tune.choice([80]),
            'gamma_ratio': tune.choice([0.5, 0.1, 0.8]),
            'simple_layer_num_512': tune.choice([2, 3, 4, 5]),
            'simple_layer_num_256': tune.choice([2, 3]),
            'simple_layer_drop_prob': tune.choice([0.05, 0.0, 0.1]),
            'concat_layer_num_64': tune.choice([2, 3, 4, 5]),
            'concat_layer_num_32': tune.choice([2, 3]),
            'concat_layer_drop_prob': tune.choice([0.05, 0.0, 0.1]),
            'conv_channels_layer_1': tune.choice([1, 2]),
            'conv_channels_layer_2': tune.choice([1, 2]),
            'embed_layer_num_32': tune.choice([2, 3, 4, 5]),
            'embed_layer_num_16': tune.choice([2, 3, 4]),
        }
    )
    tuner.fit()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"You are using '{device}' device!")

    model_name = './../model_saved/checkpoint-140000'
    extractor = EmbeddingExtractor(model_name, device)

    for prop in ['Elongation_value']:
        with open('./outputs/reg_model.csv', 'a+') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([
                'prop', 'train_r2', 'best_val_r2', 'best_new_text_r2', 'best_exp_r2',
                'seed', 'split_ratio', 'perplexity', 'train_batch', 'epoch', 'lr', 'step', 'gamma_ratio'
            ])

        for simple_layer_list in [[512, 512, 512, 256]]:
            for concat_layer_list in [[64, 64, 32, 8, 4, 1]]:
                for seq_embed_con1d_list in [[1, 1]]:
                    for split_ratio in [0.75]:
                        for step in [80]:
                            for layer_dorp in [0]:
                                for seed in [42]:
                                    train_batch = 32
                                    perplexity = 3
                                    epoch = 250
                                    lr = 0.01
                                    gamma_ratio = 0.5

                                    set_global_seed(seed=seed)
                                    fes = ['com', 'com_embed', 'text_embed', 'action_embed']
                                    paras_string = f"{prop}_{split_ratio}_{train_batch}_{epoch}_{lr}_{step}_{gamma_ratio}__{seed}"

                                    train_data, test_data = load_data(
                                        label='train_data',
                                        pred_prop=prop,
                                        fes=fes,
                                        split_ratio=split_ratio,
                                        seed=seed,
                                        perplexity=perplexity,
                                        extractor=extractor
                                    )
                                    new_text_data = load_data(
                                        label='text_test',
                                        pred_prop=prop,
                                        fes=fes,
                                        perplexity=perplexity,
                                        extractor=extractor
                                    )
                                    exp_data = load_data(
                                        label='exp_test',
                                        pred_prop=prop,
                                        fes=fes,
                                        perplexity=perplexity,
                                        extractor=extractor
                                    )
                                    print(f"train_data.shape:{train_data.shape}, test_data.shape:{test_data.shape}")

                                    new_text_dataloader = DataLoader(
                                        CustomSimpleDataset(**gen_data_class(new_text_data), device=device),
                                        batch_size=len(new_text_data)
                                    )
                                    exp_dataloader = DataLoader(
                                        CustomSimpleDataset(**gen_data_class(exp_data), device=device),
                                        batch_size=len(exp_data)
                                    )
                                    all_train_dataloader = DataLoader(
                                        CustomSimpleDataset(**gen_data_class(train_data), device=device),
                                        batch_size=len(train_data),
                                        shuffle=False,
                                        drop_last=False
                                    )
                                    train_dataloader = DataLoader(
                                        CustomSimpleDataset(**gen_data_class(train_data), device=device),
                                        batch_size=train_batch,
                                        shuffle=True,
                                        drop_last=False
                                    )
                                    val_dataloader = DataLoader(
                                        CustomSimpleDataset(**gen_data_class(test_data), device=device),
                                        batch_size=len(test_data)
                                    )

                                    reg_model = CustomSimpleModel(
                                        simple_layer_list=simple_layer_list,
                                        concat_layer_list=concat_layer_list,
                                        seq_embed_con1d_list=seq_embed_con1d_list,
                                        seq_embed_fc_list=[32, 16],
                                        seq_embed_con2d_list=[1, 1],
                                        seq_embed_2d_fc_list=[32, 16],
                                        simple_layer_drop_prob=layer_dorp,
                                        concat_layer_drop_prob=layer_dorp,
                                    ).to(device)

                                    loss_fn = nn.MSELoss()
                                    optimizer = torch.optim.AdamW(reg_model.parameters(), lr=lr, weight_decay=0.01)
                                    scheduler = torch.optim.lr_scheduler.StepLR(
                                        optimizer,
                                        step_size=step,
                                        gamma=gamma_ratio
                                    )

                                    dt_string = datetime.now().strftime("%d_%m_%H.%M.%S")
                                    writer = SummaryWriter('./outputs/runs', flush_secs=20)

                                    best_val_r2 = -1e5
                                    best_new_text_r2 = -1e5
                                    best_exp_r2 = -1e5

                                    for epoch_num in tqdm(range(epoch)):
                                        reg_model.train()
                                        for batch, inputs in enumerate(train_dataloader):
                                            y = inputs['labels'].unsqueeze(1)
                                            preds = reg_model(**inputs)
                                            train_loss = loss_fn(preds, y)
                                            train_r2 = eval_model(y, preds)['r2']

                                            train_loss.backward()
                                            optimizer.step()
                                            optimizer.zero_grad()
                                        scheduler.step()

                                        reg_model.eval()
                                        with torch.no_grad():
                                            for batch, inputs in enumerate(val_dataloader):
                                                y = inputs['labels'].unsqueeze(1)
                                                preds = reg_model(**inputs)
                                                val_loss = loss_fn(preds, y).item()
                                                val_r2 = eval_model(y, preds)['r2']

                                            for batch, inputs in enumerate(new_text_dataloader):
                                                y = inputs['labels'].unsqueeze(1)
                                                preds = reg_model(**inputs)
                                                new_text_r2 = eval_model(y, preds)['r2']

                                                if new_text_r2 > best_new_text_r2:
                                                    best_new_text_r2 = new_text_r2
                                                    plot_best_text_y = y.detach().cpu().numpy()
                                                    plot_best_text_preds = preds.detach().cpu().numpy()
                                                    plot_best_text_result = eval_model(
                                                        plot_best_text_y, plot_best_text_preds
                                                    )

                                            for batch, inputs in enumerate(exp_dataloader):
                                                y = inputs['labels'].unsqueeze(1)
                                                preds = reg_model(**inputs)
                                                exp_r2 = eval_model(y, preds)['r2']
                                                if exp_r2 > best_exp_r2:
                                                    best_exp_r2 = exp_r2
                                                    plot_best_exp_y = y.detach().cpu().numpy()
                                                    plot_best_exp_preds = preds.detach().cpu().numpy()
                                                    plot_best_exp_result = eval_model(
                                                        plot_best_exp_y, plot_best_exp_preds
                                                    )

                                        writer.add_scalars(
                                            'Reg_'+dt_string+'/Loss',
                                            tag_scalar_dict={'train_loss': train_loss, 'val_loss': val_loss},
                                            global_step=epoch_num+1
                                        )
                                        writer.add_scalars(
                                            'Reg_'+dt_string+'/R2',
                                            tag_scalar_dict={'train_r2': train_r2, 'val_r2': val_r2},
                                            global_step=epoch_num+1
                                        )
                                        writer.add_scalars(
                                            'Reg_'+dt_string+'/Test_R2',
                                            tag_scalar_dict={'new_text_r2': new_text_r2, 'exp_r2': exp_r2},
                                            global_step=epoch_num+1
                                        )

                                    writer.close()

                                    best_model = reg_model

                                    best_model.eval()
                                    with torch.no_grad():
                                        for batch, inputs in enumerate(all_train_dataloader):
                                            y_train = inputs['labels'].unsqueeze(1).detach().cpu().numpy()
                                            y_train_preds = best_model(**inputs).detach().cpu().numpy()
                                            train_result = eval_model(y_train, y_train_preds)

                                        for batch, inputs in enumerate(val_dataloader):
                                            y_val = inputs['labels'].unsqueeze(1).detach().cpu().numpy()
                                            y_val_preds = best_model(**inputs).detach().cpu().numpy()
                                            val_result = eval_model(y_val, y_val_preds)

                                        for batch, inputs in enumerate(new_text_dataloader):
                                            best_new_text_y = inputs['labels'].unsqueeze(1)
                                            best_new_text_preds = best_model(**inputs)
                                            best_new_text_result = eval_model(
                                                best_new_text_y, best_new_text_preds
                                            )

                                        for batch, inputs in enumerate(exp_dataloader):
                                            best_exp_y = inputs['labels'].unsqueeze(1)
                                            best_exp_preds = best_model(**inputs)
                                            best_exp_result = eval_model(best_exp_y, best_exp_preds)

                                    plot_test_data(
                                        y_train, y_train_preds, train_result,
                                        y_val, y_val_preds, val_result,
                                        fig_name=Path(f"./outputs/figs/train_{paras_string}.png"),
                                        labels=["Model train data", "Model test data"],
                                        point_size=5
                                    )

                                    plot_test_data(
                                        plot_best_text_y, plot_best_text_preds, plot_best_text_result,
                                        plot_best_exp_y, plot_best_exp_preds, plot_best_exp_result,
                                        fig_name=Path(f"./outputs/figs/test_{paras_string}.png"),
                                        labels=["New literature data", "Experiment data"],
                                        point_size=15
                                    )

                                    print(y_train.shape)
                                    y_train = y_train.T.tolist()[0]
                                    y_train_preds = y_train_preds.T.tolist()[0]

                                    y_val = y_val.T.tolist()[0]
                                    y_val_preds = y_val_preds.T.tolist()[0]

                                    plot_best_text_y = plot_best_text_y.T.tolist()[0]
                                    plot_best_text_preds = plot_best_text_preds.T.tolist()[0]

                                    plot_best_exp_y = plot_best_exp_y.T.tolist()[0]
                                    plot_best_exp_preds = plot_best_exp_preds.T.tolist()[0]

                                    max_len = len(y_train)

                                    y_val += [6666.0 for i in range(max_len-len(y_val))]
                                    y_val_preds += [6666.0 for i in range(max_len-len(y_val_preds))]
                                    plot_best_text_y += [6666.0 for i in range(max_len-len(plot_best_text_y))]
                                    plot_best_text_preds += [6666.0 for i in range(max_len-len(plot_best_text_preds))]
                                    plot_best_exp_y += [6666.0 for i in range(max_len-len(plot_best_exp_y))]
                                    plot_best_exp_preds += [6666.0 for i in range(max_len-len(plot_best_exp_preds))]

                                    plot_result = pd.DataFrame({
                                        'y_train': y_train,
                                        'y_train_preds': y_train_preds,
                                        'y_val': y_val,
                                        'y_val_preds': y_val_preds,
                                        'text_y': plot_best_text_y,
                                        'text_preds': plot_best_text_preds,
                                        'exp_y': plot_best_exp_y,
                                        'exp_preds': plot_best_exp_preds
                                    })
                                    plot_result.to_excel(f"./outputs/preds/preds_{paras_string}.xlsx", index=None)

                                    with open('./outputs/reg_model.csv', 'a+') as csvfile:
                                        csvwriter = csv.writer(csvfile)
                                        csvwriter.writerow([
                                            prop, train_result['r2'], val_result['r2'],
                                            best_new_text_result['r2'], best_exp_result['r2'],
                                            seed, split_ratio, perplexity, train_batch, epoch,
                                            lr, step, gamma_ratio
                                        ])
