import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import random
import torch
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from tokenizers.normalizers import BertNormalizer
from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
    Trainer,
    TrainingArguments,
)
import wandb

sys.path.append('..')

parser = ArgumentParser()
parser.add_argument('--model_name', choices=['steelberta', 'scibert', 'matscibert', 'bert'], required=True, type=str)
parser.add_argument('--model_save_dir', required=True, type=str)
parser.add_argument('--preds_save_dir', default=None, type=str)
parser.add_argument('--cache_dir', default=None, type=str)
parser.add_argument('--seeds', nargs='+', default=None, type=int)
parser.add_argument('--lm_lrs', nargs='+', default=None, type=float)
parser.add_argument('--non_lm_lr', default=3e-4, type=float)
args = parser.parse_args()

if args.model_name == 'steelberta':
    model_name = './../model_saved/checkpoint-140000'
    to_normalize = True
elif args.model_name == 'scibert':
    model_name = 'allenai/scibert_scivocab_uncased'
    to_normalize = True
elif args.model_name == 'matscibert':
    model_name = 'm3rg-iitd/matscibert'
    to_normalize = True
elif args.model_name == 'bert':
    model_name = 'bert-base-uncased'
    to_normalize = True
else:
    raise NotImplementedError

if args.seeds is None:
    args.seeds = [666, 888, 123, 555, 856]
if args.lm_lrs is None:
    args.lm_lrs = [5e-6]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


def ensure_dir(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    return dir_path


cache_dir = ensure_dir(args.cache_dir) if args.cache_dir else None
output_dir = ensure_dir(args.model_save_dir)
preds_save_dir = ensure_dir(args.preds_save_dir) if args.preds_save_dir else None

dataset_dir = Path('dataset')
data_files = {split: dataset_dir / f'{split}.csv' for split in ['train', 'val', 'test']}
datasets = load_dataset('csv', data_files=data_files, encoding="utf-8")

label_list = datasets['train'].unique('label')
num_labels = len(label_list)

max_seq_length = 512
model_revision = 'main'

tokenizer_kwargs = {
    'cache_dir': cache_dir,
    'use_fast': True,
    'revision': model_revision,
    'use_auth_token': None,
    'model_max_length': max_seq_length
}
tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)

norm = BertNormalizer(clean_text=True, handle_chinese_chars=True, strip_accents=True, lowercase=False)


def preprocess_function(examples):
    if to_normalize:
        examples['sentence'] = list(map(norm.normalize_str, examples['sentence']))
    result = tokenizer(examples['sentence'], padding=False, max_length=max_seq_length, truncation=True)
    result['label'] = [l for l in examples['label']]
    return result


tokenized_datasets = datasets.map(preprocess_function, batched=True)
train_dataset = tokenized_datasets['train']
val_dataset = tokenized_datasets['val']
test_dataset = tokenized_datasets['test']


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=1)

    results = {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds)
    }

    return results


sweep_configuration = {
    'method': 'grid',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'test_acc'},
    'parameters':
    {
        'lr': {'values': args.lm_lrs},
        'SEED': {'values': args.seeds},
        'train_epochs': {'values': [1]}
    }
}
sweep_id = wandb.sweep(sweep=sweep_configuration, project='pros_cls_steelberta')


def main():
    run = wandb.init(tags=[args.model_name])

    lr = wandb.config.lr
    SEED = wandb.config.SEED
    train_epochs = wandb.config.train_epochs

    set_seed(SEED)
    torch.backends.cudnn.benchmark = False

    train_args = TrainingArguments(
        num_train_epochs=train_epochs,
        output_dir=Path(output_dir) / f"{args.model_name}_{SEED}",
        evaluation_strategy='epoch',
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=lr,
        warmup_ratio=0.2,
        save_strategy='epoch',
        save_total_limit=2,
        seed=SEED,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        greater_is_better=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        from_tf=False,
        cache_dir=cache_dir,
        use_auth_token=None,
        num_labels=num_labels
    ).to(device)

    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not 'bert' in n],
            'lr': args.non_lm_lr
        },
        {
            'params': [p for n, p in model.named_parameters() if 'bert' in n],
            'lr': lr
        }
    ]
    optimizer_kwargs = {
        'betas': (0.9, 0.999),
        'eps': 1e-8,
    }
    optimizer = AdamW(optimizer_grouped_parameters, **optimizer_kwargs)

    customeTrainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None)
    )

    train_result = customeTrainer.train()
    print('*'*20, 'train', train_result, '*'*20)
    val_result = customeTrainer.evaluate()
    print('*'*20, 'val', val_result, '*'*20)
    test_result = customeTrainer.evaluate(test_dataset)
    print('*'*20, 'test', test_result, '*'*20)

    wandb.log({
        'SEED': SEED,
        'model_name': args.model_name,
        'val_acc': val_result['eval_accuracy'],
        'val_prec': val_result['eval_precision'],
        'val_recall': val_result['eval_recall'],
        'val_f1': val_result['eval_f1'],
        'test_acc': test_result['eval_accuracy'],
        'test_prec': test_result['eval_precision'],
        'test_recall': test_result['eval_recall'],
        'test_f1': test_result['eval_f1'],
    })

    run.finish()


wandb.agent(sweep_id, function=main, count=5)
