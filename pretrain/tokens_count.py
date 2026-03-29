from argparse import ArgumentParser
from pathlib import Path

import torch
import pandas as pd
import datasets
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from tqdm import tqdm

SEED = 666

def count_len(tokenizer, element):
    outputs = tokenizer(
        element["text"],
        truncation=False,
        padding=False,
        return_length=True,
    )
    tokens_len = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        tokens_len.append(length)
    return {"tokens_len": tokens_len}


if __name__ == "__main__":
    import torch
    from transformers import set_seed

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    parser = ArgumentParser()
    parser.add_argument('--train_norm_file', required=True, type=str)
    parser.add_argument('--val_norm_file', required=True, type=str)
    parser.add_argument('--save_dir', required=True, type=str)
    args = parser.parse_args()

    tokens_len_dir = Path(args.save_dir) / 'tokens_len'
    tokenizer_saved_dir = Path(args.save_dir) / 'tokenizer_saved'

    raw_datasets = load_dataset("text",
        data_files={"train": args.train_norm_file, "val": args.val_norm_file})
    print(f"raw_datasets >>>>>>>>> \n {raw_datasets}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_saved_dir)

    tokens_lens = raw_datasets.map(
        count_len, batched=True, remove_columns=raw_datasets["train"].column_names,
        fn_kwargs={"tokenizer": tokenizer}
    )

    tokens_lens.save_to_disk(tokens_len_dir)
    print(tokens_lens)
