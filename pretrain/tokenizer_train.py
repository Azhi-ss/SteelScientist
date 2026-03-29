from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, set_seed

set_seed(123)


def get_training_corpus(raw_datasets):
    return (
        raw_datasets["train"][i : i + 10000]["text"]
        for i in range(0, len(raw_datasets["train"]), 10000)
    )


def full_sent_tokenize(tokenizer, file_name, max_seq_length=512):
    from tqdm import tqdm

    with open(file_name, 'r', encoding='utf-8') as f:
        sents = f.read().strip().split('\n')

    start_tok = tokenizer.convert_tokens_to_ids('[CLS]')
    sep_tok = tokenizer.convert_tokens_to_ids('[SEP]')

    tok_sents = [tokenizer(s, padding=False, truncation=False)['input_ids'] for s in tqdm(sents)]
    for s in tok_sents:
        s.pop(0)

    res = [[]]
    l_curr = 0

    for s in tok_sents:
        l_s = len(s)
        idx = 0
        while idx < l_s - 1:
            if l_curr == 0:
                res[-1].append(start_tok)
                l_curr = 1
            s_end = min(l_s, idx + max_seq_length - l_curr) - 1
            res[-1].extend(s[idx:s_end] + [sep_tok])
            idx = s_end
            if len(res[-1]) == max_seq_length:
                res.append([])
            l_curr = len(res[-1])

    attention_mask = []
    for s in res:
        attention_mask.append([1] * len(s) + [0] * (max_seq_length - len(s)))

    return {'input_ids': res, 'attention_mask': attention_mask}


if __name__ == "__main__":
    import torch
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    parser = ArgumentParser()
    parser.add_argument('--train_norm_file', required=True, type=str)
    parser.add_argument('--val_norm_file', required=True, type=str)
    parser.add_argument('--cache_dir', default=None, type=str)
    parser.add_argument('--save_dir', required=True, type=str)
    args = parser.parse_args()

    tokenizer_saved_dir = Path(args.save_dir) / 'tokenizer_saved'
    tokens_saved_dir = Path(args.save_dir) / 'tokens_saved'

    raw_datasets = load_dataset("text",
        data_files={"train": args.train_norm_file, "val": args.val_norm_file})
    print(f"raw_datasets >>>>>>>>> \n {raw_datasets}")

    training_corpus = get_training_corpus(raw_datasets)

    old_tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/deberta-v3-base",
        cache_dir=args.cache_dir)

    tokenizer = old_tokenizer.train_new_from_iterator(
        text_iterator=training_corpus,
        vocab_size=128_100)
    print(f"The length of old tokenizer: {len(old_tokenizer)}")
    print(f"The length of new tokenizer: {len(tokenizer)}")

    tokenizer.save_pretrained(tokenizer_saved_dir)
    print("The pretrained new tokenizer has been saved!")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_saved_dir)

    df_train = pd.DataFrame(full_sent_tokenize(tokenizer, args.train_norm_file))
    df_val = pd.DataFrame(full_sent_tokenize(tokenizer, args.val_norm_file))
    tokenized_datasets = DatasetDict({
        'train': Dataset.from_pandas(df_train),
        'val': Dataset.from_pandas(df_val)
    })

    print(f"tokenized_datasets >>>>>>>>> \n {tokenized_datasets}")

    tokenized_datasets.save_to_disk(tokens_saved_dir)
    print(tokens_saved_dir)
