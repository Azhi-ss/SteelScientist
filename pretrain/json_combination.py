import json
import random
from tqdm import tqdm
from argparse import ArgumentParser
from collections import Counter

random.seed(123)


def load_corpus(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def split_corpus(corpus, eval_ratio=0.1):
    train_dict = {}
    eval_dict = {}
    for k in ['steel', 'article', 'meeting', 'patent']:
        train_dict[k], eval_dict[k] = [], []
        temp_li = list(range(len(corpus[k])))
        eval_idx = random.sample(temp_li, round(len(temp_li) * eval_ratio) + 1)
        print(f"{k}_sum: {len(temp_li)}, eval: {len(eval_idx)}, train: {len(temp_li) - len(eval_idx)}")
        for i, text in enumerate(tqdm(corpus[k])):
            if i in eval_idx:
                eval_dict[k].append(text)
            else:
                train_dict[k].append(text)
    return train_dict, eval_dict


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--abstract_file', required=True, type=str)
    parser.add_argument('--fulltext_file', required=True, type=str)
    parser.add_argument('--train_output', required=True, type=str)
    parser.add_argument('--val_output', required=True, type=str)
    parser.add_argument('--corpus_output', default=None, type=str)
    args = parser.parse_args()

    abs_text = load_corpus(args.abstract_file)
    full_text = load_corpus(args.fulltext_file)
    abs_text['steel'] = full_text['steel']

    for k, v in abs_text.items():
        print(f"{k}: {len(abs_text[k])}")

    publisher = []
    for i in tqdm(range(len(full_text['steel']))):
        if len(full_text['steel'][i]['content']) > 0:
            publisher.append(full_text['steel'][i]['publisher'])
    print(Counter(publisher))

    train_dict, eval_dict = split_corpus(abs_text)

    with open(args.train_output, 'w', encoding='utf-8') as f:
        json.dump(train_dict, f, indent=4, ensure_ascii=False)

    with open(args.val_output, 'w', encoding='utf-8') as f:
        json.dump(eval_dict, f, indent=4, ensure_ascii=False)

    if args.corpus_output:
        new_dic = dict(abs_text)
        new_dic['steel'] = full_text['steel']
        with open(args.corpus_output, 'w', encoding='utf-8') as f:
            json.dump(new_dic, f, indent=4, ensure_ascii=False)
