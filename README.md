# SteelScientist

Steel design based on a large language model.

## Paper

**Paper link:** [Steel design based on a large language model](https://doi.org/10.1016/j.actamat.2024.120663)

```bibtex
@article{tian2025steel,
    title={Steel design based on a large language model},
    author={Tian Shaohan and Jiang Xue and Wang Weiren and Jing Zhihua and Zhang Chi and Zhang Cheng and Lookman Turab and Su Yanjing},
    journal={Acta Materialia},
    volume={285},
    pages={120663},
    year={2025},
    publisher={Elsevier}
}
```

> **Note:** The fully parsed markdown version of the original paper (extracted via MinerU), along with all high-resolution figures and detailed training hyperparameters (e.g., 0.96 Billion words corpus, AdamW $\beta_1=0.9$, learning rate setups, 188M parameters DeBERTa base), can be found locally at `docs/paper/steel-design-mineru.md` and in the [AGENT.md](./.agent/AGENT.md) documentation.

## Project Structure

```
.
├── requirements.txt       # Simplified dependencies (34 packages)
├── README.md
├── classification/       # Classification tasks
│   ├── cls.py
│   └── run.sh
├── regression/           # Regression tasks
│   └── reg_v1.py
├── pretrain/             # Pre-training pipeline
│   ├── corpus_normalize.py
│   ├── tokenizer_train.py
│   ├── model_train.py
│   └── run.sh
├── evaluation/           # Evaluation notebooks
└── datasets/             # Dataset files
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Classification

```bash
cd classification
bash run.sh
```

### Regression

```bash
cd regression
python reg_v1.py
```

### Pre-training

```bash
cd pretrain
bash run.sh
```

Or run steps individually:

```bash
cd pretrain
python -u corpus_normalize.py --train_corpus_file <path> --val_corpus_file <path> ...
python -u tokenizer_train.py ...
python -u model_train.py ...
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CACHE_DIR` | `./cache` | HuggingFace cache directory |
| `SAVE_DIR` | `./output` | Model/tokenizer save directory |
| `MODEL_NAME` | `steelberta` | Model for classification |

## Models

- **steelberta** - Custom steel domain BERT (default)
- **scibert** - Science BERT
- **bert** - Base BERT
- **matscibert** - Materials Science BERT
