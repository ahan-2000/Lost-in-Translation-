# Paper Workspace

Research code and data workspace for experiments on Latin-to-Occitan gender transfer, with a focus on former neuter nouns.

The repository currently combines three lines of work:

- lemma-level feature engineering and modeling (`lemma analysis`)
- context-aware neural experiments for RQ2 (`RQ2: Context Analysis`)
- preliminary embedding/tokenizer experiments (`priliminary_analysis`)

## What this repo contains

- `data/`: primary text and spreadsheet sources plus a corpus utility script
- `lemma analysis/`: phase-based pipeline refactored from notebooks (cleaning, stats, feature engineering, ablation/error analysis)
- `RQ2: Context Analysis/`: word-vs-context model training scripts, K-fold evaluation, and explainability utilities
- `priliminary_analysis/`: early experiments for encoder comparisons and MLM/tokenizer work

## Environment setup

Python 3.10+ is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r "lemma analysis/requirements.txt"
pip install pyarrow rich wandb sentencepiece openpyxl
```

Notes:
- `wandb` is used by training scripts in `RQ2: Context Analysis/training` and `priliminary_analysis/tokenizer`.
- `pyarrow` is required for `pandas.read_parquet(...)` in RQ2 scripts.
- `openpyxl` is useful for reading Excel source files in `data/`.



