#!/usr/bin/env python3
"""
K-Fold Cross-Validation Training Script for Exp1 and Exp2
Trains FieldsOnlyModel (Exp1) and ContextReaderModel (Exp2) with cross-validation.
"""

import argparse, os, time, json, numpy as np, pandas as pd, torch
import sys
from pathlib import Path

# Add parent directory to path to access utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer
from rich import print as rprint

from utils.seed import set_seed
from word_context_analysis.training.data import build_loader, normalize_label
from word_context_analysis.training.models import FieldsOnlyModel, ContextReaderModel
from word_context_analysis.training.train_eval import train_model, eval_model, predict_probs

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--sentence_col", required=True)
    ap.add_argument("--noun_index_col", required=True)
    ap.add_argument("--label_col", required=True)
    ap.add_argument("--occitan_word_col", required=True)
    ap.add_argument("--latin_lemma_col", required=True)
    ap.add_argument("--latin_gender_col", required=True)

    ap.add_argument("--pretrained", default="bert-base-multilingual-cased")
    ap.add_argument("--max_len_sent", type=int, default=128)
    ap.add_argument("--max_len_fields", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--hidden", type=int, default=512)  # Increased hidden size
    ap.add_argument("--dropout", type=float, default=0.1)  # Reduced dropout for better learning
    ap.add_argument("--heads", type=int, default=8)  # Increased from 4 to 8 for better attention
    ap.add_argument("--dk", type=int, default=64)
    ap.add_argument("--dv", type=int, default=64)
    ap.add_argument("--layer_kv", type=int, default=-1)
    ap.add_argument("--use_rel_bias", type=int, default=1)
    ap.add_argument("--no_self_peek", type=int, default=0)
    ap.add_argument("--freeze_encoders", type=int, default=1)

    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--group_col", default=None, help="Set e.g. occitan_word or latin_lemma to avoid leakage")
    ap.add_argument("--index_in_whitespace", type=int, default=1)
    ap.add_argument("--masked_token", default="NOUNTOKEN")

    # >>> NEW: restrict evaluation set to Latin neuter nouns
    ap.add_argument("--only_neuter", type=int, default=1,
                    help="If 1, filter the dataframe to rows where latin_gender is neuter before K-fold.")
    ap.add_argument("--latin_neuter_values", nargs="+",
                    default=["N","NEUT","NEUTER","NEUTRUM"],
                    help="Case-insensitive values in latin_gender_col treated as neuter.")
    return ap.parse_args()

def make_folds(df, y, folds, seed, group_col=None):
    if group_col is not None and group_col in df.columns:
        gkf = GroupKFold(n_splits=folds)
        return list(gkf.split(df, y, groups=df[group_col]))
    else:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        return list(skf.split(df, y))

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs("outputs", exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join("outputs", stamp); os.makedirs(outdir, exist_ok=True)

    df = pd.read_parquet(args.data_path)
    rprint(f"[bold cyan]Loaded[/] {df.shape[0]} rows.")

    # >>> NEW: filter to Latin neuter only (before any splits)
    if args.only_neuter:
        allowed = {s.lower() for s in args.latin_neuter_values}
        # treat missing values as non-neuter
        mask = df[args.latin_gender_col].astype(str).str.strip().str.lower().isin(allowed)
        before = len(df)
        df = df.loc[mask].reset_index(drop=True)
        rprint(f"[bold magenta]Filter:[/] kept {len(df)} / {before} rows with Latin neuter "
               f"({', '.join(args.latin_neuter_values)}).")
        if len(df) == 0:
            raise ValueError("Filtering to Latin neuter produced an empty dataset. "
                             "Adjust --latin_neuter_values or check the column.")

    # labels for stratification come from the Occitan gender column
    y = df[args.label_col].map(normalize_label).astype(int).values
    
    # Calculate class weights to handle imbalance
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    rprint(f"[bold yellow]Class weights:[/] {dict(zip(['masculine', 'feminine'], class_weights))}")
    
    splits = make_folds(df, y, args.folds, args.seed, group_col=args.group_col)

    sent_tok = AutoTokenizer.from_pretrained(args.pretrained)
    fields_tok = AutoTokenizer.from_pretrained(args.pretrained)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    oof = []
    for fold,(tr_idx, va_idx) in enumerate(splits, start=1):
        rprint(f"[bold yellow]Fold {fold}/{args.folds}[/] train={len(tr_idx)} val={len(va_idx)}")
        df_tr = df.iloc[tr_idx].reset_index(drop=True)
        df_va = df.iloc[va_idx].reset_index(drop=True)

        # DataLoaders
        tr1 = build_loader(df_tr, sent_tok, fields_tok,
                           args.sentence_col, args.noun_index_col, args.label_col,
                           args.occitan_word_col, args.latin_lemma_col, args.latin_gender_col,
                           max_len_sent=args.max_len_sent, max_len_fields=args.max_len_fields,
                           batch_size=args.batch_size, shuffle=True,
                           index_in_whitespace=bool(args.index_in_whitespace),
                           mask_noun=False, masked_token=args.masked_token)
        va1 = build_loader(df_va, sent_tok, fields_tok, args.sentence_col, args.noun_index_col, args.label_col,
                           args.occitan_word_col, args.latin_lemma_col, args.latin_gender_col,
                           max_len_sent=args.max_len_sent, max_len_fields=args.max_len_fields,
                           batch_size=args.batch_size, shuffle=False,
                           index_in_whitespace=bool(args.index_in_whitespace),
                           mask_noun=False, masked_token=args.masked_token)

        tr2, va2 = tr1, va1  # same text; model differs

        # Models
        exp1 = FieldsOnlyModel(pretrained=args.pretrained, hidden=args.hidden, dropout=args.dropout,
                               freeze_encoder=bool(args.freeze_encoders))
        exp2 = ContextReaderModel(pretrained=args.pretrained, hidden=args.hidden, dropout=args.dropout,
                                  heads=args.heads, dk=128, dv=128,  # Increased from 64 to 128
                                  use_rel_bias=bool(args.use_rel_bias),
                                  rel_window=64, layer_kv=args.layer_kv,  # Increased window from 32 to 64
                                  freeze_encoder=False,  # Unfreeze for Exp2 to learn sentence representations
                                  no_self_peek=True)  # Enable no_self_peek for better attention

        # Train
        print("Train Exp1 (FieldsOnly)")
        exp1 = train_model(exp1, tr1, va1, lr=args.lr, weight_decay=args.weight_decay,
                           epochs=args.epochs, warmup_ratio=args.warmup_ratio, device=device,
                           experiment_name="Exp1_FieldsOnly", fold=fold, class_weights=class_weights, save_dir=outdir)

        print("Train Exp2 (Reader+Context)")
        exp2 = train_model(exp2, tr2, va2, lr=args.lr*0.5, weight_decay=args.weight_decay,  # Lower LR for larger model
                           epochs=args.epochs, warmup_ratio=args.warmup_ratio, device=device,
                           experiment_name="Exp2_ReaderContext", fold=fold, class_weights=class_weights, save_dir=outdir)

        # Predict OOF for this fold
        row1, p1, lp1, yv = predict_probs(exp1, va1, device=device)
        _,   p2, lp2, _  = predict_probs(exp2, va2, device=device)

        fold_df = pd.DataFrame({
            "row_id": row1, "fold": fold, "y_true": yv,
            "p_exp1": p1, "lp_exp1": lp1,
            "p_exp2": p2, "lp_exp2": lp2,
        })
        # deltas vs Exp1
        fold_df["delta1_prob"] = fold_df["p_exp2"] - fold_df["p_exp1"]
        fold_df["delta1_logp"] = fold_df["lp_exp2"] - fold_df["lp_exp1"]
        oof.append(fold_df)

    oof_df = pd.concat(oof, ignore_index=True)
    oof_path = os.path.join(outdir, "oof_predictions.csv")
    oof_df.to_csv(oof_path, index=False)
    rprint(f"[bold cyan]Saved OOF[/] to {oof_path}")

    # summary (bootstrap mean + 95% CI)
    def mean_ci(x):
        rng = np.random.default_rng(13)
        B=2000; xs=np.asarray(x); n=len(xs)
        boots = [xs[rng.choice(n, size=n, replace=True)].mean() for _ in range(B)]
        boots = np.sort(np.array(boots))
        return float(xs.mean()), float(boots[int(0.025*B)]), float(boots[int(0.975*B)])

    summary = {}
    for name in ["delta1_prob","delta1_logp"]:
        m,lo,hi = mean_ci(oof_df[name].values)
        summary[name] = {"mean": m, "ci95": [lo,hi]}

    # OOF Acc/F1 per exp (approx via thresholding p(class=1) reconstructed from true-class prob)
    from sklearn.metrics import accuracy_score, f1_score
    def metric_for(prefix):
        if prefix=="exp1": p = oof_df["p_exp1"].values
        elif prefix=="exp2": p = oof_df["p_exp2"].values
        y = oof_df["y_true"].values
        p1 = np.where(y==1, p, 1.0-p)  # convert true-class prob to P(class=1)
        y_pred = (p1>=0.5).astype(int)
        return float(accuracy_score(y, y_pred)), float(f1_score(y, y_pred, average="macro"))
    for prefix in ["exp1","exp2"]:
        acc, f1 = metric_for(prefix)
        summary[prefix] = {"acc": acc, "f1_macro": f1}

    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    rprint("[bold green]Done.[/] Summary written to summary.json.")

if __name__ == "__main__":
    main()

