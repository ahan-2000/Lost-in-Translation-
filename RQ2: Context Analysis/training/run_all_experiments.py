#!/usr/bin/env python3
"""
Run Exp1, Exp2, Exp3 together under the same K-fold splits and compute Table 16 deltas.

  Exp1: FieldsOnlyModel        — P(Y | OccitanWord, LatinLemma, LatinGender)  [word-level]
  Exp2: ContextReaderModel      — P(Y | Sentence(full noun), i, fields)        [context + noun visible]
  Exp3: ContextReaderModel      — P(Y | Sentence(noun masked), i, fields)      [context + noun masked]

Delta statistics (Table 16):
  Δprob_1  = mean[ P_exp2(y_true) - P_exp1(y_true) ]   context (full noun) vs word-level
  Δprob_2  = mean[ P_exp3(y_true) - P_exp1(y_true) ]   context (masked noun) vs word-level
  Δlogp_1  = mean[ logP_exp2(y_true) - logP_exp1(y_true) ]
  Δlogp_2  = mean[ logP_exp3(y_true) - logP_exp1(y_true) ]
"""

import argparse, os, time, json, random, numpy as np, pandas as pd, torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer
from rich import print as rprint

from data import build_loader, normalize_label
from models import FieldsOnlyModel, ContextReaderModel
from train_eval import train_model, eval_model, predict_probs


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dk", type=int, default=128)
    ap.add_argument("--dv", type=int, default=128)
    ap.add_argument("--layer_kv", type=int, default=-1)
    ap.add_argument("--use_rel_bias", type=int, default=1)
    ap.add_argument("--no_self_peek", type=int, default=1)

    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--group_col", default=None)
    ap.add_argument("--index_in_whitespace", type=int, default=1)
    ap.add_argument("--masked_token", default="NOUNTOKEN")

    ap.add_argument("--only_neuter", type=int, default=1)
    ap.add_argument("--latin_neuter_values", nargs="+",
                    default=["N", "NEUT", "NEUTER", "NEUTRUM"])

    # Ablation: exclude Latin fields from the fields encoder
    ap.add_argument("--exclude_latin", type=int, default=0,
                    help="If 1, fields encoder receives only Occitan word (no Latin lemma/gender).")
    return ap.parse_args()


def make_folds(df, y, folds, seed, group_col=None):
    if group_col is not None and group_col in df.columns:
        gkf = GroupKFold(n_splits=folds)
        return list(gkf.split(df, y, groups=df[group_col]))
    else:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        return list(skf.split(df, y))


def mean_ci(x, rng_seed=13, B=2000):
    """Bootstrap mean + 95% CI."""
    rng = np.random.default_rng(rng_seed)
    xs = np.asarray(x)
    n = len(xs)
    boots = np.sort([xs[rng.choice(n, size=n, replace=True)].mean() for _ in range(B)])
    return float(xs.mean()), float(boots[int(0.025 * B)]), float(boots[int(0.975 * B)])


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs("outputs", exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = "all_exps_no_latin" if args.exclude_latin else "all_exps"
    outdir = os.path.join("outputs", f"{tag}_{stamp}")
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_parquet(args.data_path)
    rprint(f"[bold cyan]Loaded[/] {df.shape[0]} rows.")

    if args.only_neuter:
        allowed = {s.lower() for s in args.latin_neuter_values}
        mask = df[args.latin_gender_col].astype(str).str.strip().str.lower().isin(allowed)
        before = len(df)
        df = df.loc[mask].reset_index(drop=True)
        rprint(f"[bold magenta]Filter:[/] kept {len(df)} / {before} rows with Latin neuter")

    y = df[args.label_col].map(normalize_label).astype(int).values
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    rprint(f"[bold yellow]Class weights:[/] {dict(zip(['masculine', 'feminine'], class_weights))}")

    exclude_latin = bool(args.exclude_latin)
    rprint(f"[bold red]exclude_latin = {exclude_latin}[/]")

    splits = make_folds(df, y, args.folds, args.seed, group_col=args.group_col)

    sent_tok = AutoTokenizer.from_pretrained(args.pretrained)
    fields_tok = AutoTokenizer.from_pretrained(args.pretrained)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    oof = []
    fold_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
        rprint(f"\n[bold yellow]{'='*60}[/]")
        rprint(f"[bold yellow]Fold {fold}/{args.folds}[/] train={len(tr_idx)} val={len(va_idx)}")
        rprint(f"[bold yellow]{'='*60}[/]")
        df_tr = df.iloc[tr_idx].reset_index(drop=True)
        df_va = df.iloc[va_idx].reset_index(drop=True)

        common_loader_kw = dict(
            sentence_col=args.sentence_col, noun_index_col=args.noun_index_col,
            label_col=args.label_col, occitan_word_col=args.occitan_word_col,
            latin_lemma_col=args.latin_lemma_col, latin_gender_col=args.latin_gender_col,
            max_len_sent=args.max_len_sent, max_len_fields=args.max_len_fields,
            batch_size=args.batch_size,
            index_in_whitespace=bool(args.index_in_whitespace),
            masked_token=args.masked_token,
            exclude_latin=exclude_latin,
        )

        # ------- Data loaders -------
        # Exp1 & Exp2: mask_noun=False  (noun visible)
        tr_full = build_loader(df_tr, sent_tok, fields_tok, shuffle=True,
                               mask_noun=False, **common_loader_kw)
        va_full = build_loader(df_va, sent_tok, fields_tok, shuffle=False,
                               mask_noun=False, **common_loader_kw)

        # Exp3: mask_noun=True  (noun replaced by NOUNTOKEN)
        tr_mask = build_loader(df_tr, sent_tok, fields_tok, shuffle=True,
                               mask_noun=True, **common_loader_kw)
        va_mask = build_loader(df_va, sent_tok, fields_tok, shuffle=False,
                               mask_noun=True, **common_loader_kw)

        # ------- Exp1: FieldsOnly (word-level baseline) -------
        rprint(f"[bold green]Training Exp1 (FieldsOnly) — Fold {fold}[/]")
        exp1 = FieldsOnlyModel(pretrained=args.pretrained, hidden=args.hidden,
                               dropout=args.dropout, freeze_encoder=True)
        exp1 = train_model(exp1, tr_full, va_full,
                           lr=args.lr, weight_decay=args.weight_decay,
                           epochs=args.epochs, warmup_ratio=args.warmup_ratio,
                           device=device, experiment_name="Exp1_FieldsOnly",
                           fold=fold, class_weights=class_weights,
                           save_dir=outdir, patience=3)

        # ------- Exp2: ContextReader (full noun visible) -------
        rprint(f"[bold green]Training Exp2 (ContextReader, full noun) — Fold {fold}[/]")
        exp2 = ContextReaderModel(
            pretrained=args.pretrained, hidden=args.hidden, dropout=args.dropout,
            heads=args.heads, dk=args.dk, dv=args.dv,
            use_rel_bias=bool(args.use_rel_bias), rel_window=64,
            layer_kv=args.layer_kv, freeze_encoder=False,
            no_self_peek=bool(args.no_self_peek))
        exp2 = train_model(exp2, tr_full, va_full,
                           lr=args.lr, weight_decay=args.weight_decay,
                           epochs=args.epochs, warmup_ratio=args.warmup_ratio,
                           device=device, experiment_name="Exp2_ContextFull",
                           fold=fold, class_weights=class_weights,
                           save_dir=outdir, patience=3)

        # ------- Exp3: ContextReader (noun masked) -------
        rprint(f"[bold green]Training Exp3 (ContextReader, noun masked) — Fold {fold}[/]")
        exp3 = ContextReaderModel(
            pretrained=args.pretrained, hidden=args.hidden, dropout=args.dropout,
            heads=args.heads, dk=args.dk, dv=args.dv,
            use_rel_bias=bool(args.use_rel_bias), rel_window=64,
            layer_kv=args.layer_kv, freeze_encoder=False,
            no_self_peek=bool(args.no_self_peek))
        exp3 = train_model(exp3, tr_mask, va_mask,
                           lr=args.lr, weight_decay=args.weight_decay,
                           epochs=args.epochs, warmup_ratio=args.warmup_ratio,
                           device=device, experiment_name="Exp3_ContextMasked",
                           fold=fold, class_weights=class_weights,
                           save_dir=outdir, patience=3)

        # ------- OOF predictions -------
        rprint(f"[bold cyan]Computing OOF predictions — Fold {fold}[/]")
        row1, p1, lp1, yv = predict_probs(exp1, va_full, device=device)
        _,    p2, lp2, _  = predict_probs(exp2, va_full, device=device)
        _,    p3, lp3, _  = predict_probs(exp3, va_mask, device=device)

        fold_df = pd.DataFrame({
            "row_id": row1, "fold": fold, "y_true": yv,
            "p_exp1": p1, "lp_exp1": lp1,
            "p_exp2": p2, "lp_exp2": lp2,
            "p_exp3": p3, "lp_exp3": lp3,
        })
        # Δ_1  = context (full noun) – word-level
        fold_df["delta1_prob"] = fold_df["p_exp2"] - fold_df["p_exp1"]
        fold_df["delta1_logp"] = fold_df["lp_exp2"] - fold_df["lp_exp1"]
        # Δ_2  = context (masked noun) – word-level
        fold_df["delta2_prob"] = fold_df["p_exp3"] - fold_df["p_exp1"]
        fold_df["delta2_logp"] = fold_df["lp_exp3"] - fold_df["lp_exp1"]
        oof.append(fold_df)

        # ------- Per-fold eval metrics -------
        vl1, va1_acc, vf1 = eval_model(exp1, va_full, device)
        vl2, va2_acc, vf2 = eval_model(exp2, va_full, device)
        vl3, va3_acc, vf3 = eval_model(exp3, va_mask, device)
        fold_metrics.append({
            "fold": fold,
            "exp1_loss": vl1, "exp1_acc": va1_acc, "exp1_f1": vf1,
            "exp2_loss": vl2, "exp2_acc": va2_acc, "exp2_f1": vf2,
            "exp3_loss": vl3, "exp3_acc": va3_acc, "exp3_f1": vf3,
        })
        rprint(f"[bold cyan]Fold {fold} Exp1:[/] Acc={va1_acc:.4f} F1={vf1:.4f}")
        rprint(f"[bold cyan]Fold {fold} Exp2:[/] Acc={va2_acc:.4f} F1={vf2:.4f}")
        rprint(f"[bold cyan]Fold {fold} Exp3:[/] Acc={va3_acc:.4f} F1={vf3:.4f}")

    # ===== Aggregate =====
    oof_df = pd.concat(oof, ignore_index=True)
    oof_df.to_csv(os.path.join(outdir, "oof_predictions.csv"), index=False)

    metrics_df = pd.DataFrame(fold_metrics)
    metrics_df.to_csv(os.path.join(outdir, "fold_metrics.csv"), index=False)

    # ===== Table 16 delta statistics =====
    summary = {"exclude_latin": exclude_latin}
    for name in ["delta1_prob", "delta1_logp", "delta2_prob", "delta2_logp"]:
        m, lo, hi = mean_ci(oof_df[name].values)
        summary[name] = {"mean": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)]}

    # Per-experiment aggregate metrics
    for prefix, label in [("exp1", "Exp1_FieldsOnly"),
                          ("exp2", "Exp2_ContextFull"),
                          ("exp3", "Exp3_ContextMasked")]:
        acc_mean = metrics_df[f"{prefix}_acc"].mean()
        acc_std  = metrics_df[f"{prefix}_acc"].std()
        f1_mean  = metrics_df[f"{prefix}_f1"].mean()
        f1_std   = metrics_df[f"{prefix}_f1"].std()
        loss_mean = metrics_df[f"{prefix}_loss"].mean()
        summary[prefix] = {
            "label": label,
            "mean_acc": round(acc_mean, 4), "std_acc": round(acc_std, 4),
            "mean_f1": round(f1_mean, 4),   "std_f1": round(f1_std, 4),
            "mean_loss": round(loss_mean, 4),
        }

    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ===== Print summary =====
    rprint(f"\n[bold green]{'='*60}[/]")
    rprint(f"[bold green]ALL EXPERIMENTS COMPLETE[/]")
    rprint(f"[bold green]{'='*60}[/]")
    rprint(f"[bold]exclude_latin = {exclude_latin}[/]")
    rprint(f"\n[bold cyan]Per-experiment results:[/]")
    for prefix in ["exp1", "exp2", "exp3"]:
        s = summary[prefix]
        rprint(f"  {s['label']:25s}  Acc={s['mean_acc']:.4f}±{s['std_acc']:.4f}  "
               f"F1={s['mean_f1']:.4f}±{s['std_f1']:.4f}")

    rprint(f"\n[bold cyan]Table 16 — Delta statistics:[/]")
    for name, label in [("delta1_prob", "Δprob_1 (full noun − word)"),
                        ("delta2_prob", "Δprob_2 (masked noun − word)"),
                        ("delta1_logp", "Δlogp_1 (full noun − word)"),
                        ("delta2_logp", "Δlogp_2 (masked noun − word)")]:
        d = summary[name]
        rprint(f"  {label:35s}  {d['mean']:.4f}  95%CI [{d['ci95'][0]:.4f}, {d['ci95'][1]:.4f}]")

    rprint(f"\n[bold cyan]Results saved to:[/] {outdir}")


if __name__ == "__main__":
    main()
