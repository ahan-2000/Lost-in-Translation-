#!/usr/bin/env python3
"""
Exp2 No-Latin Ablation: Context Reader without Latin lemma / Latin gender fields.
Fields encoder receives only the Occitan word: "[OCC]word"
Everything else is identical to run_exp2_only.py.
"""

import argparse, os, time, json, random, numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

# Add the training directory itself to path so we can import sibling modules
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer
from rich import print as rprint

from data import build_loader, normalize_label
from models import ContextReaderModel
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
                    default=["N","NEUT","NEUTER","NEUTRUM"])
    
    return ap.parse_args()

def make_folds(df, y, folds, seed, group_col=None):
    if group_col is not None and group_col in df.columns:
        gkf = GroupKFold(n_splits=folds)
        return list(gkf.split(df, y, groups=df[group_col]))
    else:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        return list(skf.split(df, y))

def visualize_attention(model, loader, tokenizer, device, save_dir, fold, num_samples=10):
    """Visualize attention patterns for sample sentences"""
    model.eval()
    attention_data = []
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_samples:
                break
                
            # Move batch to device
            for k in batch: 
                batch[k] = batch[k].to(device)
            
            # Get predictions with attention
            output = model(**batch, return_attention=True)
            
            # Decode tokens
            tokens = tokenizer.convert_ids_to_tokens(batch["input_ids_sent"][0])
            noun_pos = batch["noun_pos"][0].item()
            true_label = batch["label"][0].item()
            pred_label = output["logits"][0].argmax().item()
            
            # Get attention weights (B, heads, T)
            attention_weights = output["attention_weights"][0]  # (heads, T)
            
            attention_data.append({
                "tokens": tokens,
                "noun_pos": noun_pos,
                "true_label": true_label,
                "pred_label": pred_label,
                "attention_weights": attention_weights.numpy(),
                "attention_scores": output["attention_scores"][0].numpy()
            })
    
    # Create visualizations
    os.makedirs(os.path.join(save_dir, "attention_viz"), exist_ok=True)
    
    for i, data in enumerate(attention_data):
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'Fold {fold} - Sample {i+1} (No Latin): '
                     f'{"correct" if data["true_label"] == data["pred_label"] else "wrong"}', 
                     fontsize=16)
        
        # Plot each attention head
        for head in range(min(8, data["attention_weights"].shape[0])):
            row, col = head // 4, head % 4
            
            # Create attention heatmap
            attn_matrix = data["attention_weights"][head].reshape(1, -1)
            
            sns.heatmap(attn_matrix, 
                       xticklabels=data["tokens"][:len(attn_matrix[0])],
                       yticklabels=[f'Head {head+1}'],
                       cmap='Blues',
                       ax=axes[row, col],
                       cbar=True)
            
            # Highlight noun position
            if data["noun_pos"] < len(data["tokens"]):
                axes[row, col].axvline(x=data["noun_pos"], color='red', linestyle='--', alpha=0.7)
            
            axes[row, col].set_title(f'Head {head+1}')
            axes[row, col].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "attention_viz", f"fold_{fold}_sample_{i+1}.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    # Save attention data
    with open(os.path.join(save_dir, f"attention_data_fold_{fold}.json"), "w") as f:
        json.dump({
            "samples": [
                {
                    "tokens": data["tokens"],
                    "noun_pos": data["noun_pos"],
                    "true_label": data["true_label"],
                    "pred_label": data["pred_label"],
                    "attention_weights": data["attention_weights"].tolist(),
                    "attention_scores": data["attention_scores"].tolist()
                }
                for data in attention_data
            ]
        }, f, indent=2)

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs("outputs", exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join("outputs", f"exp2_no_latin_{stamp}")
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_parquet(args.data_path)
    rprint(f"[bold cyan]Loaded[/] {df.shape[0]} rows.")

    # Filter to Latin neuter only
    if args.only_neuter:
        allowed = {s.lower() for s in args.latin_neuter_values}
        mask = df[args.latin_gender_col].astype(str).str.strip().str.lower().isin(allowed)
        before = len(df)
        df = df.loc[mask].reset_index(drop=True)
        rprint(f"[bold magenta]Filter:[/] kept {len(df)} / {before} rows with Latin neuter")

    # Calculate class weights
    y = df[args.label_col].map(normalize_label).astype(int).values
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    rprint(f"[bold yellow]Class weights:[/] {dict(zip(['masculine', 'feminine'], class_weights))}")
    
    splits = make_folds(df, y, args.folds, args.seed, group_col=args.group_col)

    sent_tok = AutoTokenizer.from_pretrained(args.pretrained)
    fields_tok = AutoTokenizer.from_pretrained(args.pretrained)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rprint(f"[bold red]== ABLATION: exclude_latin=True  (fields = Occitan word only) ==[/]")

    results = []
    
    for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
        rprint(f"[bold yellow]Fold {fold}/{args.folds}[/] train={len(tr_idx)} val={len(va_idx)}")
        df_tr = df.iloc[tr_idx].reset_index(drop=True)
        df_va = df.iloc[va_idx].reset_index(drop=True)

        # DataLoaders  — exclude_latin=True strips Latin lemma & gender from fields
        tr_loader = build_loader(df_tr, sent_tok, fields_tok,
                                args.sentence_col, args.noun_index_col, args.label_col,
                                args.occitan_word_col, args.latin_lemma_col, args.latin_gender_col,
                                max_len_sent=args.max_len_sent, max_len_fields=args.max_len_fields,
                                batch_size=args.batch_size, shuffle=True,
                                index_in_whitespace=bool(args.index_in_whitespace),
                                mask_noun=False, masked_token=args.masked_token,
                                exclude_latin=True)
        
        va_loader = build_loader(df_va, sent_tok, fields_tok,
                                args.sentence_col, args.noun_index_col, args.label_col,
                                args.occitan_word_col, args.latin_lemma_col, args.latin_gender_col,
                                max_len_sent=args.max_len_sent, max_len_fields=args.max_len_fields,
                                batch_size=args.batch_size, shuffle=False,
                                index_in_whitespace=bool(args.index_in_whitespace),
                                mask_noun=False, masked_token=args.masked_token,
                                exclude_latin=True)

        # Exp2 Model (same architecture — only the fields input changes)
        exp2 = ContextReaderModel(pretrained=args.pretrained, hidden=args.hidden, dropout=args.dropout,
                                  heads=args.heads, dk=args.dk, dv=args.dv,
                                  use_rel_bias=bool(args.use_rel_bias),
                                  rel_window=64, layer_kv=args.layer_kv,
                                  freeze_encoder=False,
                                  no_self_peek=bool(args.no_self_peek))

        # Train
        rprint(f"[bold green]Training Exp2-NoLatin (ContextReader, no Latin fields) - Fold {fold}[/]")
        exp2 = train_model(exp2, tr_loader, va_loader, 
                          lr=args.lr, weight_decay=args.weight_decay,
                          epochs=args.epochs, warmup_ratio=args.warmup_ratio, device=device,
                          experiment_name="Exp2_NoLatin", fold=fold, 
                          class_weights=class_weights, save_dir=outdir, patience=3)

        # Evaluate
        val_loss, val_acc, val_f1 = eval_model(exp2, va_loader, device)
        results.append({
            "fold": fold,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1
        })
        
        rprint(f"[bold cyan]Fold {fold} Results:[/] Loss={val_loss:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f}")

        # Generate attention visualizations
        rprint(f"[bold magenta]Generating attention visualizations for fold {fold}...[/]")
        visualize_attention(exp2, va_loader, sent_tok, device, outdir, fold, num_samples=5)

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(outdir, "exp2_no_latin_results.csv"), index=False)
    
    # Summary statistics
    summary = {
        "experiment": "Exp2_NoLatin",
        "description": "ContextReaderModel with Latin lemma and Latin gender EXCLUDED from fields. "
                       "Fields encoder receives only the Occitan word.",
        "mean_val_loss": float(results_df["val_loss"].mean()),
        "std_val_loss": float(results_df["val_loss"].std()),
        "mean_val_acc": float(results_df["val_acc"].mean()),
        "std_val_acc": float(results_df["val_acc"].std()),
        "mean_val_f1": float(results_df["val_f1"].mean()),
        "std_val_f1": float(results_df["val_f1"].std()),
        "best_fold": int(results_df["val_f1"].idxmax() + 1),
        "best_f1": float(results_df["val_f1"].max())
    }
    
    with open(os.path.join(outdir, "exp2_no_latin_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    rprint(f"[bold green]Exp2 No-Latin Training Complete![/]")
    rprint(f"[bold cyan]Mean F1:[/] {summary['mean_val_f1']:.4f} +/- {summary['std_val_f1']:.4f}")
    rprint(f"[bold cyan]Best Fold:[/] {summary['best_fold']} (F1: {summary['best_f1']:.4f})")
    rprint(f"[bold cyan]Results saved to:[/] {outdir}")

if __name__ == "__main__":
    main()
