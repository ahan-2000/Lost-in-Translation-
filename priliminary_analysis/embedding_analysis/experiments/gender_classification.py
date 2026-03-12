"""Gender Classification Experiment

Predict Occitan gender (Genus_ok) from surface form (Akk.-Sing.) under a 
leave-lemma-out split using various backends: TF-IDF, FastText, mBERT, ByT5.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, matthews_corrcoef,
    roc_auc_score, classification_report
)
from sklearn.exceptions import ConvergenceWarning

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.encoders import TFIDFEncoder, FastTextEncoder, hf_encode
from src.utils import normalize_texts
from src.data_processing import load_data, get_splits, create_label_mapping

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def train_eval(Xtr, ytr, Xte, yte, classes):
    """
    Train logistic regression classifier and evaluate.
    
    Returns:
        Dictionary of metrics and trained classifier
    """
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, ytr)
    ypr = clf.predict(Xte)
    yprob = clf.predict_proba(Xte)
    
    acc = accuracy_score(yte, ypr)
    macro_f1 = f1_score(yte, ypr, average="macro")
    mcc = matthews_corrcoef(yte, ypr)
    
    # Handle AUROC for binary vs multiclass
    if yprob.shape[1] == 2:
        auroc = roc_auc_score(yte, yprob[:, 1])
    else:
        auroc = roc_auc_score(yte, yprob, average="macro", multi_class="ovr")
    
    cm = confusion_matrix(yte, ypr)
    report = classification_report(yte, ypr, target_names=classes, output_dict=True)
    
    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "mcc": mcc,
        "auroc": auroc,
        "cm": cm,
        "report": report
    }, clf


def plot_confusion_matrix(cm, classes, backend_name, out_dir):
    """Plot and save confusion matrix."""
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title(f"Confusion Matrix — {backend_name}")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center")
    
    ax.set_ylabel('True')
    ax.set_xlabel('Pred')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"confusion_matrix_{backend_name}.png"), dpi=150)
    plt.close()


def plot_macro_f1_bar(metrics_df, out_dir):
    """Plot macro-F1 bar chart."""
    plt.figure(figsize=(6, 4))
    plt.bar(metrics_df['backend'], metrics_df['macro_f1'])
    plt.title("Macro-F1 by backend")
    plt.xlabel("Backend")
    plt.ylabel("Macro-F1")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "macro_f1_comparison.png"), dpi=150)
    plt.close()


def main():
    """Main experiment function."""
    # Configuration - adjust paths based on where script is run from
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    DATA_PATH = os.path.join(project_root, "latin_occitan_with_variations.csv")
    SPLIT_PATH = os.path.join(project_root, "latin_occitan_group_split.csv")
    OUT_DIR = os.path.join(script_dir, "..", "results", "gender_results")
    OUT_DIR = os.path.abspath(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)
    
    LOWERCASE = False
    STRIP_ACCENTS = False
    
    # Load data
    _, df = load_data(DATA_PATH, SPLIT_PATH)
    train, val, test = get_splits(df)
    
    # Normalize texts
    xtr = normalize_texts(train['Akk.-Sing.'].tolist(), LOWERCASE, STRIP_ACCENTS)
    xte = normalize_texts(test['Akk.-Sing.'].tolist(), LOWERCASE, STRIP_ACCENTS)
    
    # Prepare labels
    classes = sorted(df['Genus_ok'].unique().tolist())
    lab2id = create_label_mapping(classes)
    ytr = np.array([lab2id[x] for x in train['Genus_ok']], dtype=np.int64)
    yte = np.array([lab2id[x] for x in test['Genus_ok']], dtype=np.int64)
    
    print(f"Train: {len(xtr)}, Test: {len(xte)}, Classes: {classes}")
    
    # Backend configuration
    backends = {
        "tfidf": True,
        "fasttext": True,
        "mbert": True,
        "byt5": True
    }
    
    results = []
    packs = {}
    
    # TF-IDF
    if backends["tfidf"]:
        print("Running TF-IDF...")
        tfv = TFIDFEncoder().fit(xtr + xte)
        Xtr = tfv.encode(xtr)
        Xte = tfv.encode(xte)
        m, _ = train_eval(Xtr, ytr, Xte, yte, classes)
        packs["tfidf"] = m
        results.append({
            "backend": "tfidf",
            **{k: float(v) for k, v in m.items() if k in ["acc", "macro_f1", "mcc", "auroc"]}
        })
        plot_confusion_matrix(m["cm"], classes, "tfidf", OUT_DIR)
    
    # FastText
    if backends["fasttext"]:
        print("Running FastText...")
        ft = FastTextEncoder().fit(xtr)
        Xtr = ft.encode(xtr)
        Xte = ft.encode(xte)
        m, _ = train_eval(Xtr, ytr, Xte, yte, classes)
        packs["fasttext"] = m
        results.append({
            "backend": "fasttext",
            **{k: float(v) for k, v in m.items() if k in ["acc", "macro_f1", "mcc", "auroc"]}
        })
        plot_confusion_matrix(m["cm"], classes, "fasttext", OUT_DIR)
    
    # mBERT
    if backends["mbert"]:
        print("Running mBERT...")
        Xtr = hf_encode("bert-base-multilingual-cased", xtr, pooling="mean")
        Xte = hf_encode("bert-base-multilingual-cased", xte, pooling="mean")
        m, _ = train_eval(Xtr, ytr, Xte, yte, classes)
        packs["mbert"] = m
        results.append({
            "backend": "mbert",
            **{k: float(v) for k, v in m.items() if k in ["acc", "macro_f1", "mcc", "auroc"]}
        })
        plot_confusion_matrix(m["cm"], classes, "mbert", OUT_DIR)
    
    # ByT5
    if backends["byt5"]:
        print("Running ByT5...")
        Xtr = hf_encode("google/byt5-base", xtr, pooling="mean")
        Xte = hf_encode("google/byt5-base", xte, pooling="mean")
        m, _ = train_eval(Xtr, ytr, Xte, yte, classes)
        packs["byt5"] = m
        results.append({
            "backend": "byt5",
            **{k: float(v) for k, v in m.items() if k in ["acc", "macro_f1", "mcc", "auroc"]}
        })
        plot_confusion_matrix(m["cm"], classes, "byt5", OUT_DIR)
    
    # Save results
    metrics_df = pd.DataFrame(results).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    metrics_df.to_csv(os.path.join(OUT_DIR, "metrics_summary.csv"), index=False)
    
    # Save per-class reports
    for name, m in packs.items():
        report_df = pd.DataFrame(m["report"]).T
        report_df.to_csv(os.path.join(OUT_DIR, f"classification_report_{name}.csv"))
    
    # Plot macro-F1 comparison
    plot_macro_f1_bar(metrics_df, OUT_DIR)
    
    print("\nResults:")
    print(metrics_df)
    print(f"\nResults saved to {OUT_DIR}")


if __name__ == "__main__":
    main()

