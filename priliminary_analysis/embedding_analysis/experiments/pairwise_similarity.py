"""Pairwise Similarity & Retrieval Experiment

(1) Binary same-lemma? prediction via cosine similarity.
(2) Nearest-neighbor retrieval of forms from the same lemma.

Metrics: ROC-AUC, Average Precision; Recall@K, MRR, nDCG@10.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.neighbors import NearestNeighbors
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.encoders import TFIDFEncoder, FastTextEncoder, hf_encode
from src.utils import normalize_texts

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def create_pairs(df, pairs_path, rng_seed=42, max_pos=4000, max_neg=4000):
    """
    Create positive and negative pairs for similarity evaluation.
    
    Args:
        df: DataFrame with 'Lemma' and 'Akk.-Sing.' columns
        pairs_path: Path to save pairs CSV
        rng_seed: Random seed
        max_pos: Maximum positive pairs
        max_neg: Maximum negative pairs
    """
    rng = np.random.default_rng(rng_seed)
    pos, neg = [], []
    
    # Positive pairs: same lemma
    for lemma, g in df.groupby('Lemma'):
        if len(g) >= 2:
            idx = g.index.tolist()
            if len(idx) > 8:
                idx = list(rng.choice(idx, size=8, replace=False))
            for i in range(0, len(idx) - 1, 2):
                pos.append((
                    df.loc[idx[i], 'Akk.-Sing.'],
                    df.loc[idx[i+1], 'Akk.-Sing.'],
                    1
                ))
    
    # Negative pairs: different lemma, same gender
    for ok, g in df.groupby('Genus_ok'):
        idxs = g.index.to_list()
        if len(idxs) < 2:
            continue
        samp = list(rng.choice(idxs, size=min(len(idxs), 4000), replace=False))
        for i in range(0, len(samp) - 1, 2):
            i1, i2 = samp[i], samp[i+1]
            if df.loc[i1, 'Lemma'] != df.loc[i2, 'Lemma']:
                neg.append((
                    df.loc[i1, 'Akk.-Sing.'],
                    df.loc[i2, 'Akk.-Sing.'],
                    0
                ))
    
    pairs = pd.DataFrame(
        pos[:max_pos] + neg[:max_neg],
        columns=['text1', 'text2', 'label']
    )
    pairs.to_csv(pairs_path, index=False)
    return pairs


def pair_metrics(emb, pairs, index):
    """
    Compute pairwise similarity metrics.
    
    Returns:
        Dictionary with metrics and curves
    """
    sims, labs = [], []
    for _, r in pairs.iterrows():
        a, b, y = r['text1'], r['text2'], int(r['label'])
        ia, ib = index[a], index[b]
        va, vb = emb[ia], emb[ib]
        sim = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))
        sims.append(sim)
        labs.append(y)
    
    sims = np.array(sims)
    labs = np.array(labs)
    
    roc = roc_auc_score(labs, sims)
    ap = average_precision_score(labs, sims)
    fpr, tpr, _ = roc_curve(labs, sims)
    prec, rec, _ = precision_recall_curve(labs, sims)
    
    return {
        "roc_auc": roc,
        "ap": ap,
        "fpr": fpr,
        "tpr": tpr,
        "prec": prec,
        "rec": rec,
        "sims": sims,
        "labs": labs
    }


def retrieval_scores_gpu(X, lemma_ids, ks=(5, 10), batch_size=2048, device=None):
    """
    Compute retrieval metrics using GPU-accelerated computation.
    
    Args:
        X: Embedding matrix (n, d)
        lemma_ids: Array of lemma IDs for each text
        ks: Tuple of K values for Recall@K and nDCG@K
        batch_size: Batch size for GPU computation
        device: Device to use
    
    Returns:
        Dictionary of retrieval metrics
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    X = torch.tensor(X, dtype=torch.float32, device=device)
    X = torch.nn.functional.normalize(X, p=2, dim=1)
    
    n = X.shape[0]
    k_max = max(ks)
    first_pos = torch.zeros(n, dtype=torch.int32, device=device)
    lem = torch.tensor(lemma_ids, device=device)
    
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        Q = X[start:end]
        sims = Q @ X.T
        row_idx = torch.arange(start, end, device=device)
        sims[torch.arange(end - start, device=device), row_idx] = -1e9
        
        topk_scores, topk_idx = sims.topk(k=k_max, dim=1, largest=True)
        rel = (lem[topk_idx] == lem[row_idx][:, None])
        has_rel = rel.any(dim=1)
        pos = torch.where(
            has_rel,
            rel.float().argmax(dim=1) + 1,
            torch.zeros_like(has_rel, dtype=torch.int64)
        )
        first_pos[start:end] = pos.to(torch.int32)
        
        del sims, topk_scores, topk_idx, rel, has_rel
    
    first_pos = first_pos.cpu().numpy()
    mrr = np.where(first_pos > 0, 1.0 / first_pos, 0.0).mean()
    
    out = {"mrr": float(mrr)}
    for K in ks:
        ndcg = np.where(
            (first_pos > 0) & (first_pos <= K),
            1.0 / np.log2(first_pos + 1),
            0.0
        ).mean()
        recall = ((first_pos > 0) & (first_pos <= K)).mean()
        out[f"ndcg@{K}"] = float(ndcg)
        out[f"recall@{K}"] = float(recall)
    
    return out


def show_neighbors(X, texts, q_idx, topk=5):
    """Show nearest neighbors for a query text."""
    nn = NearestNeighbors(n_neighbors=min(topk + 1, len(texts)), metric="cosine").fit(X)
    dist, idxs = nn.kneighbors(X[q_idx:q_idx+1], return_distance=True)
    idxs = idxs[0].tolist()
    if idxs[0] == q_idx:
        idxs = idxs[1:]
    return [texts[j] for j in idxs[:topk]]


def plot_roc_curve(fpr, tpr, roc_auc, backend_name, out_dir):
    """Plot ROC curve."""
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], '--')
    plt.title(f"ROC — {backend_name} (AUC={roc_auc:.3f})")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"roc_{backend_name}.png"), dpi=150)
    plt.close()


def plot_pr_curve(rec, prec, ap, backend_name, out_dir):
    """Plot Precision-Recall curve."""
    plt.figure(figsize=(5, 4))
    plt.plot(rec, prec)
    plt.title(f"PR — {backend_name} (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"pr_{backend_name}.png"), dpi=150)
    plt.close()


def main():
    """Main experiment function."""
    # Configuration - adjust paths based on where script is run from
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    DATA_PATH = os.path.join(project_root, "latin_occitan_with_variations.csv")
    PAIRS_PATH = os.path.join(project_root, "latin_occitan_pairs_sample.csv")
    OUT_DIR = os.path.join(script_dir, "..", "results", "pairwise_results")
    OUT_DIR = os.path.abspath(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)
    
    LOWERCASE = False
    STRIP_ACCENTS = False
    
    # Load data and create pairs if needed
    df = pd.read_csv(DATA_PATH)
    if not os.path.exists(PAIRS_PATH):
        print("Creating pairs...")
        pairs = create_pairs(df, PAIRS_PATH)
    else:
        pairs = pd.read_csv(PAIRS_PATH)
    
    # Get unique texts and normalize
    uniq_texts = pd.unique(pd.concat([pairs['text1'], pairs['text2']], ignore_index=True)).tolist()
    uniq_norm = normalize_texts(uniq_texts, LOWERCASE, STRIP_ACCENTS)
    
    # Create index mapping
    index = {t: i for i, t in enumerate(uniq_texts)}
    
    # Backend configuration
    backends = {"tfidf": True, "fasttext": True, "mbert": True, "byt5": True}
    embeds = {}
    
    # Encode with each backend
    if backends["tfidf"]:
        print("Encoding with TF-IDF...")
        tfv = TFIDFEncoder().fit(uniq_norm)
        embeds["tfidf"] = tfv.encode(uniq_norm).toarray()
    
    if backends["fasttext"]:
        print("Encoding with FastText...")
        ft = FastTextEncoder().fit(uniq_norm)
        embeds["fasttext"] = ft.encode(uniq_norm)
    
    if backends["mbert"]:
        print("Encoding with mBERT...")
        embeds["mbert"] = hf_encode("bert-base-multilingual-cased", uniq_norm)
    
    if backends["byt5"]:
        print("Encoding with ByT5...")
        embeds["byt5"] = hf_encode("google/byt5-base", uniq_norm)
    
    # Pairwise similarity metrics
    print("\nComputing pairwise similarity metrics...")
    rows = []
    curves = {}
    for name, X in embeds.items():
        m = pair_metrics(X, pairs, index)
        rows.append({
            "backend": name,
            "roc_auc": m["roc_auc"],
            "average_precision": m["ap"]
        })
        curves[name] = m
        plot_roc_curve(m["fpr"], m["tpr"], m["roc_auc"], name, OUT_DIR)
        plot_pr_curve(m["rec"], m["prec"], m["ap"], name, OUT_DIR)
    
    metrics_df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    metrics_df.to_csv(os.path.join(OUT_DIR, "pairwise_metrics.csv"), index=False)
    print("\nPairwise metrics:")
    print(metrics_df)
    
    # Retrieval metrics
    print("\nComputing retrieval metrics...")
    full = pd.read_csv(DATA_PATH)
    lemma_map = full.groupby('Akk.-Sing.')['Lemma'].agg(
        lambda s: s.value_counts().idxmax()
    ).to_dict()
    lemma_ids = pd.Categorical([
        lemma_map.get(t, f"UNK_{i}") for i, t in enumerate(uniq_texts)
    ]).codes
    
    retr_rows = []
    for name, X in embeds.items():
        print(f"  Computing retrieval for {name}...")
        s = retrieval_scores_gpu(X, lemma_ids, ks=(5, 10), batch_size=2048, device=DEVICE)
        retr_rows.append({"backend": name, **s})
    
    retrieval_df = pd.DataFrame(retr_rows)
    retrieval_df.to_csv(os.path.join(OUT_DIR, "retrieval_metrics.csv"), index=False)
    print("\nRetrieval metrics:")
    print(retrieval_df)
    
    # Qualitative examples
    print("\nQualitative nearest neighbors:")
    np.random.seed(0)
    for name, X in embeds.items():
        print(f"\nBackend: {name}")
        for _ in range(3):
            i = np.random.randint(0, len(uniq_texts))
            print(f"  Query: {uniq_texts[i]}")
            print(f"  Neighbors: {show_neighbors(X, uniq_texts, i, topk=5)}")
    
    print(f"\nResults saved to {OUT_DIR}")


if __name__ == "__main__":
    main()

