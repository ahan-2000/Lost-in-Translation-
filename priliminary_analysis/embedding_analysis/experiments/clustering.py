"""Clustering by Lemma Experiment

Embed forms, cluster with k-means, and evaluate alignment to gold Lemma.

Metrics: NMI, ARI, Homogeneity, Completeness, Silhouette.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import (
    normalized_mutual_info_score, adjusted_rand_score,
    homogeneity_score, completeness_score, silhouette_score
)
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.encoders import TFIDFEncoder, FastTextEncoder, hf_encode
from src.utils import normalize_texts

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Try FAISS for GPU-accelerated k-means
USE_FAISS = True
try:
    import faiss
    FAISS_OK = True
    FAISS_NUM_GPUS = getattr(faiss, "get_num_gpus", lambda: 0)()
except Exception:
    FAISS_OK = False
    FAISS_NUM_GPUS = 0


def _to_float32(X):
    """Convert array to float32 and ensure C-contiguous."""
    X = np.asarray(X, dtype=np.float32)
    if not X.flags["C_CONTIGUOUS"]:
        X = np.ascontiguousarray(X)
    return X


def kmeans_fit_predict(X, k, seed=0):
    """
    KMeans clustering with FAISS GPU support if available.
    
    Args:
        X: Feature matrix (n, d)
        k: Number of clusters
        seed: Random seed
    
    Returns:
        Cluster labels (np.int32)
    """
    X = _to_float32(X)
    
    if USE_FAISS and FAISS_OK:
        d = X.shape[1]
        km = faiss.Kmeans(
            d, k, niter=20, verbose=False, seed=seed,
            gpu=(FAISS_NUM_GPUS > 0)
        )
        km.train(X)
        
        # Assign to nearest centroid
        index = faiss.IndexFlatL2(d)
        if FAISS_NUM_GPUS > 0:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        index.add(km.centroids)
        D, I = index.search(X, 1)
        return I.ravel().astype(np.int32)
    else:
        # Fallback to sklearn
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        return km.fit_predict(X).astype(np.int32)


def embed_tfidf(texts):
    """Embed texts using TF-IDF."""
    enc = TFIDFEncoder()
    return enc.fit(texts).encode(texts).toarray().astype(np.float32)


def embed_fasttext(texts):
    """Embed texts using FastText."""
    enc = FastTextEncoder()
    return enc.fit(texts).encode(texts)


def embed_hf(model_name, texts, pooling="mean", max_length=32, batch_size=256):
    """Embed texts using HuggingFace model."""
    return hf_encode(model_name, texts, pooling, max_length, batch_size, DEVICE)


def evaluate_backend(name, texts_sub, lemma_ids, K_SHARED, seed=0):
    """
    Evaluate a single backend.
    
    Args:
        name: Backend name
        texts_sub: List of texts
        lemma_ids: Array of lemma IDs
        K_SHARED: Number of clusters
        seed: Random seed
    
    Returns:
        Dictionary of metrics
    """
    try:
        # Embed
        if name == "tfidf":
            X = embed_tfidf(texts_sub)
        elif name == "fasttext":
            X = embed_fasttext(texts_sub)
        elif name == "mbert":
            X = embed_hf("bert-base-multilingual-cased", texts_sub, pooling="mean")
        elif name == "byt5":
            X = embed_hf("google/byt5-base", texts_sub, pooling="mean")
        else:
            raise ValueError(f"Unknown backend: {name}")
        
        # Cluster
        labels = kmeans_fit_predict(X, K_SHARED, seed=seed)
        
        # Compute metrics
        nmi = normalized_mutual_info_score(lemma_ids, labels)
        ari = adjusted_rand_score(lemma_ids, labels)
        homo = homogeneity_score(lemma_ids, labels)
        comp = completeness_score(lemma_ids, labels)
        
        # Silhouette (sample for speed)
        try:
            n_samp = min(2000, len(X))
            rng = np.random.default_rng(seed)
            samp = rng.choice(len(X), size=n_samp, replace=False)
            sil = silhouette_score(X[samp], labels[samp])
        except Exception:
            sil = float("nan")
        
        return {
            "backend": name,
            "subset": len(texts_sub),
            "k": K_SHARED,
            "NMI": nmi,
            "ARI": ari,
            "Homogeneity": homo,
            "Completeness": comp,
            "Silhouette": sil,
            "faiss_gpu": bool(USE_FAISS and FAISS_OK and FAISS_NUM_GPUS > 0),
            "error": ""
        }
    except Exception as e:
        return {
            "backend": name,
            "subset": len(texts_sub),
            "k": K_SHARED,
            "NMI": np.nan,
            "ARI": np.nan,
            "Homogeneity": np.nan,
            "Completeness": np.nan,
            "Silhouette": np.nan,
            "faiss_gpu": bool(USE_FAISS and FAISS_OK and FAISS_NUM_GPUS > 0),
            "error": str(e)
        }


def plot_umap(X, colors, title, out_path, figsize=(6, 5)):
    """
    Plot UMAP projection.
    
    Args:
        X: 2D coordinates from UMAP
        colors: Color values for each point
        title: Plot title
        out_path: Path to save figure
        figsize: Figure size
    """
    try:
        import umap
    except ImportError:
        print("UMAP not available, skipping visualization")
        return
    
    plt.figure(figsize=figsize)
    plt.scatter(X[:, 0], X[:, 1], s=4, alpha=0.6, c=colors)
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    """Main experiment function."""
    # Configuration - adjust paths based on where script is run from
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    DATA_PATH = os.path.join(project_root, "latin_occitan_group_split.csv")
    OUT_DIR = os.path.join(script_dir, "..", "results", "clustering_results")
    OUT_DIR = os.path.abspath(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)
    
    LOWERCASE = False
    STRIP_ACCENTS = False
    
    # Load data
    df = pd.read_csv(DATA_PATH)
    texts = df['Akk.-Sing.'].tolist()
    lemmas = df['Lemma'].tolist()
    genders = pd.Categorical(df['Genus_ok']).codes
    
    # Normalize texts
    texts = normalize_texts(texts, LOWERCASE, STRIP_ACCENTS)
    
    # Create subset for fair comparison
    SUBSET = 8000 if len(texts) > 8000 else len(texts)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(texts), size=SUBSET, replace=False)
    texts_sub = [texts[i] for i in idx]
    lemmas_sub = [lemmas[i] for i in idx]
    genders_sub = [genders[i] for i in idx]
    
    lemma_ids = pd.Categorical(lemmas_sub).codes
    K_SHARED = min(len(np.unique(lemma_ids)), 2000)
    
    print(f"Subset size: {SUBSET}, Number of clusters: {K_SHARED}")
    
    # Evaluate all backends
    backends = ["tfidf", "fasttext", "mbert", "byt5"]
    rows = []
    
    for backend in backends:
        print(f"\nEvaluating {backend}...")
        result = evaluate_backend(backend, texts_sub, lemma_ids, K_SHARED, seed=0)
        rows.append(result)
        if result["error"]:
            print(f"  Error: {result['error']}")
        else:
            print(f"  NMI: {result['NMI']:.4f}, ARI: {result['ARI']:.4f}")
    
    metrics_all = pd.DataFrame(rows).sort_values(
        ["NMI", "ARI"], ascending=False
    ).reset_index(drop=True)
    
    metrics_all.to_csv(os.path.join(OUT_DIR, "clustering_metrics.csv"), index=False)
    print("\nClustering metrics:")
    print(metrics_all)
    
    # UMAP visualizations
    print("\nGenerating UMAP visualizations...")
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=0)
        
        for backend in backends:
            print(f"  Processing {backend}...")
            try:
                # Embed
                if backend == "tfidf":
                    X = embed_tfidf(texts_sub)
                elif backend == "fasttext":
                    X = embed_fasttext(texts_sub)
                elif backend == "mbert":
                    X = embed_hf("bert-base-multilingual-cased", texts_sub, pooling="mean")
                elif backend == "byt5":
                    X = embed_hf("google/byt5-base", texts_sub, pooling="mean")
                else:
                    continue
                
                # UMAP projection
                X2 = reducer.fit_transform(X)
                
                # Plot by gender
                plot_umap(
                    X2, genders_sub,
                    f"UMAP — colored by Genus_ok — {backend}",
                    os.path.join(OUT_DIR, f"umap_{backend}_gender.png")
                )
                
                # Plot by sampled lemmas
                uniq_ids = np.unique(lemma_ids)
                pick = set(rng.choice(uniq_ids, size=min(15, len(uniq_ids)), replace=False))
                colors = [lemma_ids[i] if lemma_ids[i] in pick else -1 for i in range(len(lemma_ids))]
                plot_umap(
                    X2, colors,
                    f"UMAP — sampled lemmas highlighted — {backend}",
                    os.path.join(OUT_DIR, f"umap_{backend}_lemmas.png")
                )
            except Exception as e:
                print(f"    Error: {e}")
    except ImportError:
        print("UMAP not available, skipping visualizations")
    
    # Cluster purity analysis
    print("\nComputing cluster purity...")
    for backend in backends:
        try:
            # Embed and cluster
            if backend == "tfidf":
                X = embed_tfidf(texts_sub)
            elif backend == "fasttext":
                X = embed_fasttext(texts_sub)
            elif backend == "mbert":
                X = embed_hf("bert-base-multilingual-cased", texts_sub, pooling="mean")
            elif backend == "byt5":
                X = embed_hf("google/byt5-base", texts_sub, pooling="mean")
            else:
                continue
            
            pred = kmeans_fit_predict(X, K_SHARED, seed=0)
            
            # Compute purity
            tab = pd.DataFrame({"pred": pred, "lemma": lemmas_sub})
            major = tab.groupby('pred')['lemma'].agg(lambda s: s.value_counts().idxmax())
            purity = tab.groupby('pred')['lemma'].agg(lambda s: s.value_counts().max() / len(s))
            sizes = tab['pred'].value_counts().rename("size")
            summary = pd.DataFrame({
                "majority_lemma": major,
                "purity": purity
            }).join(sizes)
            
            summary_sorted = summary.sort_values(
                ["purity", "size"], ascending=[False, False]
            ).head(15)
            
            summary_sorted.to_csv(
                os.path.join(OUT_DIR, f"cluster_purity_top15_{backend}.csv"),
                index=True
            )
        except Exception as e:
            print(f"  Error computing purity for {backend}: {e}")
    
    print(f"\nResults saved to {OUT_DIR}")


if __name__ == "__main__":
    main()

