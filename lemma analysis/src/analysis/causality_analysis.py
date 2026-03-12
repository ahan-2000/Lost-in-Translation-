"""
Causality Analysis
Refactored from Causality.ipynb
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


def load_data(df_path="new_data_clean.csv"):
    """
    Load cleaned dataset and prepare for causality analysis.
    
    Args:
        df_path: Path to cleaned CSV file
        
    Returns:
        DataFrame with processed data
    """
    df = pd.read_csv(df_path)
    df = df[['Lemma_std', 'Genus_ok']].dropna()
    df['Genus_ok'] = df['Genus_ok'].str.lower().map({'f': 0, 'm': 1})
    df = df.dropna().rename(columns={'Lemma_std': 'lemma', 'Genus_ok': 'gender'})
    
    print(f"Dataset size: {len(df)}")
    return df


def train_suffix_classifier(df):
    """
    Train suffix-based classifier for counterfactual analysis.
    
    Args:
        df: DataFrame with lemma and gender columns
        
    Returns:
        Trained classifier and vectorizer
    """
    X = df['lemma'].str.lower().str.strip().str[-2:]
    y = df['gender']
    
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 2))
    X_vec = vectorizer.fit_transform(X)
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_vec, y)
    
    print("clf.classes_ (class ordering):", clf.classes_)
    
    return clf, vectorizer


def predict_proba(word, clf, vectorizer):
    """
    Predict probability of female class for a word.
    
    Args:
        word: Input word string
        clf: Trained classifier
        vectorizer: Fitted vectorizer
        
    Returns:
        Probability of female class
    """
    suffix = str(word)[-2:]
    vec = vectorizer.transform([suffix])
    return float(clf.predict_proba(vec)[0, 0])


def generate_counterfactual_pairs(df, clf, vectorizer, candidate_suffixes=None):
    """
    Generate counterfactual pairs for causality analysis.
    
    Args:
        df: DataFrame with lemmas
        clf: Trained classifier
        vectorizer: Fitted vectorizer
        candidate_suffixes: List of candidate suffixes to test
        
    Returns:
        DataFrame with counterfactual pairs and ITE values
    """
    if candidate_suffixes is None:
        candidate_suffixes = ["us", "um", "es", "er", "a"]
    
    pairs = []
    for lemma in df['lemma'].str.lower().unique():
        stem = lemma[:-2] if len(lemma) > 2 else lemma
        original_suffix = lemma[-2:] if len(lemma) >= 2 else ''
        control_form = lemma
        control_prob = predict_proba(control_form, clf, vectorizer)
        
        for suffix in candidate_suffixes:
            if suffix == original_suffix:
                continue
            treated_form = stem + suffix
            treated_prob = predict_proba(treated_form, clf, vectorizer)
            
            pairs.append({
                'lemma': lemma,
                'control_form': control_form,
                'control_suffix': original_suffix,
                'treated_form': treated_form,
                'treated_suffix': suffix,
                'control_prob': control_prob,
                'treated_prob': treated_prob,
                'ITE': treated_prob - control_prob
            })
    
    pairs_df = pd.DataFrame(pairs)
    print(f"Generated counterfactual pairs: {len(pairs_df)}")
    return pairs_df


def compute_ate_and_flip_rate(pairs_df):
    """
    Compute Average Treatment Effect (ATE) and flip rate.
    
    Args:
        pairs_df: DataFrame with counterfactual pairs
        
    Returns:
        ATE value and flip rate
    """
    ATE = pairs_df['ITE'].mean()
    
    pairs_df['control_label'] = (pairs_df['control_prob'] >= 0.5).astype(int)
    pairs_df['treated_label'] = (pairs_df['treated_prob'] >= 0.5).astype(int)
    flip_rate = (pairs_df['control_label'] != pairs_df['treated_label']).mean()
    
    print("\nTreatment Effect Summary:")
    print(f"ATE: {ATE:.4f}")
    print(f"Flip Rate: {flip_rate:.4f}")
    
    return ATE, flip_rate


def visualize_causality_results(pairs_df, ATE):
    """
    Create visualizations for causality analysis.
    
    Args:
        pairs_df: DataFrame with counterfactual pairs
        ATE: Average Treatment Effect
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Histogram of ITEs
    axes[0, 0].hist(pairs_df['ITE'], bins=20, color='skyblue', alpha=0.7)
    axes[0, 0].axvline(ATE, color='red', linestyle='--', label=f'ATE={ATE:.3f}')
    axes[0, 0].set_title("Distribution of Individual Treatment Effects (ITE)")
    axes[0, 0].set_xlabel("ITE (Δ P(female))")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].legend()
    
    # Control vs Treated probabilities
    axes[0, 1].scatter(pairs_df['control_prob'], pairs_df['treated_prob'], alpha=0.5)
    axes[0, 1].plot([0, 1], [0, 1], 'r--')
    axes[0, 1].set_xlabel("Control P(female)")
    axes[0, 1].set_ylabel("Treated P(female)")
    axes[0, 1].set_title("Control vs Treated Probabilities")
    
    # ITE vs Control probability
    axes[1, 0].scatter(pairs_df['control_prob'], pairs_df['ITE'], alpha=0.5)
    axes[1, 0].axhline(0, color='red', linestyle='--')
    axes[1, 0].set_xlabel("Control P(female)")
    axes[1, 0].set_ylabel("ITE (treated - control)")
    axes[1, 0].set_title("ITE vs Control P(female)")
    
    # Per-suffix ATE
    suffix_ate = pairs_df.groupby("treated_suffix")["ITE"].mean().sort_values()
    colors = ["green" if v > 0 else "red" for v in suffix_ate]
    bars = axes[1, 1].bar(suffix_ate.index, suffix_ate.values, color=colors)
    for bar, val in zip(bars, suffix_ate.values):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f"{val:.2f}", ha='center', va='bottom' if val >= 0 else 'top')
    axes[1, 1].axhline(0, color="black", linewidth=1)
    axes[1, 1].set_title("Per-Suffix Average Treatment Effect (ATE)")
    axes[1, 1].set_xlabel("Suffix")
    axes[1, 1].set_ylabel("Mean ITE (Δ P(female))")
    
    plt.tight_layout()
    plt.savefig("causality_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_per_suffix_ate(pairs_df):
    """Plot per-suffix Average Treatment Effect."""
    suffix_ate = pairs_df.groupby("treated_suffix")["ITE"].mean().sort_values()
    
    plt.figure(figsize=(8, 5))
    colors = ["green" if v > 0 else "red" for v in suffix_ate]
    bars = plt.bar(suffix_ate.index, suffix_ate.values, color=colors)
    
    for bar, val in zip(bars, suffix_ate.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{val:.2f}", ha='center', va='bottom' if val >= 0 else 'top')
    
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Per-Suffix Average Treatment Effect (ATE)")
    plt.xlabel("Suffix")
    plt.ylabel("Mean ITE (Δ P(female))")
    plt.tight_layout()
    plt.savefig("per_suffix_ate.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_flip_rate_per_suffix(pairs_df):
    """Plot flip rate per suffix."""
    flip_rate_suffix = ((pairs_df["control_label"] != pairs_df["treated_label"])
                       .groupby(pairs_df["treated_suffix"]).mean().sort_values())
    
    plt.figure(figsize=(8, 5))
    colors = ["green" if v > 0.1 else "red" for v in flip_rate_suffix]
    bars = plt.bar(flip_rate_suffix.index, flip_rate_suffix.values, color=colors)
    
    for bar, val in zip(bars, flip_rate_suffix.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{val:.2%}", ha='center', va='bottom')
    
    plt.title("Per-Suffix Flip Rate (label change frequency)")
    plt.xlabel("Suffix")
    plt.ylabel("Flip Rate (%)")
    plt.tight_layout()
    plt.savefig("flip_rate_per_suffix.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_top_ites(pairs_df, n=10, positive=True):
    """
    Plot top N positive or negative ITEs.
    
    Args:
        pairs_df: DataFrame with counterfactual pairs
        n: Number of top items to show
        positive: If True, show positive ITEs, else negative
    """
    if positive:
        top = pairs_df.nlargest(n, 'ITE')[['lemma', 'control_form', 'treated_form', 'ITE']]
        title = f"Top {n} Positive ITEs (Most Feminine Suffix Changes)"
    else:
        top = pairs_df.nsmallest(n, 'ITE')[['lemma', 'control_form', 'treated_form', 'ITE']]
        title = f"Top {n} Negative ITEs (Most Masculine Suffix Changes)"
    
    plt.figure(figsize=(8, 5))
    color = 'lightcoral' if positive else 'skyblue'
    labels = top['treated_form'] + " (from " + top['control_form'] + ")"
    plt.barh(labels, top['ITE'], color=color)
    plt.xlabel("ITE (Δ P(female))")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.axvline(0, color='black', linestyle='--')
    plt.tight_layout()
    plt.savefig(f"top_{n}_{'positive' if positive else 'negative'}_ites.png", 
               dpi=300, bbox_inches='tight')
    plt.show()


def main(df_path="new_data_clean.csv"):
    """
    Main execution function for causality analysis.
    
    Args:
        df_path: Path to cleaned CSV file
    """
    print("=" * 70)
    print("CAUSALITY ANALYSIS")
    print("=" * 70)
    
    # Load data
    df = load_data(df_path)
    
    # Train suffix classifier
    clf, vectorizer = train_suffix_classifier(df)
    
    # Generate counterfactual pairs
    pairs_df = generate_counterfactual_pairs(df, clf, vectorizer)
    
    # Compute ATE and flip rate
    ATE, flip_rate = compute_ate_and_flip_rate(pairs_df)
    
    # Visualizations
    visualize_causality_results(pairs_df, ATE)
    plot_per_suffix_ate(pairs_df)
    plot_flip_rate_per_suffix(pairs_df)
    plot_top_ites(pairs_df, n=10, positive=True)
    plot_top_ites(pairs_df, n=5, positive=False)
    
    print("\nCausality analysis complete!")
    return pairs_df, ATE, flip_rate


if __name__ == "__main__":
    pairs_df, ATE, flip_rate = main()

