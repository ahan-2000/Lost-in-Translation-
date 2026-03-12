"""
Phase 1: Descriptive Statistics & Baselines
Refactored from Phase_1.ipynb
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report


def load_data(df_path="latin_occitan_cleaned.csv"):
    """
    Load cleaned dataset.
    
    Args:
        df_path: Path to cleaned CSV file
        
    Returns:
        DataFrame
    """
    df = pd.read_csv(df_path)
    return df


def compute_gender_shift_frequencies(df):
    """
    Compute and visualize gender shift frequencies.
    
    Args:
        df: DataFrame with gender information
        
    Returns:
        DataFrame with shift frequencies
    """
    df['gender_shift'] = df['Genus_lat'] + '→' + df['Genus_ok']
    sc = df['gender_shift'].value_counts().reset_index()
    sc.columns = ['Shift', 'Frequency']
    
    all_shifts = ['n→m', 'n→f', 'n→n', 'm→m', 'm→f', 'm→n', 'f→m', 'f→f', 'f→n']
    for s in all_shifts:
        if s not in sc['Shift'].values:
            sc.loc[len(sc)] = [s, 0]
    sc = sc.sort_values('Shift')
    
    # Visualize
    plt.figure(figsize=(10, 6))
    c = sns.barplot(data=sc, x='Shift', y='Frequency')
    for p in c.patches:
        c.annotate(f'{int(p.get_height())}', 
                  (p.get_x() + p.get_width() / 2.0, p.get_height() + 50), 
                  ha='center', fontsize=10)
    plt.title("Gender Shift Frequencies (Latin → Occitan)", fontsize=14)
    plt.xlabel("Gender Shift")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
    
    return sc


def analyze_lemma_endings(df_n):
    """
    Analyze distribution of lemma endings across gender shifts.
    
    Args:
        df_n: DataFrame with neuter nouns
    """
    df_n['ending'] = df_n['Lemma'].str[-2:]
    end_shift = pd.crosstab(df_n['ending'], df_n['gender_shift'])
    c_end = df_n['ending'].value_counts()
    c_end = c_end[c_end > 5].index
    end_shift = end_shift.loc[c_end]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(end_shift, cmap='YlGnBu', annot=True, fmt='d', 
                linewidths=0.5, cbar_kws={'label': 'Count'})
    plt.title("Distribution of Lemma Endings Across Gender Shifts", fontsize=16)
    plt.xlabel("Gender Shift")
    plt.ylabel("Lemma Ending (Last 2 Letters)")
    plt.tight_layout()
    plt.show()


def majority_class_baseline(df_n):
    """
    Implement majority-class baseline model.
    
    Args:
        df_n: DataFrame with neuter nouns
        
    Returns:
        Accuracy score
    """
    mc = df_n['Genus_ok'].mode()[0]
    print(f"\nMajority Occitan gender: {mc}")
    df_n['baseline_majority'] = mc
    m_acc = (df_n['baseline_majority'] == df_n['Genus_ok']).mean()
    print(f"Majority-class baseline accuracy: {m_acc:.4f}")
    return m_acc


def ending_based_baseline(df_n):
    """
    Implement most-common-ending baseline model.
    
    Args:
        df_n: DataFrame with neuter nouns
        
    Returns:
        Accuracy score
    """
    df_n['ending'] = df_n['Lemma'].str[-2:]
    end_gen = df_n.groupby('ending')['Genus_ok'].agg(lambda x: x.mode().iloc[0])
    df_n['baseline_ending'] = df_n['ending'].map(end_gen)
    end_acc = (df_n['baseline_ending'] == df_n['Genus_ok']).mean()
    print(f"Most-common-ending baseline accuracy: {end_acc:.4f}")
    return end_acc


def visualize_baseline_predictions(df_n):
    """
    Visualize baseline predictions.
    
    Args:
        df_n: DataFrame with neuter nouns and baseline predictions
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    a = sns.countplot(x='baseline_majority', data=df_n, ax=axes[0], 
                     hue=axes[0], palette='pastel', legend=False)
    for p in a.patches:
        a.annotate(f'{int(p.get_height())}', 
                  (p.get_x() + p.get_width() / 2.0, p.get_height() + 50), 
                  ha='center', fontsize=10)
    axes[0].set_title("Majority Class Baseline Prediction")
    axes[0].set_xlabel("Predicted Genus_ok")
    
    b = sns.countplot(x='baseline_ending', data=df_n, ax=axes[1], 
                     hue=axes[0], palette='crest', legend=False)
    for p in b.patches:
        b.annotate(f'{int(p.get_height())}', 
                  (p.get_x() + p.get_width() / 2.0, p.get_height() + 50), 
                  ha='center', fontsize=10)
    axes[1].set_title("Most-Common-Ending-based Baseline Prediction")
    axes[1].set_xlabel("Predicted Genus_ok")
    
    plt.tight_layout()
    plt.show()


def evaluate_baselines(df_n, m_acc, end_acc):
    """
    Evaluate baseline models and print results.
    
    Args:
        df_n: DataFrame with neuter nouns
        m_acc: Majority class accuracy
        end_acc: Ending-based accuracy
    """
    total = len(df_n)
    mis_major_count = (df_n['baseline_majority'] != df_n['Genus_ok']).sum()
    mis_end_count = (df_n['baseline_ending'] != df_n['Genus_ok']).sum()
    
    mis_major_percent = (mis_major_count / total) * 100
    mis_end_percent = (mis_end_count / total) * 100
    
    baseline_results = pd.DataFrame({
        'Model': ['Majority Class', 'Most Common Ending (2-letter)'],
        'Accuracy': [m_acc, end_acc],
        'Misclassified %': [mis_major_percent, mis_end_percent]
    })
    
    print("\nBaseline Results:")
    print(baseline_results)
    
    # F1 scores
    print("\nF1 Report for Majority-Class Baseline:")
    print(classification_report(df_n['Genus_ok'], df_n['baseline_majority'], 
                               labels=['m', 'f'], zero_division=0))
    
    print("\nF1 Report for Ending-Based Baseline:")
    print(classification_report(df_n['Genus_ok'], df_n['baseline_ending'], 
                               labels=['m', 'f'], zero_division=0))
    
    return baseline_results


def identify_ambiguous_endings(df_n):
    """
    Identify ambiguous endings that map to multiple genders.
    
    Args:
        df_n: DataFrame with neuter nouns
    """
    end = df_n.groupby('ending')['Genus_ok'].nunique().reset_index()
    end.columns = ['Ending', 'Distinct_Genders']
    amb_end = end[end['Distinct_Genders'] > 1]
    
    baseline_map = df_n.groupby('ending')['baseline_ending'].first().reset_index()
    amb_end = amb_end.merge(baseline_map, left_on='Ending', right_on='ending', how='left')
    amb_end = amb_end.drop(columns='ending')
    amb_end.columns = ['Ending', 'Distinct_Genders', 'Predicted_Genus_ok']
    
    print("\nAmbiguous endings (mapped to more than one gender):")
    print(amb_end.sort_values('Distinct_Genders', ascending=False).head(10))


def main(df_path="latin_occitan_cleaned.csv"):
    """
    Main execution function for Phase 1.
    
    Args:
        df_path: Path to cleaned CSV file
        
    Returns:
        DataFrame with baseline predictions and results
    """
    print("=" * 70)
    print("PHASE 1: DESCRIPTIVE STATISTICS & BASELINES")
    print("=" * 70)
    
    # Load data
    df = load_data(df_path)
    
    # Compute gender shift frequencies
    sc = compute_gender_shift_frequencies(df)
    
    # Filter neuter nouns
    df_n = df[df['Genus_lat'] == 'n'].copy()
    print(f"\nNumber of neuter nouns: {len(df_n)}")
    
    # Analyze lemma endings
    analyze_lemma_endings(df_n)
    
    # Implement baselines
    m_acc = majority_class_baseline(df_n)
    end_acc = ending_based_baseline(df_n)
    
    # Visualize predictions
    visualize_baseline_predictions(df_n)
    
    # Evaluate baselines
    baseline_results = evaluate_baselines(df_n, m_acc, end_acc)
    
    # Identify ambiguous endings
    identify_ambiguous_endings(df_n)
    
    return df_n, baseline_results


if __name__ == "__main__":
    df_n, baseline_results = main()

