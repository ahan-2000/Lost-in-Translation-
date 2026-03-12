"""Data loading and preprocessing utilities."""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List


def load_data(data_path: str, split_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data and create train/val/test split if needed.
    
    Args:
        data_path: Path to main data CSV
        split_path: Path to split CSV (created if doesn't exist)
    
    Returns:
        Tuple of (full dataframe, split dataframe)
    """
    df0 = pd.read_csv(data_path)
    
    # Create split if it doesn't exist (deterministic by lemma)
    if split_path is None or not os.path.exists(split_path):
        if split_path is None:
            split_path = data_path.replace('.csv', '_group_split.csv')
        
        rng = np.random.default_rng(42)
        lemmas = df0['Lemma'].unique()
        rng.shuffle(lemmas)
        n = len(lemmas)
        
        tr = set(lemmas[:int(0.7 * n)])
        va = set(lemmas[int(0.7 * n):int(0.8 * n)])
        te = set(lemmas[int(0.8 * n):])
        
        def assign_split(lemma):
            if lemma in tr:
                return "train"
            elif lemma in va:
                return "val"
            else:
                return "test"
        
        df0['split'] = df0['Lemma'].map(assign_split)
        df0.to_csv(split_path, index=False)
    
    df = pd.read_csv(split_path)
    return df0, df


def get_splits(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Get train, validation, and test splits.
    
    Args:
        df: DataFrame with 'split' column
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    train = df[df['split'] == "train"].reset_index(drop=True)
    val = df[df['split'] == "val"].reset_index(drop=True)
    test = df[df['split'] == "test"].reset_index(drop=True)
    return train, val, test


def create_label_mapping(classes: List[str]) -> Dict[str, int]:
    """
    Create label to ID mapping.
    
    Args:
        classes: List of class labels
    
    Returns:
        Dictionary mapping labels to integer IDs
    """
    if set(classes) == set(["m", "f"]):
        return {"m": 0, "f": 1}
    else:
        return {c: i for i, c in enumerate(sorted(classes))}

