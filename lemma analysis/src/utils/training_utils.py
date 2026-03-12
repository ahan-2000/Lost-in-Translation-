"""
Training utilities for model training and evaluation
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


def create_cv_splits(X, y, groups, n_splits=10, output_file="new_cv_folds.pkl"):
    """
    Create stratified group k-fold splits.
    
    Args:
        X: Feature matrix
        y: Target labels
        groups: Group IDs for stratified splitting
        n_splits: Number of folds
        output_file: Output file path
        
    Returns:
        List of (train_idx, test_idx) tuples
    """
    print(f"\nCreating {n_splits}-fold stratified group CV splits...")
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = []
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        folds.append((train_idx, test_idx))
    
    with open(output_file, "wb") as f:
        pickle.dump(folds, f)
    print(f"Saved {n_splits}-fold splits to {output_file}")
    return folds


def load_cv_splits(input_file="new_cv_folds.pkl"):
    """Load pre-computed CV splits."""
    with open(input_file, "rb") as f:
        folds = pickle.load(f)
    return folds

