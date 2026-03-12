"""
Base modeling functions for all embedding types
"""

import pickle
import numpy as np
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from collections import Counter
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def create_cv_splits(X, y, groups, n_splits=10, output_file="new_cv_folds.pkl"):
    """Create stratified group k-fold splits."""
    from sklearn.model_selection import StratifiedGroupKFold
    
    print("Creating 10-fold stratified group CV splits...")
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = []
    
    print(f"Number of CV folds: {n_splits}")
    print(f"Total samples: {len(y)}")
    print(f"Total unique groups: {len(np.unique(groups))}")
    print(f"Overall class distribution: {dict(Counter(y))}")
    print("Split information:")
    print("-" * 70)
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        folds.append((train_idx, test_idx))
        
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        train_groups = np.unique(groups[train_idx])
        test_groups = np.unique(groups[test_idx])
        
        print(f"Fold {fold + 1}:")
        print(f"  Train: {len(train_idx):4d} samples, {len(train_groups):3d} groups")
        print(f"  Test:  {len(test_idx):4d} samples, {len(test_groups):3d} groups")
        
        train_dist = dict(Counter(y_train))
        test_dist = dict(Counter(y_test))
        print(f"  Train class dist: {train_dist}")
        print(f"  Test class dist:  {test_dist}")
        
        group_overlap = set(train_groups) & set(test_groups)
        if group_overlap:
            print(f" WARNING: Group leakage detected! {len(group_overlap)} groups in both train/test")
        else:
            print(f" No group leakage")
        print()
    
    with open(output_file, "wb") as f:
        pickle.dump(folds, f)
    
    print(f"10-fold StratifiedGroupKFold splits saved to {output_file}")
    return folds


def train_logistic_regression(X, y, cv_splits, model_name="Model", output_dir="Models"):
    """Train Logistic Regression with cross-validation."""
    print(f"\nTraining Logistic Regression: {model_name}")
    os.makedirs(f"{output_dir}/LR", exist_ok=True)
    
    all_y_true, all_y_pred = [], []
    fold_acc, fold_macro_f1 = [], []
    
    for fold, (train_idx, test_idx) in enumerate(cv_splits, 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        clf = LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced')
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        
        fold_acc.append(acc)
        fold_macro_f1.append(f1)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        
        print(f"  Fold {fold}: Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
        
        joblib.dump(clf, f"{output_dir}/LR/Fold{fold}.pkl")
        joblib.dump(scaler, f"{output_dir}/LR/Fold{fold}_scaler.pkl")
    
    res_df = pd.DataFrame({
        'Fold': list(range(1, len(fold_acc)+1)),
        'Accuracy': fold_acc,
        'Macro_F1': fold_macro_f1
    })
    res_df.to_csv(f"{output_dir}/LR/lr_fold_results.csv", index=False)
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(res_df['Fold'], res_df['Accuracy'], label='Accuracy')
    plt.plot(res_df['Fold'], res_df['Macro_F1'], label='Macro F1')
    plt.title(f"{model_name} - Logistic Regression: Accuracy and Macro F1 Across Folds")
    plt.xlabel("Fold")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(range(1, len(fold_acc)+1))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/LR/lr_accuracy_plot.png")
    plt.close()
    
    # Confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=np.unique(all_y_true), yticklabels=np.unique(all_y_true))
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/LR/lr_confusion_matrix.png")
    plt.close()
    
    print(f"\nOverall - Mean Accuracy: {np.mean(fold_acc):.4f}, Mean Macro F1: {np.mean(fold_macro_f1):.4f}")
    print("\nClassification Report:")
    print(classification_report(all_y_true, all_y_pred))
    
    return res_df


def train_random_forest(X, y, cv_splits, model_name="Model", output_dir="Models"):
    """Train Random Forest with cross-validation."""
    print(f"\nTraining Random Forest: {model_name}")
    os.makedirs(f"{output_dir}/RF", exist_ok=True)
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    joblib.dump(le, f"{output_dir}/RF/label_encoder.pkl")
    
    all_y_true, all_y_pred = [], []
    fold_acc, fold_macro_f1 = [], []
    
    for fold, (train_idx, test_idx) in enumerate(cv_splits, 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]
        
        rf = RandomForestClassifier(n_estimators=200, max_depth=None,
                                   random_state=42, class_weight="balanced")
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        
        fold_acc.append(acc)
        fold_macro_f1.append(f1)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        
        print(f"  Fold {fold}: Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
        joblib.dump(rf, f"{output_dir}/RF/Fold{fold}.pkl")
    
    res_df = pd.DataFrame({
        'Fold': list(range(1, len(fold_acc)+1)),
        'Accuracy': fold_acc,
        'Macro_F1': fold_macro_f1
    })
    res_df.to_csv(f"{output_dir}/RF/rf_fold_results.csv", index=False)
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(res_df['Fold'], res_df['Accuracy'], label='Accuracy')
    plt.plot(res_df['Fold'], res_df['Macro_F1'], label='Macro F1')
    plt.title(f"{model_name} - Random Forest: Accuracy and Macro F1 Across Folds")
    plt.xlabel("Fold")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(range(1, len(fold_acc)+1))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/RF/rf_accuracy_plot.png")
    plt.close()
    
    # Confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/RF/rf_confusion_matrix.png")
    plt.close()
    
    print(f"\nOverall - Mean Accuracy: {np.mean(fold_acc):.4f}, Mean Macro F1: {np.mean(fold_macro_f1):.4f}")
    print("\nClassification Report:")
    print(classification_report(all_y_true, all_y_pred, target_names=le.classes_))
    
    return res_df


def train_xgboost(X, y, cv_splits, model_name="Model", output_dir="Models"):
    """Train XGBoost with cross-validation."""
    print(f"\nTraining XGBoost: {model_name}")
    os.makedirs(f"{output_dir}/XGB", exist_ok=True)
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    joblib.dump(le, f"{output_dir}/XGB/label_encoder.pkl")
    
    all_y_true, all_y_pred = [], []
    fold_acc, fold_macro_f1 = [], []
    
    for fold, (train_idx, test_idx) in enumerate(cv_splits, 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]
        
        classes, counts = np.unique(y_train, return_counts=True)
        scale_pos_weight = counts[0] / counts[1] if len(classes) == 2 and counts[1] != 0 else 1.0
        
        clf = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                           subsample=0.8, colsample_bytree=0.8, random_state=42,
                           scale_pos_weight=scale_pos_weight, eval_metric='logloss',
                           use_label_encoder=False)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        
        fold_acc.append(acc)
        fold_macro_f1.append(f1)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        
        print(f"  Fold {fold}: Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
        joblib.dump(clf, f"{output_dir}/XGB/Fold{fold}.pkl")
    
    res_df = pd.DataFrame({
        'Fold': list(range(1, len(fold_acc)+1)),
        'Accuracy': fold_acc,
        'Macro_F1': fold_macro_f1
    })
    res_df.to_csv(f"{output_dir}/XGB/xgb_fold_results.csv", index=False)
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(res_df['Fold'], res_df['Accuracy'], label='Accuracy')
    plt.plot(res_df['Fold'], res_df['Macro_F1'], label='Macro F1')
    plt.title(f"{model_name} - XGBoost: Accuracy and Macro F1 Across Folds")
    plt.xlabel("Fold")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(range(1, len(fold_acc)+1))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/XGB/xgb_accuracy_plot.png")
    plt.close()
    
    # Confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/XGB/xgb_confusion_matrix.png")
    plt.close()
    
    print(f"\nOverall - Mean Accuracy: {np.mean(fold_acc):.4f}, Mean Macro F1: {np.mean(fold_macro_f1):.4f}")
    print("\nClassification Report:")
    print(classification_report(all_y_true, all_y_pred, target_names=le.classes_))
    
    return res_df

