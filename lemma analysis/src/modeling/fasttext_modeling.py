"""
FastText Modeling Pipeline
Refactored from FT_Phase_3_4.ipynb
"""

import pickle
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modeling.base_modeling import (
    create_cv_splits, train_logistic_regression, 
    train_random_forest, train_xgboost
)
from utils.models import BiLSTMClassifier, FFN


def train_bilstm(X, y, cv_splits, model_name="FastText", output_dir="FT_Models",
                 hidden_dim=128, num_layers=2, dropout=0.3, num_epochs=50, batch_size=32):
    """Train BiLSTM with cross-validation."""
    print(f"\nTraining BiLSTM: {model_name}")
    os.makedirs(f"{output_dir}/BiLSTM", exist_ok=True)
    
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(np.unique(y))
    
    all_y_true, all_y_pred = [], []
    fold_acc, fold_f1 = [], []
    
    for fold, (train_idx, test_idx) in enumerate(cv_splits, 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        seq_len, emb_dim = X_train.shape[1], X_train.shape[2]
        
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).to(device)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).to(device)
        
        train_ds = TensorDataset(X_train_tensor, y_train_tensor)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        
        model = BiLSTMClassifier(input_dim=emb_dim, hidden_dim=hidden_dim,
                               output_dim=num_classes, num_layers=num_layers,
                               dropout=dropout).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        print(f"  Fold {fold}: Training for {num_epochs} epochs...")
        for epoch in tqdm(range(num_epochs), desc=f"Fold {fold}"):
            model.train()
            for xb, yb in train_dl:
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
        
        model.eval()
        with torch.no_grad():
            logits = model(X_test_tensor)
            y_pred = torch.argmax(logits, dim=1).cpu().numpy()
            y_true = y_test_tensor.cpu().numpy()
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        
        fold_acc.append(acc)
        fold_f1.append(f1)
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        
        print(f"  Fold {fold}: Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
        
        torch.save(model, f"{output_dir}/BiLSTM/Fold{fold}.pt")
        torch.save(model.state_dict(), f"{output_dir}/BiLSTM/Fold{fold}.pth")
    
    res_df = pd.DataFrame({
        "Fold": list(range(1, len(fold_acc) + 1)),
        "Accuracy": fold_acc,
        "Macro_F1": fold_f1
    })
    res_df.to_csv(f"{output_dir}/BiLSTM/bilstm_fold_results.csv", index=False)
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(res_df["Fold"], res_df["Accuracy"], label="Accuracy")
    plt.plot(res_df["Fold"], res_df["Macro_F1"], label="Macro F1")
    plt.xlabel("Fold")
    plt.ylabel("Score")
    plt.title(f"{model_name} - BiLSTM Fold-wise Accuracy & Macro F1")
    plt.ylim(0, 1)
    plt.xticks(range(1, len(fold_acc)+1))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/BiLSTM/bilstm_accuracy_plot.png")
    plt.close()
    
    # Confusion matrix
    decoded_true = le.inverse_transform(all_y_true)
    decoded_pred = le.inverse_transform(all_y_pred)
    labels = le.classes_
    cm = confusion_matrix(decoded_true, decoded_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("BiLSTM - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/BiLSTM/bilstm_confusion_matrix.png")
    plt.close()
    
    print(f"\nOverall - Mean Accuracy: {np.mean(fold_acc):.4f}, Mean Macro F1: {np.mean(fold_f1):.4f}")
    print("\nClassification Report:")
    print(classification_report(decoded_true, decoded_pred, target_names=le.classes_))
    
    return res_df


def train_fnn(X, y, cv_splits, model_name="FastText", output_dir="FT_Models",
              hidden_dim=128, num_epochs=50, batch_size=32):
    """Train Feedforward Neural Network."""
    print(f"\nTraining FFN: {model_name}")
    os.makedirs(f"{output_dir}/NN", exist_ok=True)
    
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(np.unique(y))
    
    all_y_true, all_y_pred = [], []
    fold_acc, fold_f1 = [], []
    
    for fold, (train_idx, test_idx) in enumerate(cv_splits, 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        input_dim = X_train.shape[1] * X_train.shape[2]
        
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).to(device)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).to(device)
        
        train_ds = TensorDataset(X_train_tensor, y_train_tensor)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        
        model = FFN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        print(f"  Fold {fold}: Training for {num_epochs} epochs...")
        for epoch in tqdm(range(num_epochs), desc=f"Fold {fold}"):
            model.train()
            for xb, yb in train_dl:
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
        
        model.eval()
        with torch.no_grad():
            logits = model(X_test_tensor)
            y_pred = torch.argmax(logits, dim=1).cpu().numpy()
            y_true = y_test_tensor.cpu().numpy()
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        
        fold_acc.append(acc)
        fold_f1.append(f1)
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        
        print(f"  Fold {fold}: Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
        
        torch.save(model, f"{output_dir}/NN/Fold{fold}.pt")
        torch.save(model.state_dict(), f"{output_dir}/NN/Fold{fold}.pth")
    
    res_df = pd.DataFrame({
        "Fold": list(range(1, len(fold_acc) + 1)),
        "Accuracy": fold_acc,
        "Macro_F1": fold_f1
    })
    res_df.to_csv(f"{output_dir}/NN/nn_fold_results.csv", index=False)
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(res_df["Fold"], res_df["Accuracy"], label="Accuracy")
    plt.plot(res_df["Fold"], res_df["Macro_F1"], label="Macro F1")
    plt.xlabel("Fold")
    plt.ylabel("Score")
    plt.title(f"{model_name} - NN Fold-wise Accuracy & Macro F1")
    plt.ylim(0, 1)
    plt.xticks(range(1, len(fold_acc)+1))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/NN/nn_accuracy_plot.png")
    plt.close()
    
    # Confusion matrix
    decoded_true = le.inverse_transform(all_y_true)
    decoded_pred = le.inverse_transform(all_y_pred)
    labels = le.classes_
    cm = confusion_matrix(decoded_true, decoded_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("NN - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/NN/nn_confusion_matrix.png")
    plt.close()
    
    print(f"\nOverall - Mean Accuracy: {np.mean(fold_acc):.4f}, Mean Macro F1: {np.mean(fold_f1):.4f}")
    print("\nClassification Report:")
    print(classification_report(decoded_true, decoded_pred, target_names=le.classes_))
    
    return res_df


def main():
    """Main execution function for FastText modeling."""
    print("=" * 70)
    print("FASTTEXT MODELING PIPELINE")
    print("=" * 70)
    
    # Load data
    print("Loading data...")
    with open("x_ft_1.pkl", "rb") as f:
        X = pickle.load(f)
    with open("new_y.pkl", "rb") as f:
        y = pickle.load(f)
    with open("new_lexeme_ids.pkl", "rb") as f:
        groups = pickle.load(f)
    
    print("Data loaded successfully!")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels shape: {len(y)}")
    print(f"Lexeme IDs shape: {len(groups)}\n")
    
    # Create CV splits
    cv_splits = create_cv_splits(X, y, groups, output_file="new_cv_folds.pkl")
    
    # Train models
    train_logistic_regression(X, y, cv_splits, "FastText", "FT_Models")
    train_random_forest(X, y, cv_splits, "FastText", "FT_Models")
    train_xgboost(X, y, cv_splits, "FastText", "FT_Models")
    
    # Load sequence data for neural networks
    with open("x_ft_2.pkl", "rb") as f:
        X_seq = pickle.load(f)
    
    train_bilstm(X_seq, y, cv_splits, "FastText", "FT_Models")
    train_fnn(X_seq, y, cv_splits, "FastText", "FT_Models")
    
    print("\nFastText modeling complete!")


if __name__ == "__main__":
    main()

