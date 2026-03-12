"""
Phase 2: MMBert Feature Engineering
Refactored from MMBert_Phase_2.ipynb
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch


def load_data(df_path="new_data_clean.csv"):
    """Load cleaned dataset."""
    df = pd.read_csv(df_path)
    print(f"Loaded {len(df)} samples")
    return df


def generate_mmbert_embeddings(df):
    """Generate mmBERT embeddings for features."""
    print("\nGenerating mmBERT embeddings...")
    
    tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")
    model = AutoModel.from_pretrained("jhu-clsp/mmBERT-base")
    model.eval()
    
    char_cols = [
        "lat_ngram_1", "lat_ngram_2", "lat_ngram_3", "lat_ngram_4",
        "occ_ngram_1", "occ_ngram_2", "occ_ngram_3", "occ_ngram_4",
        "vc_lat", "vc_occ"
    ]
    num_cols = ["syl_lat", "syl_occ", "word_len", "frequency"]
    word_cols = ["stress_lat", "stress_occ"]
    
    def get_mmbert_embedding(text):
        if pd.isna(text) or text.strip() == "":
            return [0.0] * model.config.hidden_size
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
    
    bert_df = pd.DataFrame()
    for col in tqdm(char_cols + word_cols):
        bert_df[col] = df[col].apply(get_mmbert_embedding)
    
    scaler = StandardScaler()
    scaled_nums = scaler.fit_transform(df[num_cols])
    scaled_nums_df = pd.DataFrame(scaled_nums, columns=num_cols)
    
    f_df = pd.concat([bert_df, scaled_nums_df.reset_index(drop=True)], axis=1)
    f_df.to_pickle("x4.pkl")
    print(f"Saved x4.pkl with shape: {f_df.shape}")
    
    # Construct final feature matrix
    final_df = pd.DataFrame()
    for col in (char_cols + word_cols):
        final_df[col] = f_df[col].apply(lambda v: np.mean(v) if isinstance(v, (list, np.ndarray)) else v)
    
    for col in num_cols:
        final_df[col] = f_df[col].apply(lambda v: v[0] if isinstance(v, (list, np.ndarray)) and len(v) == 1 else v)
    
    final_df.to_pickle("x_mmbert_1.pkl")
    print(f"Saved x_mmbert_1.pkl with shape: {final_df.shape}")
    
    # Feature matrix for BiLSTM
    new_cols = [col for col in f_df.columns if col not in num_cols]
    X_array = np.stack([np.stack(f_df[col].values) for col in new_cols], axis=1)
    X_array = X_array.astype(np.float32)
    with open("x_mmbert_2.pkl", "wb") as f:
        pickle.dump(X_array, f)
    print(f"Saved x_mmbert_2.pkl with shape: {X_array.shape}")
    
    return f_df


def main(df_path="new_data_clean.csv"):
    """
    Main execution function for MMBert feature engineering.
    
    Args:
        df_path: Path to cleaned CSV file
    """
    print("=" * 70)
    print("PHASE 2: MMBERT FEATURE ENGINEERING")
    print("=" * 70)
    
    # Load data (assumes Phase 2 features already extracted)
    df = load_data(df_path)
    
    # Generate mmBERT embeddings
    generate_mmbert_embeddings(df)
    
    print("\nMMBert feature engineering complete!")


if __name__ == "__main__":
    main()

