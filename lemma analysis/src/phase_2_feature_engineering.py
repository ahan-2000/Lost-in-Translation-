"""
Phase 2: Feature Engineering
Refactored from Phase_2.ipynb
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import pickle
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from gensim.models import FastText
from transformers import AutoTokenizer, AutoModel
import torch


def load_data(df_path="new_data_clean.csv"):
    """Load cleaned dataset."""
    df = pd.read_csv(df_path)
    print(f"Loaded {len(df)} samples")
    return df


def extract_ngrams(df):
    """Extract n-grams from Latin and Occitan words."""
    print("\nExtracting n-grams...")
    for n in range(1, 5):
        df[f'lat_ngram_{n}'] = df['Lemma_std'].str[-n:]
        df[f'occ_ngram_{n}'] = df['Akk_Sing_std'].str[-n:]
    return df


def count_syllables(df):
    """Count syllables in Latin and Occitan words."""
    vowel_pattern = re.compile(r"[aeiou]")
    
    def count_syl(word):
        if pd.isna(word):
            return 0
        return len(vowel_pattern.findall(word.lower()))
    
    df['syl_lat'] = df['Lemma_std'].apply(count_syl)
    df['syl_occ'] = df['Akk_Sing_std'].apply(count_syl)
    return df


def estimate_stress_position(df):
    """Estimate stress position in words."""
    def stress_pos(word):
        if pd.isna(word) or len(word) < 2:
            return 'unknown'
        if word[-1] in 'aeiou':
            return 'penultimate'
        else:
            return 'ultimate'
    
    df['stress_lat'] = df['Lemma_std'].apply(stress_pos)
    df['stress_occ'] = df['Akk_Sing_std'].apply(stress_pos)
    return df


def extract_vc_patterns(df):
    """Extract vowel-consonant patterns."""
    def vc_pattern(word):
        if pd.isna(word):
            return ''
        pattern = ''
        for ch in word.lower():
            if ch in 'aeiou':
                pattern += 'V'
            elif ch.isalpha():
                pattern += 'C'
            else:
                pattern += ch
        return pattern
    
    df['vc_lat'] = df['Lemma_std'].apply(vc_pattern)
    df['vc_occ'] = df['Akk_Sing_std'].apply(vc_pattern)
    return df


def compute_word_length_and_frequency(df):
    """Compute word length and frequency."""
    df['word_len'] = df['Lemma_std'].apply(lambda x: len(str(x)))
    lemma_counts = Counter(df['Lemma_std'])
    df['frequency'] = df['Lemma_std'].apply(lambda x: lemma_counts[x])
    return df


def generate_fasttext_embeddings(df):
    """Generate FastText embeddings for features."""
    print("\nGenerating FastText embeddings...")
    
    char_cols = [
        "lat_ngram_1", "lat_ngram_2", "lat_ngram_3", "lat_ngram_4",
        "occ_ngram_1", "occ_ngram_2", "occ_ngram_3", "occ_ngram_4",
        "vc_lat", "vc_occ"
    ]
    num_cols = ["syl_lat", "syl_occ", "word_len", "frequency"]
    word_cols = ["stress_lat", "stress_occ"]
    
    df[char_cols + word_cols] = df[char_cols + word_cols].fillna("")
    
    # Train FastText on character sequences
    char_sentences = []
    for col in char_cols:
        for val in df[col]:
            char_sentences.append(list(str(val)))
    
    char_model = FastText(vector_size=50, window=3, min_count=1, min_n=1, max_n=1)
    char_model.build_vocab(corpus_iterable=char_sentences)
    char_model.train(corpus_iterable=char_sentences, total_examples=len(char_sentences), epochs=10)
    
    char_embeds = {}
    for col in char_cols:
        char_embeds[col] = [
            np.mean([char_model.wv[ch] for ch in list(str(val))], axis=0)
            for val in df[col]
        ]
    
    # Scale numeric features
    scaler = StandardScaler()
    num_scaled = scaler.fit_transform(df[num_cols])
    num_embeds = {}
    for i, col in enumerate(num_cols):
        num_embeds[col] = [np.array([val]) for val in num_scaled[:, i]]
    
    # Train FastText on word sequences
    word_sentences = []
    for col in word_cols:
        for val in df[col]:
            word_sentences.append(str(val).split())
    
    word_model = FastText(vector_size=50, window=3, min_count=1, min_n=2, max_n=4)
    word_model.build_vocab(corpus_iterable=word_sentences)
    word_model.train(corpus_iterable=word_sentences, total_examples=len(word_sentences), epochs=10)
    
    word_embeds = {}
    for col in word_cols:
        word_embeds[col] = [
            np.mean([word_model.wv[w] for w in str(val).split()], axis=0)
            for val in df[col]
        ]
    
    # Combine into DataFrame
    df_1 = pd.DataFrame()
    for col in char_cols:
        df_1[col] = char_embeds[col]
    for col in num_cols:
        df_1[col] = num_embeds[col]
    for col in word_cols:
        df_1[col] = word_embeds[col]
    
    # Save feature matrix
    vec_cols = [
        'lat_ngram_1', 'lat_ngram_2', 'lat_ngram_3', 'lat_ngram_4',
        'occ_ngram_1', 'occ_ngram_2', 'occ_ngram_3', 'occ_ngram_4',
        'vc_lat', 'vc_occ', 'stress_lat', 'stress_occ'
    ]
    
    for col in vec_cols:
        df_1[col] = df_1[col].apply(lambda v: np.mean(v) if isinstance(v, (list, np.ndarray)) else v)
    
    scaled_num_cols = ['syl_lat', 'syl_occ', 'word_len', 'frequency']
    for col in scaled_num_cols:
        df_1[col] = df_1[col].apply(lambda v: v[0] if isinstance(v, (list, np.ndarray)) and len(v) == 1 else v)
    
    df_1.to_pickle("x_ft_1.pkl")
    print(f"Saved x_ft_1.pkl with shape: {df_1.shape}")
    
    # Save feature matrix for BiLSTM
    scalar_cols = ['syl_lat', 'syl_occ', 'word_len', 'frequency']
    new_cols = [col for col in df_1.columns if col not in scalar_cols]
    X_array = np.stack([np.stack(df_1[col].values) for col in new_cols], axis=1)
    X_array = X_array.astype(np.float32)
    with open("x_ft_2.pkl", "wb") as f:
        pickle.dump(X_array, f)
    print(f"Saved x_ft_2.pkl with shape: {X_array.shape}")
    
    return df_1


def generate_bert_embeddings(df):
    """Generate mBERT embeddings for features."""
    print("\nGenerating mBERT embeddings...")
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    model = AutoModel.from_pretrained("bert-base-multilingual-cased")
    model.eval()
    
    char_cols = [
        "lat_ngram_1", "lat_ngram_2", "lat_ngram_3", "lat_ngram_4",
        "occ_ngram_1", "occ_ngram_2", "occ_ngram_3", "occ_ngram_4",
        "vc_lat", "vc_occ"
    ]
    num_cols = ["syl_lat", "syl_occ", "word_len", "frequency"]
    word_cols = ["stress_lat", "stress_occ"]
    
    def get_mbert_embedding(text):
        if pd.isna(text) or text.strip() == "":
            return [0.0] * 768
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
    
    bert_df = pd.DataFrame()
    for col in tqdm(char_cols + word_cols):
        bert_df[col] = df[col].apply(get_mbert_embedding)
    
    scaler = StandardScaler()
    scaled_nums = scaler.fit_transform(df[num_cols])
    scaled_nums_df = pd.DataFrame(scaled_nums, columns=num_cols)
    
    f_df = pd.concat([bert_df, scaled_nums_df.reset_index(drop=True)], axis=1)
    f_df.to_pickle("x2.pkl")
    print(f"Saved x2.pkl with shape: {f_df.shape}")
    
    # Construct final feature matrix
    final_df = pd.DataFrame()
    vec_cols = [
        'lat_ngram_1', 'lat_ngram_2', 'lat_ngram_3', 'lat_ngram_4',
        'occ_ngram_1', 'occ_ngram_2', 'occ_ngram_3', 'occ_ngram_4',
        'vc_lat', 'vc_occ', 'stress_lat', 'stress_occ'
    ]
    
    for col in vec_cols:
        final_df[col] = f_df[col].apply(lambda v: np.mean(v) if isinstance(v, (list, np.ndarray)) else v)
    
    scaled_num_cols = ['syl_lat', 'syl_occ', 'word_len', 'frequency']
    for col in scaled_num_cols:
        final_df[col] = f_df[col].apply(lambda v: v[0] if isinstance(v, (list, np.ndarray)) and len(v) == 1 else v)
    
    final_df.to_pickle("x_bert_1.pkl")
    print(f"Saved x_bert_1.pkl with shape: {final_df.shape}")
    
    # Feature matrix for BiLSTM
    scalar_cols = ['syl_lat', 'syl_occ', 'word_len', 'frequency']
    new_cols = [col for col in f_df.columns if col not in scalar_cols]
    X_array = np.stack([np.stack(f_df[col].values) for col in new_cols], axis=1)
    X_array = X_array.astype(np.float32)
    with open("x_bert_2.pkl", "wb") as f:
        pickle.dump(X_array, f)
    print(f"Saved x_bert_2.pkl with shape: {X_array.shape}")
    
    return f_df


def generate_byt5_embeddings(df):
    """Generate ByT5 embeddings for features."""
    print("\nGenerating ByT5 embeddings...")
    
    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
    model = AutoModel.from_pretrained("google/byt5-small")
    model.eval()
    
    char_cols = [
        "lat_ngram_1", "lat_ngram_2", "lat_ngram_3", "lat_ngram_4",
        "occ_ngram_1", "occ_ngram_2", "occ_ngram_3", "occ_ngram_4",
        "vc_lat", "vc_occ"
    ]
    num_cols = ["syl_lat", "syl_occ", "word_len", "frequency"]
    word_cols = ["stress_lat", "stress_occ"]
    
    def get_byt5_embedding(text):
        if isinstance(text, (list, tuple)):
            text = " ".join(map(str, text))
        if pd.isna(text) or str(text).strip() == "":
            return [0.0] * model.config.d_model
        inputs = tokenizer(str(text), return_tensors="pt", truncation=True, padding=True, max_length=64)
        with torch.no_grad():
            outputs = model.encoder(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
    
    byt5_df = pd.DataFrame()
    for col in tqdm(char_cols + word_cols):
        byt5_df[col] = df[col].apply(get_byt5_embedding)
    
    scaler = StandardScaler()
    scaled_nums = scaler.fit_transform(df[num_cols])
    scaled_nums_df = pd.DataFrame(scaled_nums, columns=num_cols)
    
    f1_df = pd.concat([byt5_df, scaled_nums_df.reset_index(drop=True)], axis=1)
    f1_df.to_pickle("X_byt5.pkl")
    print(f"Saved X_byt5.pkl with shape: {f1_df.shape}")
    
    # Construct final feature matrix
    vec_cols = [
        'lat_ngram_1', 'lat_ngram_2', 'lat_ngram_3', 'lat_ngram_4',
        'occ_ngram_1', 'occ_ngram_2', 'occ_ngram_3', 'occ_ngram_4',
        'vc_lat', 'vc_occ', 'stress_lat', 'stress_occ'
    ]
    
    for col in vec_cols:
        f1_df[col] = f1_df[col].apply(lambda v: np.mean(v) if isinstance(v, (list, np.ndarray)) else v)
    
    scaled_num_cols = ['syl_lat', 'syl_occ', 'word_len', 'frequency']
    for col in scaled_num_cols:
        f1_df[col] = f1_df[col].apply(lambda v: v[0] if isinstance(v, (list, np.ndarray)) and len(v) == 1 else v)
    
    f1_df.to_pickle("x_byt5_1.pkl")
    print(f"Saved x_byt5_1.pkl with shape: {f1_df.shape}")
    
    # Feature matrix for BiLSTM
    scalar_cols = ['syl_lat', 'syl_occ', 'word_len', 'frequency']
    new_cols = [col for col in f1_df.columns if col not in scalar_cols]
    X_array = np.stack([np.stack(f1_df[col].values) for col in new_cols], axis=1)
    X_array = X_array.astype(np.float32)
    with open("x_byt5_2.pkl", "wb") as f:
        pickle.dump(X_array, f)
    print(f"Saved x_byt5_2.pkl with shape: {X_array.shape}")
    
    return f1_df


def save_targets_and_ids(df):
    """Save target labels and lexeme IDs."""
    y = df['Genus_ok'].copy()
    with open("y.pkl", "wb") as f:
        pickle.dump(y, f)
    print(f"\ny.pkl saved with target class counts:\n{y.value_counts()}")
    
    lexeme_ids = df['Lexeme_ID'].copy()
    with open("lexeme_ids.pkl", "wb") as f:
        pickle.dump(lexeme_ids, f)
    print(f"lexeme_ids.pkl saved with total lexemes: {len(lexeme_ids)}")
    
    # Save feature names
    feature_names = [
        "lat_ngram_1", "lat_ngram_2", "lat_ngram_3", "lat_ngram_4",
        "occ_ngram_1", "occ_ngram_2", "occ_ngram_3", "occ_ngram_4",
        "vc_lat", "vc_occ",
        "stress_lat", "stress_occ"
    ]
    with open("feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)
    print("Saved feature_names.pkl")


def main(df_path="new_data_clean.csv"):
    """
    Main execution function for Phase 2.
    
    Args:
        df_path: Path to cleaned CSV file
    """
    print("=" * 70)
    print("PHASE 2: FEATURE ENGINEERING")
    print("=" * 70)
    
    # Load data
    df = load_data(df_path)
    
    # Extract features
    df = extract_ngrams(df)
    df = count_syllables(df)
    df = estimate_stress_position(df)
    df = extract_vc_patterns(df)
    df = compute_word_length_and_frequency(df)
    
    # Generate embeddings
    generate_fasttext_embeddings(df)
    generate_bert_embeddings(df)
    generate_byt5_embeddings(df)
    
    # Save targets and IDs
    save_targets_and_ids(df)
    
    print("\nPhase 2 complete!")


if __name__ == "__main__":
    main()

