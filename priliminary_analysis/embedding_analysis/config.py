"""Configuration file for experiments."""

import os

# Data paths
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(DATA_DIR, "..", "latin_occitan_with_variations.csv")
SPLIT_PATH = os.path.join(DATA_DIR, "..", "latin_occitan_group_split.csv")
PAIRS_PATH = os.path.join(DATA_DIR, "..", "latin_occitan_pairs_sample.csv")

# Output directories
RESULTS_DIR = os.path.join(DATA_DIR, "results")
GENDER_RESULTS_DIR = os.path.join(RESULTS_DIR, "gender_results")
PAIRWISE_RESULTS_DIR = os.path.join(RESULTS_DIR, "pairwise_results")
CLUSTERING_RESULTS_DIR = os.path.join(RESULTS_DIR, "clustering_results")

# Text normalization
LOWERCASE = False
STRIP_ACCENTS = False

# Encoder settings
TFIDF_NGRAM_MIN = 3
TFIDF_NGRAM_MAX = 5
TFIDF_MIN_DF = 2

FASTTEXT_VECTOR_SIZE = 200
FASTTEXT_WINDOW = 3
FASTTEXT_MIN_COUNT = 1
FASTTEXT_EPOCHS = 10
FASTTEXT_SG = 1  # Skip-gram

HF_POOLING = "mean"
HF_MAX_LENGTH = 32
HF_BATCH_SIZE = 128

# Clustering settings
CLUSTERING_SUBSET_SIZE = 8000
CLUSTERING_MAX_K = 2000
CLUSTERING_SEED = 0

# Random seeds
RANDOM_SEED = 42

