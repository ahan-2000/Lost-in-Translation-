"""Text encoders for various backends: TF-IDF, FastText, and HuggingFace models."""

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import FastText as GFastText
from transformers import AutoTokenizer, AutoModel, T5EncoderModel


class TFIDFEncoder:
    """TF-IDF character n-gram encoder."""
    
    def __init__(self, ngram_min=3, ngram_max=5, min_df=2):
        """
        Args:
            ngram_min: Minimum n-gram size
            ngram_max: Maximum n-gram size
            min_df: Minimum document frequency
        """
        self.vec = TfidfVectorizer(
            analyzer='char',
            ngram_range=(ngram_min, ngram_max),
            min_df=min_df
        )
    
    def fit(self, texts):
        """Fit the vectorizer on texts."""
        self.vec.fit(texts)
        return self
    
    def encode(self, texts):
        """Encode texts to TF-IDF vectors."""
        return self.vec.transform(texts)


class FastTextEncoder:
    """FastText word embedding encoder using gensim."""
    
    def __init__(self, vector_size=200, window=3, min_count=1, epochs=10, sg=1):
        """
        Args:
            vector_size: Dimension of word vectors
            window: Context window size
            min_count: Minimum word count
            epochs: Number of training epochs
            sg: Skip-gram (1) or CBOW (0)
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.sg = sg
        self.model = None
    
    def fit(self, texts):
        """Train FastText model on texts."""
        corpus = [[t] for t in texts]
        self.model = GFastText(
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sg=self.sg
        )
        self.model.build_vocab(corpus_iterable=corpus)
        self.model.train(
            corpus_iterable=corpus,
            total_examples=len(corpus),
            epochs=self.epochs
        )
        return self
    
    def encode(self, texts):
        """Encode texts to FastText vectors."""
        return np.vstack([
            self.model.wv.get_vector(t) for t in texts
        ]).astype(np.float32)


def hf_encode(model_name, texts, pooling="mean", max_length=32, batch_size=128, device=None):
    """
    Encode texts using HuggingFace transformer models.
    
    Args:
        model_name: HuggingFace model identifier
        texts: List of text strings
        pooling: Pooling strategy ('mean' or 'cls')
        max_length: Maximum sequence length
        batch_size: Batch size for encoding
        device: Device to use ('cuda' or 'cpu'), auto-detected if None
    
    Returns:
        numpy array of shape (n_texts, hidden_dim)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tok = AutoTokenizer.from_pretrained(model_name)
    
    if "t5" in model_name.lower():
        model = T5EncoderModel.from_pretrained(model_name).to(device).eval()
    else:
        model = AutoModel.from_pretrained(model_name).to(device).eval()
    
    outs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        toks = tok(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            out = model(**toks)
            hidden = out.last_hidden_state
            
            if pooling == "cls" and not hasattr(model, "pooler"):
                pooling = "mean"
            
            if pooling == "cls":
                pooled = hidden[:, 0]
            else:
                mask = toks['attention_mask'].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        
        outs.append(pooled.detach().cpu().numpy())
    
    return np.vstack(outs).astype(np.float32)

