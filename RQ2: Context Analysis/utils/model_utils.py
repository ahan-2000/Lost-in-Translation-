"""
Model loading utilities
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
from transformers import AutoTokenizer
from models import ContextReaderModel


def load_best_exp2_model(model_path: str = None,
                        pretrained: str = "bert-base-multilingual-cased",
                        hidden: int = 256,
                        dropout: float = 0.2,
                        heads: int = 8,
                        dk: int = 128,
                        dv: int = 128,
                        rel_window: int = 64,
                        freeze_encoder: bool = True) -> tuple:
    """
    Load the best Exp2 model weights
    
    Args:
        model_path: Path to model weights (if None, uses default)
        pretrained: Pretrained model name
        hidden: Hidden size
        dropout: Dropout rate
        heads: Number of attention heads
        dk: Key dimension
        dv: Value dimension
        rel_window: Relative window size
        freeze_encoder: Whether to freeze encoder
        
    Returns:
        Tuple of (model, tokenizer, device)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if model_path is None:
        # Default model path - can be overridden
        model_path = "/root/occ/outputs/exp2_only_20251010_123308/Exp2_ContextReader_fold_3.pth"
    
    model = ContextReaderModel(
        pretrained=pretrained,
        hidden=hidden,
        dropout=dropout,
        heads=heads,
        dk=dk,
        dv=dv,
        rel_window=rel_window,
        freeze_encoder=freeze_encoder
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    
    print("✅ Best Exp2 model loaded successfully!")
    return model, tokenizer, device

