"""
Token processing utilities
"""

from typing import List


def clean_tokens(tokens: List[str]) -> List[str]:
    """
    Remove special tokens and clean token names for better readability
    
    Args:
        tokens: List of token strings
        
    Returns:
        List of cleaned tokens
    """
    cleaned_tokens = []
    for token in tokens:
        if token in ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]']:
            continue
        if token.startswith('##'):
            token = token[2:]
        if len(token.strip()) < 2:
            continue
        cleaned_tokens.append(token)
    return cleaned_tokens

