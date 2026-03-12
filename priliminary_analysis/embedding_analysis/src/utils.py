"""Utility functions for data processing and normalization."""

from unidecode import unidecode


def normalize_text(text, lowercase=False, strip_accents=False):
    """
    Normalize text by optionally lowercasing and/or removing accents.
    
    Args:
        text: Input text string
        lowercase: Whether to lowercase the text
        strip_accents: Whether to remove accents using unidecode
    
    Returns:
        Normalized text string
    """
    result = text.lower() if lowercase else text
    result = unidecode(result) if strip_accents else result
    return result


def normalize_texts(texts, lowercase=False, strip_accents=False):
    """
    Normalize a list of texts.
    
    Args:
        texts: List of text strings
        lowercase: Whether to lowercase the texts
        strip_accents: Whether to remove accents
    
    Returns:
        List of normalized text strings
    """
    return [normalize_text(t, lowercase, strip_accents) for t in texts]

