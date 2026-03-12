"""
Part-of-Speech (PoS) Tagging Module
Provides PoS tagging functionality using various methods (Phi-4, manual, etc.)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from typing import List, Dict, Tuple, Optional
import torch
from transformers import AutoTokenizer
try:
    from transformers import AutoModelForCausalLM
    PHI4_AVAILABLE = True
except ImportError:
    PHI4_AVAILABLE = False


class PoSTagger:
    """Base class for PoS tagging"""
    
    def tag(self, tokens: List[str], sentence: str) -> List[str]:
        """
        Tag tokens with PoS tags
        
        Args:
            tokens: List of token strings
            sentence: Original sentence text
            
        Returns:
            List of PoS tags (one per token)
        """
        raise NotImplementedError


class ManualPoSTagger(PoSTagger):
    """Manual rule-based PoS tagger for Occitan"""
    
    # Common Occitan determiners/articles
    DETERMINERS = {'la', 'lo', 'le', 'les', 'un', 'una', 'uno', 'li', 'las', 'los'}
    
    # Common Occitan pronouns
    PRONOUNS = {'el', 'ela', 'els', 'elas', 'me', 'te', 'se', 'nos', 'vos', 'aquill', 'aquella'}
    
    # Common Occitan prepositions
    PREPOSITIONS = {'de', 'en', 'a', 'per', 'amb', 'sos', 'sus', 'entre', 'contra', 'vers'}
    
    # Common Occitan conjunctions
    CONJUNCTIONS = {'e', 'o', 'mas', 'pero', 'que', 'com', 'si', 'quan'}
    
    def tag(self, tokens: List[str], sentence: str) -> List[str]:
        """Tag tokens using manual rules"""
        pos_tags = []
        
        for token in tokens:
            # Clean token
            clean_token = token.lower().strip()
            if token.startswith('##'):
                clean_token = token[2:].lower().strip()
            
            # Skip special tokens
            if token in ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]']:
                pos_tags.append('X')
                continue
            
            # Rule-based tagging
            if clean_token in self.DETERMINERS:
                pos_tags.append('DET')
            elif clean_token in self.PRONOUNS:
                pos_tags.append('PRON')
            elif clean_token in self.PREPOSITIONS:
                pos_tags.append('ADP')
            elif clean_token in self.CONJUNCTIONS:
                if clean_token in {'e', 'o', 'mas', 'pero'}:
                    pos_tags.append('CCONJ')
                else:
                    pos_tags.append('SCONJ')
            else:
                # Default to NOUN (can be refined)
                pos_tags.append('NOUN')
        
        return pos_tags


class Phi4PoSTagger(PoSTagger):
    """PoS tagger using Microsoft Phi-4 model via Hugging Face"""
    
    def __init__(self, model_name: str = "microsoft/Phi-4-mini-instruct"):
        """
        Initialize Phi-4 PoS tagger
        
        Args:
            model_name: Hugging Face model identifier
        """
        if not PHI4_AVAILABLE:
            raise ImportError("transformers library not available or doesn't support Phi-4")
        
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load Phi-4 model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load Phi-4 model: {e}")
    
    def tag(self, tokens: List[str], sentence: str) -> List[str]:
        """
        Tag tokens using Phi-4 model
        
        Args:
            tokens: List of token strings
            sentence: Original sentence text
            
        Returns:
            List of PoS tags (Universal Dependencies format)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Create prompt for Phi-4
        prompt = f"""Analyze the provided text and assign to each word Universal Dependencies Part-of-Speech tags.

Text: {sentence}

Words: {', '.join(tokens)}

Provide PoS tags in the format: word1:TAG1, word2:TAG2, ..."""
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse response to extract PoS tags
        pos_tags = self._parse_response(response, tokens)
        
        return pos_tags
    
    def _parse_response(self, response: str, tokens: List[str]) -> List[str]:
        """Parse Phi-4 response to extract PoS tags"""
        # Simple parsing - can be improved
        pos_tags = []
        
        # Try to extract tags from response
        # Format: word1:TAG1, word2:TAG2, ...
        tag_map = {}
        parts = response.split(',')
        for part in parts:
            if ':' in part:
                word_tag = part.strip().split(':')
                if len(word_tag) == 2:
                    word = word_tag[0].strip()
                    tag = word_tag[1].strip().upper()
                    tag_map[word.lower()] = tag
        
        # Map tokens to tags
        for token in tokens:
            clean_token = token.lower().strip()
            if token.startswith('##'):
                clean_token = token[2:].lower().strip()
            
            if clean_token in tag_map:
                pos_tags.append(tag_map[clean_token])
            else:
                pos_tags.append('X')  # Unknown
        
        return pos_tags


def get_pos_tagger(method: str = "manual", **kwargs) -> PoSTagger:
    """
    Factory function to get a PoS tagger
    
    Args:
        method: Tagging method ('manual' or 'phi4')
        **kwargs: Additional arguments for specific taggers
        
    Returns:
        PoSTagger instance
    """
    if method == "manual":
        return ManualPoSTagger()
    elif method == "phi4":
        return Phi4PoSTagger(**kwargs)
    else:
        raise ValueError(f"Unknown PoS tagging method: {method}")

