"""
PoS Tag Importance Analysis
Analyzes token importance aggregated by Part-of-Speech tags
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from word_context_analysis.pos_analysis.pos_tagger import get_pos_tagger, PoSTagger


class PoSImportanceAnalyzer:
    """Analyze token importance aggregated by PoS tags"""
    
    def __init__(self, tagger: Optional[PoSTagger] = None, method: str = "manual"):
        """
        Initialize PoS importance analyzer
        
        Args:
            tagger: PoS tagger instance (if None, will create one)
            method: Tagging method if tagger is None
        """
        if tagger is None:
            self.tagger = get_pos_tagger(method=method)
        else:
            self.tagger = tagger
    
    def aggregate_by_pos(self, 
                        tokens: List[str],
                        importance_scores: List[float],
                        sentence: str,
                        pos_tags: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Aggregate token importance scores by PoS tags
        
        Args:
            tokens: List of token strings
            importance_scores: List of importance scores (one per token)
            sentence: Original sentence text
            pos_tags: Pre-computed PoS tags (if None, will compute)
            
        Returns:
            Dictionary mapping PoS tags to aggregated statistics
        """
        if pos_tags is None:
            pos_tags = self.tagger.tag(tokens, sentence)
        
        # Group tokens by PoS tag
        pos_importance = defaultdict(list)
        pos_tokens = defaultdict(list)
        
        for token, score, tag in zip(tokens, importance_scores, pos_tags):
            pos_importance[tag].append(score)
            pos_tokens[tag].append(token)
        
        # Calculate statistics per PoS tag
        pos_stats = {}
        for tag, scores in pos_importance.items():
            scores_array = np.array(scores)
            pos_stats[tag] = {
                'mean': float(np.mean(scores_array)),
                'std': float(np.std(scores_array)),
                'median': float(np.median(scores_array)),
                'min': float(np.min(scores_array)),
                'max': float(np.max(scores_array)),
                'count': len(scores),
                'tokens': pos_tokens[tag]
            }
        
        return pos_stats
    
    def aggregate_across_samples(self, 
                                sample_results: List[Dict]) -> Dict[str, Dict]:
        """
        Aggregate PoS importance across multiple samples
        
        Args:
            sample_results: List of dictionaries, each containing:
                - 'pos_stats': Dictionary from aggregate_by_pos
                - Other sample metadata (optional)
                
        Returns:
            Dictionary mapping PoS tags to aggregated statistics across all samples
        """
        # Collect all scores per PoS tag
        pos_all_scores = defaultdict(list)
        pos_counts = defaultdict(int)
        
        for sample in sample_results:
            pos_stats = sample.get('pos_stats', {})
            for tag, stats in pos_stats.items():
                # Reconstruct scores from stats (approximate)
                mean = stats['mean']
                std = stats.get('std', 0)
                count = stats['count']
                
                # For aggregation, we'll use the mean scores
                # In practice, you might want to store raw scores
                pos_all_scores[tag].extend([mean] * count)
                pos_counts[tag] += count
        
        # Calculate final statistics
        final_stats = {}
        for tag, scores in pos_all_scores.items():
            scores_array = np.array(scores)
            final_stats[tag] = {
                'mean': float(np.mean(scores_array)),
                'std': float(np.std(scores_array)),
                'median': float(np.median(scores_array)),
                'min': float(np.min(scores_array)),
                'max': float(np.max(scores_array)),
                'count': len(scores),
                'total_samples': pos_counts[tag]
            }
        
        return final_stats
    
    def analyze_occlusion_by_pos(self,
                                model,
                                tokenizer,
                                sample: Dict,
                                tokens: List[str],
                                sentence: str,
                                pos_tags: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Analyze importance by occluding tokens grouped by PoS tags
        
        Args:
            model: Trained model
            tokenizer: Tokenizer
            sample: Sample dictionary with model inputs
            tokens: List of token strings
            sentence: Original sentence text
            pos_tags: Pre-computed PoS tags (if None, will compute)
            
        Returns:
            Dictionary mapping PoS tags to occlusion impact scores
        """
        import torch
        
        if pos_tags is None:
            pos_tags = self.tagger.tag(tokens, sentence)
        
        # Get baseline prediction
        with torch.no_grad():
            baseline_output = model(**sample)
            baseline_logits = baseline_output['logits']
            baseline_prob = torch.softmax(baseline_logits, dim=-1)
            baseline_conf = baseline_prob[0, baseline_logits.argmax()].item()
        
        # Group tokens by PoS tag
        pos_token_indices = defaultdict(list)
        for idx, tag in enumerate(pos_tags):
            pos_token_indices[tag].append(idx)
        
        # Occlude each PoS category
        pos_impacts = {}
        for tag, indices in pos_token_indices.items():
            # Create masked sample
            masked_sample = sample.copy()
            input_ids = sample['input_ids_sent'].clone()
            
            # Mask tokens in this PoS category
            for idx in indices:
                if 0 <= idx < input_ids.shape[1]:
                    input_ids[0, idx] = tokenizer.mask_token_id
            
            masked_sample['input_ids_sent'] = input_ids
            
            # Get masked prediction
            with torch.no_grad():
                masked_output = model(**masked_sample)
                masked_logits = masked_output['logits']
                masked_prob = torch.softmax(masked_logits, dim=-1)
                masked_conf = masked_prob[0, masked_logits.argmax()].item()
            
            # Calculate impact
            impact = baseline_conf - masked_conf
            pos_impacts[tag] = impact
        
        return pos_impacts

