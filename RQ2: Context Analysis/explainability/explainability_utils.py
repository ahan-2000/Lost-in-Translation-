"""
Explainability utilities for Exp2 ContextReaderModel
Provides various explanation methods for gender prediction in Occitan sentences
"""

import sys
import os
# Add parent directories to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import pandas as pd
from transformers import AutoTokenizer
from captum.attr import IntegratedGradients, Saliency, GradientShap, InputXGradient
from captum.attr import visualization as viz
import warnings
warnings.filterwarnings('ignore')

class Exp2Explainer:
    """Main explainer class for ContextReaderModel"""
    
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        
    def prepare_sample(self, sentence: str, noun_pos: int, occitan_word: str, 
                      latin_lemma: str, latin_gender: str) -> Dict:
        """Prepare a single sample for explanation"""
        
        # Tokenize sentence
        sent_enc = self.tokenizer(
            sentence, 
            max_length=128,
            truncation=True, 
            padding="max_length",
            return_tensors="pt"
        )
        
        # Handle invalid noun positions
        if noun_pos < 0:
            # If noun position is invalid, use position 1 (after [CLS])
            noun_pos = 1
            print(f"Warning: Invalid noun position, using position {noun_pos}")
        
        # Ensure noun_pos is within valid range
        seq_len = sent_enc["input_ids"].shape[1]
        noun_pos = min(max(noun_pos, 0), seq_len - 1)
        
        # Build fields text
        lg = str(latin_gender).strip().upper()
        if lg in ["F","FEM","FEMININE"]: lg_tok = "LATGENDER_F"
        elif lg in ["M","MASC","MASCULINE"]: lg_tok = "LATGENDER_M"
        elif lg in ["N","NEUT","NEUTER"]: lg_tok = "LATGENDER_N"
        else: lg_tok = f"LATGENDER_{lg}"
        
        fields_text = f"[OCC]{occitan_word} [LAT]{latin_lemma} [LG]{lg_tok}"
        fields_enc = self.tokenizer(
            fields_text,
            max_length=32,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids_sent": sent_enc["input_ids"].to(self.device),
            "attention_mask_sent": sent_enc["attention_mask"].to(self.device),
            "noun_pos": torch.tensor([noun_pos]).to(self.device),
            "input_ids_fields": fields_enc["input_ids"].to(self.device),
            "attention_mask_fields": fields_enc["attention_mask"].to(self.device),
            "tokens": self.tokenizer.convert_ids_to_tokens(sent_enc["input_ids"][0])
        }
    
    def get_saliency_maps(self, sample: Dict, target_class: int = None) -> Dict:
        """Compute saliency maps (∂logit/∂embedding)"""
        
        input_ids = sample["input_ids_sent"]
        attention_mask = sample["attention_mask_sent"]
        noun_pos = sample["noun_pos"]
        input_ids_fields = sample["input_ids_fields"]
        attention_mask_fields = sample["attention_mask_fields"]
        
        # Get embeddings
        embeddings = self.model.sent_enc.embeddings.word_embeddings(input_ids)
        embeddings.requires_grad_(True)
        
        # Forward pass with embeddings
        out_s = self.model.sent_enc(inputs_embeds=embeddings, attention_mask=attention_mask)
        H_top = out_s.last_hidden_state
        
        # Get noun representation with bounds checking
        B, T, d = H_top.shape
        noun_pos_clamped = torch.clamp(noun_pos, 0, T-1)  # Ensure valid range
        idx = noun_pos_clamped.view(B,1,1).expand(-1,1,d)
        h_i = H_top.gather(dim=1, index=idx).squeeze(1)
        
        # Cross-attention
        Q = torch.einsum("bd,hdk->bhk", h_i, self.model.WQ)
        K = torch.einsum("btd,hdk->bthk", H_top, self.model.WK)
        V = torch.einsum("btd,hdv->bthv", H_top, self.model.WV)
        scores = torch.einsum("bhk,bthk->bht", Q, K) / (self.model.dk**0.5)
        
        # Apply masks and softmax
        am = attention_mask.unsqueeze(1).expand(-1, self.model.heads, -1)
        scores = scores.masked_fill(am==0, -1e9)
        alpha = scores.softmax(dim=-1)
        
        # Get context
        c = torch.einsum("bht,bthv->bhv", alpha, V).reshape(B, self.model.heads*self.model.dv)
        
        # Fields processing
        out_f = self.model.fields_enc(input_ids=input_ids_fields, attention_mask=attention_mask_fields)
        pooled_f = self._mean_pool(out_f.last_hidden_state, attention_mask_fields)
        
        # Final prediction
        r = torch.cat([h_i, c, pooled_f], dim=-1)
        logits = self.model.mlp(self.model.proj(r))
        
        if target_class is None:
            target_class = logits.argmax().item()
        
        # Compute gradients
        logit = logits[0, target_class]
        grad = torch.autograd.grad(logit, embeddings, retain_graph=True)[0]
        
        return {
            "saliency": grad[0].detach().cpu().numpy(),
            "grad_x_input": (grad * embeddings)[0].detach().cpu().numpy(),
            "target_class": target_class,
            "logits": logits[0].detach().cpu().numpy()
        }
    
    def get_smooth_grad(self, sample: Dict, target_class: int = None, 
                       n_samples: int = 50, noise_level: float = 0.1) -> Dict:
        """Compute SmoothGrad attribution"""
        
        saliencies = []
        
        for _ in range(n_samples):
            # Add noise to embeddings
            input_ids = sample["input_ids_sent"]
            embeddings = self.model.sent_enc.embeddings.word_embeddings(input_ids)
            noise = torch.randn_like(embeddings) * noise_level
            noisy_embeddings = embeddings + noise
            noisy_embeddings.requires_grad_(True)
            
            # Forward pass
            out_s = self.model.sent_enc(inputs_embeds=noisy_embeddings, 
                                      attention_mask=sample["attention_mask_sent"])
            H_top = out_s.last_hidden_state
            
            # Get noun representation and attention
            B, T, d = H_top.shape
            idx = sample["noun_pos"].view(B,1,1).expand(-1,1,d)
            h_i = H_top.gather(dim=1, index=idx).squeeze(1)
            
            Q = torch.einsum("bd,hdk->bhk", h_i, self.model.WQ)
            K = torch.einsum("btd,hdk->bthk", H_top, self.model.WK)
            V = torch.einsum("btd,hdv->bthv", H_top, self.model.WV)
            scores = torch.einsum("bhk,bthk->bht", Q, K) / (self.model.dk**0.5)
            
            am = sample["attention_mask_sent"].unsqueeze(1).expand(-1, self.model.heads, -1)
            scores = scores.masked_fill(am==0, -1e9)
            alpha = scores.softmax(dim=-1)
            c = torch.einsum("bht,bthv->bhv", alpha, V).reshape(B, self.model.heads*self.model.dv)
            
            # Fields
            out_f = self.model.fields_enc(input_ids=sample["input_ids_fields"], 
                                        attention_mask=sample["attention_mask_fields"])
            pooled_f = self._mean_pool(out_f.last_hidden_state, sample["attention_mask_fields"])
            
            # Prediction
            r = torch.cat([h_i, c, pooled_f], dim=-1)
            logits = self.model.mlp(self.model.proj(r))
            
            if target_class is None:
                target_class = logits.argmax().item()
            
            logit = logits[0, target_class]
            grad = torch.autograd.grad(logit, noisy_embeddings, retain_graph=True)[0]
            saliencies.append(grad[0].detach().cpu().numpy())
        
        smooth_grad = np.mean(saliencies, axis=0)
        var_grad = np.var(saliencies, axis=0)
        
        return {
            "smooth_grad": smooth_grad,
            "var_grad": var_grad,
            "target_class": target_class
        }
    
    def get_integrated_gradients(self, sample: Dict, target_class: int = None, 
                                steps: int = 50) -> Dict:
        """Compute Integrated Gradients"""
        
        def forward_func(embeddings):
            out_s = self.model.sent_enc(inputs_embeds=embeddings, 
                                      attention_mask=sample["attention_mask_sent"])
            H_top = out_s.last_hidden_state
            
            B, T, d = H_top.shape
            idx = sample["noun_pos"].view(B,1,1).expand(-1,1,d)
            h_i = H_top.gather(dim=1, index=idx).squeeze(1)
            
            Q = torch.einsum("bd,hdk->bhk", h_i, self.model.WQ)
            K = torch.einsum("btd,hdk->bthk", H_top, self.model.WK)
            V = torch.einsum("btd,hdv->bthv", H_top, self.model.WV)
            scores = torch.einsum("bhk,bthk->bht", Q, K) / (self.model.dk**0.5)
            
            am = sample["attention_mask_sent"].unsqueeze(1).expand(-1, self.model.heads, -1)
            scores = scores.masked_fill(am==0, -1e9)
            alpha = scores.softmax(dim=-1)
            c = torch.einsum("bht,bthv->bhv", alpha, V).reshape(B, self.model.heads*self.model.dv)
            
            out_f = self.model.fields_enc(input_ids=sample["input_ids_fields"], 
                                        attention_mask=sample["attention_mask_fields"])
            pooled_f = self._mean_pool(out_f.last_hidden_state, sample["attention_mask_fields"])
            
            r = torch.cat([h_i, c, pooled_f], dim=-1)
            logits = self.model.mlp(self.model.proj(r))
            
            return logits
        
        # Get baseline (zero embeddings)
        baseline = torch.zeros_like(self.model.sent_enc.embeddings.word_embeddings(sample["input_ids_sent"]))
        
        # Get current embeddings
        current = self.model.sent_enc.embeddings.word_embeddings(sample["input_ids_sent"])
        
        # Compute IG manually
        attributions = torch.zeros_like(current)
        
        for i in range(steps):
            alpha = i / (steps - 1)
            interpolated = baseline + alpha * (current - baseline)
            interpolated.requires_grad_(True)
            
            logits = forward_func(interpolated)
            if target_class is None:
                target_class = logits.argmax().item()
            
            logit = logits[0, target_class]
            grad = torch.autograd.grad(logit, interpolated, retain_graph=True)[0]
            attributions += grad / steps
        
        attributions *= (current - baseline)
        
        return {
            "integrated_gradients": attributions[0].detach().cpu().numpy(),
            "target_class": target_class
        }
    
    def get_attention_analysis(self, sample: Dict) -> Dict:
        """Get comprehensive attention analysis"""
        
        with torch.no_grad():
            output = self.model(**sample, return_attention=True)
            
            # Get attention weights and scores
            attention_weights = output["attention_weights"][0]  # (heads, T)
            attention_scores = output["attention_scores"][0]     # (heads, T)
            
            # Compute attention rollout
            rollout = self._compute_attention_rollout(attention_weights)
            
            # Compute attention flow (weighted by gradients)
            flow = self._compute_attention_flow(sample, attention_weights)
            
            return {
                "raw_attention": attention_weights.cpu().numpy(),
                "attention_scores": attention_scores.cpu().numpy(),
                "attention_rollout": rollout,
                "attention_flow": flow,
                "logits": output["logits"][0].cpu().numpy()
            }
    
    def _compute_attention_rollout(self, attention_weights: torch.Tensor) -> np.ndarray:
        """Compute attention rollout across heads"""
        # Simple rollout: average attention across heads
        rollout = attention_weights.mean(dim=0).cpu().numpy()
        return rollout
    
    def _compute_attention_flow(self, sample: Dict, attention_weights: torch.Tensor) -> np.ndarray:
        """Compute attention flow weighted by gradients"""
        
        # Simplified attention flow: use attention magnitude as proxy
        # Since computing gradients through attention is complex, we'll use
        # the attention weights directly as a measure of flow
        flow = attention_weights.sum(dim=0).cpu().numpy()
        
        return flow
    
    def _mean_pool(self, H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean pooling utility"""
        return (H*mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0).unsqueeze(-1)
    
    def analyze_head_importance(self, sample: Dict, method: str = "masking") -> Dict:
        """Analyze importance of different attention heads"""
        
        original_output = self.model(**sample)
        original_logits = original_output["logits"][0]
        original_pred = original_logits.argmax().item()
        
        head_importance = []
        
        for head_idx in range(self.model.heads):
            if method == "masking":
                # Mask specific head
                importance = self._mask_head_importance(sample, head_idx, original_logits)
            else:
                # Use attention magnitude
                importance = self._magnitude_head_importance(sample, head_idx)
            
            head_importance.append(importance)
        
        return {
            "head_importance": np.array(head_importance),
            "original_prediction": original_pred,
            "original_logits": original_logits.detach().cpu().numpy()
        }
    
    def _mask_head_importance(self, sample: Dict, head_idx: int, original_logits: torch.Tensor) -> float:
        """Compute importance by masking a specific head"""
        
        # This would require modifying the model to allow head masking
        # For now, return attention magnitude as proxy
        with torch.no_grad():
            output = self.model(**sample, return_attention=True)
            attention_weights = output["attention_weights"][0]
            head_attention = attention_weights[head_idx]
            importance = head_attention.sum().item()
        
        return importance
    
    def _magnitude_head_importance(self, sample: Dict, head_idx: int) -> float:
        """Compute importance based on attention magnitude"""
        
        with torch.no_grad():
            output = self.model(**sample, return_attention=True)
            attention_weights = output["attention_weights"][0]
            head_attention = attention_weights[head_idx]
            importance = head_attention.sum().item()
        
        return importance


def visualize_attributions(attributions: Dict, tokens: List[str], 
                          method_name: str, save_path: str = None):
    """Visualize attribution scores"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{method_name} Attribution Analysis', fontsize=16)
    
    # Plot different attribution methods
    methods = ['saliency', 'grad_x_input', 'smooth_grad', 'integrated_gradients']
    titles = ['Saliency Maps', 'Grad×Input', 'SmoothGrad', 'Integrated Gradients']
    
    for i, (method, title) in enumerate(zip(methods, titles)):
        if method in attributions:
            ax = axes[i//2, i%2]
            
            # Get attribution scores (sum over embedding dimension)
            scores = attributions[method].sum(axis=-1)
            
            # Create bar plot
            valid_tokens = tokens[:len(scores)]
            bars = ax.bar(range(len(scores)), scores)
            
            # Color bars by magnitude
            colors = plt.cm.RdYlBu_r(np.abs(scores) / np.abs(scores).max())
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax.set_title(title)
            ax.set_xlabel('Token Position')
            ax.set_ylabel('Attribution Score')
            ax.set_xticks(range(len(valid_tokens)))
            ax.set_xticklabels(valid_tokens, rotation=45, ha='right')
            
            # Highlight noun position
            if 'noun_pos' in attributions:
                noun_pos = attributions['noun_pos']
                ax.axvline(x=noun_pos, color='red', linestyle='--', alpha=0.7, label='Noun Position')
                ax.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def visualize_attention_heads(attention_data: Dict, tokens: List[str], 
                             save_path: str = None):
    """Visualize attention patterns across heads"""
    
    raw_attention = attention_data["raw_attention"]  # (heads, T)
    n_heads = raw_attention.shape[0]
    
    fig, axes = plt.subplots(2, (n_heads + 1) // 2, figsize=(20, 8))
    if n_heads == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    fig.suptitle('Attention Patterns Across Heads', fontsize=16)
    
    for head_idx in range(n_heads):
        ax = axes[head_idx]
        
        # Create heatmap
        attention_matrix = raw_attention[head_idx].reshape(1, -1)
        im = ax.imshow(attention_matrix, cmap='Blues', aspect='auto')
        
        ax.set_title(f'Head {head_idx}')
        ax.set_xlabel('Token Position')
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_yticks([])
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
    
    # Hide unused subplots
    for i in range(n_heads, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_summary_visualization(sample_data: Dict, save_path: str = None):
    """Create comprehensive summary visualization"""
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Comprehensive Explainability Analysis', fontsize=16)
    
    tokens = sample_data["tokens"]
    
    # 1. Saliency Map
    if "saliency" in sample_data["attributions"]:
        ax = axes[0, 0]
        scores = sample_data["attributions"]["saliency"].sum(axis=-1)
        bars = ax.bar(range(len(scores)), scores, color='skyblue')
        ax.set_title('Saliency Map')
        ax.set_xlabel('Token Position')
        ax.set_ylabel('Attribution Score')
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
    
    # 2. Integrated Gradients
    if "integrated_gradients" in sample_data["attributions"]:
        ax = axes[0, 1]
        scores = sample_data["attributions"]["integrated_gradients"].sum(axis=-1)
        bars = ax.bar(range(len(scores)), scores, color='lightcoral')
        ax.set_title('Integrated Gradients')
        ax.set_xlabel('Token Position')
        ax.set_ylabel('Attribution Score')
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
    
    # 3. Attention Rollout
    if "attention_rollout" in sample_data["attention"]:
        ax = axes[0, 2]
        rollout = sample_data["attention"]["attention_rollout"]
        bars = ax.bar(range(len(rollout)), rollout, color='lightgreen')
        ax.set_title('Attention Rollout')
        ax.set_xlabel('Token Position')
        ax.set_ylabel('Attention Weight')
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
    
    # 4. Head Importance
    if "head_importance" in sample_data["head_analysis"]:
        ax = axes[1, 0]
        importance = sample_data["head_analysis"]["head_importance"]
        bars = ax.bar(range(len(importance)), importance, color='gold')
        ax.set_title('Head Importance')
        ax.set_xlabel('Attention Head')
        ax.set_ylabel('Importance Score')
    
    # 5. Prediction Confidence
    ax = axes[1, 1]
    logits = sample_data["attention"]["logits"]
    probs = torch.softmax(torch.tensor(logits), dim=0).numpy()
    classes = ['Masculine', 'Feminine']
    bars = ax.bar(classes, probs, color=['lightblue', 'pink'])
    ax.set_title('Prediction Confidence')
    ax.set_ylabel('Probability')
    ax.set_ylim(0, 1)
    
    # 6. Token-level Analysis
    ax = axes[1, 2]
    if "saliency" in sample_data["attributions"]:
        saliency_scores = sample_data["attributions"]["saliency"].sum(axis=-1)
        top_tokens = np.argsort(np.abs(saliency_scores))[-5:]
        
        token_names = [tokens[i] for i in top_tokens]
        token_scores = [saliency_scores[i] for i in top_tokens]
        
        bars = ax.barh(token_names, token_scores, color='orange')
        ax.set_title('Top Contributing Tokens')
        ax.set_xlabel('Attribution Score')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
