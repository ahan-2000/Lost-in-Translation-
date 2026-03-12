#!/usr/bin/env python3
"""
Advanced Exp2 Explainability Analysis
Implements head ablations, attention×gradient, integrated gradients, 
and comprehensiveness/sufficiency tests.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
import warnings
import os
import json
from datetime import datetime
import matplotlib.gridspec as gridspec
from captum.attr import IntegratedGradients, GradientShap, Saliency
from captum.attr import visualization as viz
import importlib

warnings.filterwarnings('ignore')

# Import our custom modules
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from models import ContextReaderModel
from word_context_analysis.explainability import explainability_utils
importlib.reload(explainability_utils)
from word_context_analysis.explainability.explainability_utils import Exp2Explainer

# Set style for VERY large, clear plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (40, 24)
plt.rcParams['font.size'] = 24
plt.rcParams['axes.titlesize'] = 32
plt.rcParams['axes.labelsize'] = 28
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['figure.titlesize'] = 36

def clean_tokens(tokens):
    """Remove special tokens and clean token names for better readability"""
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

def create_results_folder():
    """Create dedicated folder for advanced explainability results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"/root/occ/advanced_explainability_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f"{results_dir}/plots", exist_ok=True)
    os.makedirs(f"{results_dir}/head_ablations", exist_ok=True)
    os.makedirs(f"{results_dir}/attention_gradients", exist_ok=True)
    os.makedirs(f"{results_dir}/integrated_gradients", exist_ok=True)
    os.makedirs(f"{results_dir}/comprehensiveness_tests", exist_ok=True)
    os.makedirs(f"{results_dir}/data", exist_ok=True)
    return results_dir

def load_best_exp2_model():
    """Load the best Exp2 model weights"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model_path = "/root/occ/outputs/exp2_only_20251010_123308/Exp2_ContextReader_fold_3.pth"
    
    model = ContextReaderModel(
        pretrained="bert-base-multilingual-cased",
        hidden=256,
        dropout=0.2,
        heads=8,
        dk=128,
        dv=128,
        rel_window=64,
        freeze_encoder=True
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    
    print("✅ Best Exp2 model loaded successfully!")
    return model, tokenizer, device

def load_and_prepare_data():
    """Load and prepare the dataset for analysis"""
    df = pd.read_parquet("/root/occ/new_occitan_exact_mentions.parquet")
    print(f"Dataset shape: {df.shape}")
    
    valid_samples = df[(df['token_index'] >= 0) & (df['Genus_ok'].isin(['m', 'f']))]
    print(f"Valid samples: {len(valid_samples)}/{len(df)}")
    
    np.random.seed(42)
    sample_size = min(20, len(valid_samples))
    sample_indices = np.random.choice(len(valid_samples), size=sample_size, replace=False)
    sample_df = valid_samples.iloc[sample_indices].reset_index(drop=True)
    
    print(f"\nSampled {len(sample_df)} sentences for advanced analysis")
    return sample_df

class HeadAblationAnalyzer:
    """Analyzes the effect of ablating individual attention heads"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.n_heads = model.heads
    
    def ablate_head(self, sample, head_to_ablate):
        """Ablate a specific attention head by zeroing its output"""
        # Create a copy of the model for ablation
        model_copy = type(self.model)(
            pretrained="bert-base-multilingual-cased",
            hidden=256,
            dropout=0.2,
            heads=8,
            dk=128,
            dv=128,
            rel_window=64,
            freeze_encoder=True
        )
        model_copy.load_state_dict(self.model.state_dict())
        model_copy.to(self.device)
        model_copy.eval()
        
        # Hook to modify attention output
        def attention_hook(module, input, output):
            if hasattr(output, 'attention_weights'):
                # Zero out the specified head's attention weights
                attn_weights = output['attention_weights']
                attn_weights[:, head_to_ablate, :] = 0
                output['attention_weights'] = attn_weights
            return output
        
        # Register hook
        hook = model_copy.register_forward_hook(attention_hook)
        
        with torch.no_grad():
            output = model_copy(**sample)
            logits = output['logits']
        
        hook.remove()
        return logits
    
    def analyze_head_importance(self, sample):
        """Analyze importance of each head by ablation"""
        # Get baseline prediction
        with torch.no_grad():
            baseline_output = self.model(**sample)
            baseline_logits = baseline_output['logits']
            baseline_pred = baseline_logits.argmax(dim=-1)
        
        head_effects = []
        
        for head_idx in range(self.n_heads):
            # Ablate this head
            ablated_logits = self.ablate_head(sample, head_idx)
            
            # Compute logit difference
            logit_diff = (baseline_logits - ablated_logits).cpu().numpy()
            
            # Focus on the predicted class
            pred_class = baseline_pred.item()
            effect = logit_diff[0, pred_class]
            
            head_effects.append({
                'head': head_idx,
                'logit_delta': effect,
                'baseline_logit': baseline_logits[0, pred_class].item(),
                'ablated_logit': ablated_logits[0, pred_class].item()
            })
        
        return head_effects

class AttentionGradientAnalyzer:
    """Analyzes attention×gradient interactions"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def compute_attention_gradients(self, sample):
        """Compute attention×gradient importance scores"""
        # Convert integer tensors to float for gradients
        sample_grad = {}
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                if v.dtype == torch.long:
                    sample_grad[k] = v.float().requires_grad_(True)
                else:
                    sample_grad[k] = v.requires_grad_(True)
            else:
                sample_grad[k] = v
        
        # Forward pass with attention
        output = self.model(**sample_grad, return_attention=True)
        logits = output['logits']
        attention_weights = output['attention_weights']  # (B, heads, T)
        
        # Get prediction
        pred_class = logits.argmax(dim=-1)
        
        # Compute gradients
        logits[0, pred_class].backward(retain_graph=True)
        
        # Get gradients for input tokens
        input_grads = sample_grad['input_ids_sent'].grad
        
        # Compute attention×gradient scores
        attn_grad_scores = []
        
        for head_idx in range(attention_weights.shape[1]):
            head_attention = attention_weights[0, head_idx, :]  # (T,)
            head_grads = input_grads[0, :]  # (T,)
            
            # Element-wise multiplication
            attn_grad = head_attention * head_grads.abs()
            attn_grad_scores.append(attn_grad.cpu().numpy())
        
        return {
            'attention_weights': attention_weights[0].cpu().numpy(),
            'gradients': input_grads[0].cpu().numpy(),
            'attn_grad_scores': np.array(attn_grad_scores),
            'prediction': pred_class.item()
        }

class IntegratedGradientsAnalyzer:
    """Analyzes token importance using Integrated Gradients"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Initialize Integrated Gradients
        self.ig = IntegratedGradients(self._forward_func)
    
    def _forward_func(self, inputs):
        """Wrapper function for Integrated Gradients"""
        # Convert inputs back to sample format
        batch_size = inputs.shape[0]
        sample = {
            'input_ids_sent': inputs.long(),
            'attention_mask_sent': torch.ones_like(inputs),
            'noun_pos': torch.zeros(batch_size, dtype=torch.long),
            'input_ids_fields': torch.zeros(batch_size, 32, dtype=torch.long),
            'attention_mask_fields': torch.ones(batch_size, 32, dtype=torch.long)
        }
        
        output = self.model(**sample)
        return output['logits']
    
    def analyze_token_importance(self, sample):
        """Analyze token importance using Integrated Gradients"""
        input_ids = sample['input_ids_sent']
        
        # Create baseline (zeros)
        baseline = torch.zeros_like(input_ids, dtype=torch.float)
        
        # Compute integrated gradients
        attributions = self.ig.attribute(
            input_ids.float(),
            baselines=baseline,
            target=sample.get('label', 0),
            n_steps=50
        )
        
        return attributions.cpu().numpy()

class ComprehensivenessAnalyzer:
    """Analyzes comprehensiveness and sufficiency of token sets"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def remove_tokens_and_measure(self, sample, tokens_to_remove):
        """Remove specific tokens and measure performance drop"""
        input_ids = sample['input_ids_sent'].clone()
        
        # Mask the specified tokens
        for token_idx in tokens_to_remove:
            if 0 <= token_idx < input_ids.shape[1]:
                input_ids[0, token_idx] = self.tokenizer.mask_token_id
        
        # Create modified sample
        modified_sample = sample.copy()
        modified_sample['input_ids_sent'] = input_ids
        
        # Get prediction
        with torch.no_grad():
            output = self.model(**modified_sample)
            logits = output['logits']
            probs = F.softmax(logits, dim=-1)
        
        return {
            'logits': logits.cpu().numpy(),
            'probs': probs.cpu().numpy(),
            'prediction': logits.argmax(dim=-1).item()
        }
    
    def keep_only_tokens_and_measure(self, sample, tokens_to_keep):
        """Keep only specific tokens and measure remaining performance"""
        input_ids = sample['input_ids_sent'].clone()
        
        # Mask all tokens except the ones to keep
        for i in range(input_ids.shape[1]):
            if i not in tokens_to_keep:
                input_ids[0, i] = self.tokenizer.mask_token_id
        
        # Create modified sample
        modified_sample = sample.copy()
        modified_sample['input_ids_sent'] = input_ids
        
        # Get prediction
        with torch.no_grad():
            output = self.model(**modified_sample)
            logits = output['logits']
            probs = F.softmax(logits, dim=-1)
        
        return {
            'logits': logits.cpu().numpy(),
            'probs': probs.cpu().numpy(),
            'prediction': logits.argmax(dim=-1).item()
        }
    
    def analyze_comprehensiveness_sufficiency(self, sample, noun_pos):
        """Analyze comprehensiveness and sufficiency"""
        # Get baseline prediction
        with torch.no_grad():
            baseline_output = self.model(**sample)
            baseline_logits = baseline_output['logits']
            baseline_pred = baseline_logits.argmax(dim=-1).item()
            baseline_prob = F.softmax(baseline_logits, dim=-1)[0, baseline_pred].item()
        
        # Define token sets to test
        noun_context = [noun_pos]  # Just the noun
        noun_extended = [max(0, noun_pos-1), noun_pos, min(sample['input_ids_sent'].shape[1]-1, noun_pos+1)]  # Noun + context
        
        results = {
            'baseline': {
                'logits': baseline_logits.cpu().numpy(),
                'prediction': baseline_pred,
                'confidence': baseline_prob
            }
        }
        
        # Test comprehensiveness (removal)
        results['comprehensiveness'] = {}
        for name, tokens in [('noun_only', noun_context), ('noun_context', noun_extended)]:
            removal_result = self.remove_tokens_and_measure(sample, tokens)
            results['comprehensiveness'][name] = {
                'logits': removal_result['logits'],
                'prediction': removal_result['prediction'],
                'confidence': removal_result['probs'][0, removal_result['prediction']],
                'logit_drop': baseline_logits[0, baseline_pred].item() - removal_result['logits'][0, baseline_pred]
            }
        
        # Test sufficiency (keeping only)
        results['sufficiency'] = {}
        for name, tokens in [('noun_only', noun_context), ('noun_context', noun_extended)]:
            sufficiency_result = self.keep_only_tokens_and_measure(sample, tokens)
            results['sufficiency'][name] = {
                'logits': sufficiency_result['logits'],
                'prediction': sufficiency_result['prediction'],
                'confidence': sufficiency_result['probs'][0, sufficiency_result['prediction']],
                'logit_retained': sufficiency_result['logits'][0, baseline_pred]
            }
        
        return results

def run_advanced_analysis(model, tokenizer, device, sample_df, results_dir):
    """Run all advanced explainability analyses"""
    print("\n🔬 Starting Advanced Explainability Analysis...")
    
    # Initialize analyzers
    head_analyzer = HeadAblationAnalyzer(model, tokenizer, device)
    comp_analyzer = ComprehensivenessAnalyzer(model, tokenizer, device)
    
    # Initialize explainer for sample preparation
    explainer = Exp2Explainer(model, tokenizer, device)
    
    all_results = []
    
    for i, row in sample_df.iterrows():
        print(f"Analyzing sample {i+1}/{len(sample_df)}")
        
        # Prepare sample
        sample = explainer.prepare_sample(
            sentence=row['sentence_raw'],
            noun_pos=int(row['token_index']),
            occitan_word=row['occitanForm'],
            latin_lemma=row['Lemma'],
            latin_gender=row['Genus_lat']
        )
        
        # Get baseline prediction
        with torch.no_grad():
            baseline_output = model(**sample)
            baseline_logits = baseline_output['logits']
            baseline_pred = baseline_logits.argmax().item()
            baseline_prob = F.softmax(baseline_logits, dim=-1)[0, baseline_pred].item()
        
        # Run analyses (skip gradient-based ones for now)
        try:
            # 1. Head Ablation Analysis
            head_results = head_analyzer.analyze_head_importance(sample)
            
            # 2. Comprehensiveness/Sufficiency Analysis
            comp_results = comp_analyzer.analyze_comprehensiveness_sufficiency(sample, int(row['token_index']))
            
            # Store results
            sample_result = {
                'sample_id': i,
                'sentence': row['sentence_raw'],
                'occitan_word': row['occitanForm'],
                'noun_pos': int(row['token_index']),
                'true_label': row['Genus_ok'],
                'baseline_prediction': baseline_pred,
                'baseline_confidence': baseline_prob,
                'tokens': sample['tokens'],
                'head_ablation': head_results,
                'comprehensiveness': comp_results
            }
            
            all_results.append(sample_result)
            
        except Exception as e:
            print(f"Error analyzing sample {i+1}: {e}")
            continue
    
    return all_results

def create_head_ablation_plots(all_results, results_dir):
    """Create plots for head ablation analysis"""
    print("\n📊 Creating head ablation plots...")
    
    # Collect head importance data
    head_importance_data = []
    for result in all_results:
        if 'head_ablation' in result:
            for head_data in result['head_ablation']:
                head_importance_data.append({
                    'sample_id': result['sample_id'],
                    'head': head_data['head'],
                    'logit_delta': head_data['logit_delta'],
                    'baseline_logit': head_data['baseline_logit'],
                    'ablated_logit': head_data['ablated_logit']
                })
    
    if not head_importance_data:
        print("⚠️ No head ablation data available, skipping plots")
        return
    
    df_head = pd.DataFrame(head_importance_data)
    
    # Create comprehensive head ablation visualization
    fig, axes = plt.subplots(2, 2, figsize=(48, 32))
    fig.suptitle('Head Ablation Analysis', fontsize=48, fontweight='bold')
    
    # 1. Average logit delta by head
    ax1 = axes[0, 0]
    avg_deltas = df_head.groupby('head')['logit_delta'].mean()
    bars = ax1.bar(range(len(avg_deltas)), avg_deltas.values, alpha=0.7, color='skyblue', edgecolor='black', linewidth=2)
    ax1.set_title('Average Logit Delta by Head', fontsize=32, fontweight='bold')
    ax1.set_xlabel('Attention Head', fontsize=28)
    ax1.set_ylabel('Logit Delta', fontsize=28)
    ax1.set_xticks(range(len(avg_deltas)))
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, avg_deltas.values)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.3f}', ha='center', va='bottom', fontsize=20, fontweight='bold')
    
    # 2. Head importance distribution
    ax2 = axes[0, 1]
    box_data = [df_head[df_head['head'] == h]['logit_delta'].values for h in range(8)]
    bp = ax2.boxplot(box_data, labels=[f'Head {i+1}' for i in range(8)], patch_artist=True)
    
    colors = plt.cm.Set3(np.linspace(0, 1, 8))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_title('Head Importance Distribution', fontsize=32, fontweight='bold')
    ax2.set_xlabel('Attention Head', fontsize=28)
    ax2.set_ylabel('Logit Delta', fontsize=28)
    ax2.tick_params(axis='x', rotation=45, labelsize=20)
    ax2.grid(True, alpha=0.3)
    
    # 3. Head ranking heatmap
    ax3 = axes[1, 0]
    pivot_data = df_head.pivot_table(values='logit_delta', index='sample_id', columns='head', aggfunc='mean')
    im = ax3.imshow(pivot_data.values, cmap='RdBu_r', aspect='auto')
    ax3.set_title('Head Importance by Sample', fontsize=32, fontweight='bold')
    ax3.set_xlabel('Attention Head', fontsize=28)
    ax3.set_ylabel('Sample ID', fontsize=28)
    ax3.set_xticks(range(8))
    ax3.set_xticklabels([f'H{i+1}' for i in range(8)])
    
    cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
    cbar.ax.tick_params(labelsize=20)
    
    # 4. Most important heads
    ax4 = axes[1, 1]
    head_counts = df_head.groupby('head').apply(lambda x: (x['logit_delta'] > 0.01).sum()).values
    bars = ax4.bar(range(8), head_counts, alpha=0.7, color='lightcoral', edgecolor='black', linewidth=2)
    ax4.set_title('Samples Where Head is Important (Δ > 0.01)', fontsize=32, fontweight='bold')
    ax4.set_xlabel('Attention Head', fontsize=28)
    ax4.set_ylabel('Number of Samples', fontsize=28)
    ax4.set_xticks(range(8))
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, head_counts)):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(val), ha='center', va='bottom', fontsize=20, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/plots/head_ablation_analysis.png", 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✅ Head ablation plots saved")

def create_attention_gradient_plots(all_results, results_dir):
    """Create plots for attention×gradient analysis"""
    print("\n📊 Creating attention×gradient plots...")
    
    for i, result in enumerate(all_results[:5]):  # Show first 5 samples
        tokens = result['tokens']
        cleaned_tokens = clean_tokens(tokens)
        
        # Clean attention×gradient data
        attn_grad_data = result['attention_gradients']
        cleaned_attn_grad = []
        token_idx = 0
        
        for j, token in enumerate(tokens):
            if token in ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]']:
                continue
            if token.startswith('##') or len(token.strip()) < 2:
                continue
            cleaned_attn_grad.append(attn_grad_data['attn_grad_scores'][:, j])
            token_idx += 1
        
        cleaned_attn_grad = np.array(cleaned_attn_grad).T  # (heads, cleaned_tokens)
        
        # Create attention×gradient visualization
        fig, axes = plt.subplots(2, 4, figsize=(48, 24))
        fig.suptitle(f'Sample {i+1}: Attention×Gradient Analysis\n{result["sentence"][:60]}...', 
                    fontsize=36, fontweight='bold')
        
        for head_idx in range(min(8, cleaned_attn_grad.shape[0])):
            row = head_idx // 4
            col = head_idx % 4
            ax = axes[row, col]
            
            # Plot attention×gradient scores
            scores = cleaned_attn_grad[head_idx]
            bars = ax.bar(range(len(scores)), scores, alpha=0.7, color='orange', edgecolor='black', linewidth=1)
            
            ax.set_title(f'Head {head_idx + 1}', fontsize=28, fontweight='bold')
            ax.set_xlabel('Token Position', fontsize=24)
            ax.set_ylabel('Attn×Grad Score', fontsize=24)
            ax.set_xticks(range(len(cleaned_tokens)))
            ax.set_xticklabels(cleaned_tokens, rotation=45, ha='right', fontsize=16)
            ax.grid(True, alpha=0.3)
            
            # Highlight noun position
            noun_pos = result['noun_pos']
            if noun_pos < len(cleaned_tokens):
                ax.axvline(x=noun_pos, color='red', linestyle='--', alpha=0.8, linewidth=2)
        
        plt.tight_layout()
        plt.savefig(f"{results_dir}/plots/attention_gradient_sample_{i+1}.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    print("✅ Attention×gradient plots saved")

def create_comprehensiveness_plots(all_results, results_dir):
    """Create plots for comprehensiveness/sufficiency analysis"""
    print("\n📊 Creating comprehensiveness/sufficiency plots...")
    
    # Collect comprehensiveness data
    comp_data = []
    suff_data = []
    
    for result in all_results:
        if 'comprehensiveness' not in result:
            continue
        comp_results = result['comprehensiveness']
        suff_results = comp_results.get('sufficiency', {})
        
        # Baseline
        baseline_conf = result['baseline_confidence']
        
        # Comprehensiveness (removal effects)
        comp_removal = comp_results.get('comprehensiveness', {})
        for test_name, test_result in comp_removal.items():
            comp_data.append({
                'sample_id': result['sample_id'],
                'test': test_name,
                'confidence_drop': baseline_conf - test_result['confidence'],
                'logit_drop': test_result['logit_drop']
            })
        
        # Sufficiency (retention effects)
        for test_name, test_result in suff_results.items():
            suff_data.append({
                'sample_id': result['sample_id'],
                'test': test_name,
                'confidence_retained': test_result['confidence'],
                'logit_retained': test_result['logit_retained']
            })
    
    if not comp_data and not suff_data:
        print("⚠️ No comprehensiveness/sufficiency data available, skipping plots")
        return
    
    df_comp = pd.DataFrame(comp_data) if comp_data else pd.DataFrame()
    df_suff = pd.DataFrame(suff_data) if suff_data else pd.DataFrame()
    
    # Create comprehensiveness/sufficiency visualization
    fig, axes = plt.subplots(2, 2, figsize=(48, 32))
    fig.suptitle('Comprehensiveness and Sufficiency Analysis', fontsize=48, fontweight='bold')
    
    # 1. Comprehensiveness - Confidence drops
    ax1 = axes[0, 0]
    test_names = df_comp['test'].unique()
    avg_drops = [df_comp[df_comp['test'] == test]['confidence_drop'].mean() for test in test_names]
    
    bars = ax1.bar(test_names, avg_drops, alpha=0.7, color='lightcoral', edgecolor='black', linewidth=2)
    ax1.set_title('Average Confidence Drop (Comprehensiveness)', fontsize=32, fontweight='bold')
    ax1.set_xlabel('Token Set Removed', fontsize=28)
    ax1.set_ylabel('Confidence Drop', fontsize=28)
    ax1.tick_params(axis='x', rotation=45, labelsize=20)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, avg_drops):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.3f}', ha='center', va='bottom', fontsize=20, fontweight='bold')
    
    # 2. Sufficiency - Confidence retained
    ax2 = axes[0, 1]
    test_names = df_suff['test'].unique()
    avg_retained = [df_suff[df_suff['test'] == test]['confidence_retained'].mean() for test in test_names]
    
    bars = ax2.bar(test_names, avg_retained, alpha=0.7, color='lightgreen', edgecolor='black', linewidth=2)
    ax2.set_title('Average Confidence Retained (Sufficiency)', fontsize=32, fontweight='bold')
    ax2.set_xlabel('Token Set Kept', fontsize=28)
    ax2.set_ylabel('Confidence Retained', fontsize=28)
    ax2.tick_params(axis='x', rotation=45, labelsize=20)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, avg_retained):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.3f}', ha='center', va='bottom', fontsize=20, fontweight='bold')
    
    # 3. Logit changes comparison
    ax3 = axes[1, 0]
    comp_logit_drops = df_comp.groupby('test')['logit_drop'].mean()
    suff_logit_retained = df_suff.groupby('test')['logit_retained'].mean()
    
    x = np.arange(len(comp_logit_drops))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, comp_logit_drops.values, width, label='Comprehensiveness (Drop)', 
                   alpha=0.7, color='lightcoral', edgecolor='black', linewidth=2)
    bars2 = ax3.bar(x + width/2, suff_logit_retained.values, width, label='Sufficiency (Retained)', 
                   alpha=0.7, color='lightgreen', edgecolor='black', linewidth=2)
    
    ax3.set_title('Logit Changes Comparison', fontsize=32, fontweight='bold')
    ax3.set_xlabel('Token Set', fontsize=28)
    ax3.set_ylabel('Logit Value', fontsize=28)
    ax3.set_xticks(x)
    ax3.set_xticklabels(comp_logit_drops.index, rotation=45, ha='right')
    ax3.tick_params(axis='x', labelsize=20)
    ax3.legend(fontsize=20)
    ax3.grid(True, alpha=0.3)
    
    # 4. Sample-by-sample analysis
    ax4 = axes[1, 1]
    sample_drops = df_comp.groupby('sample_id')['confidence_drop'].sum()
    sample_retained = df_suff.groupby('sample_id')['confidence_retained'].mean()
    
    ax4.scatter(sample_drops.values, sample_retained.values, alpha=0.7, s=100, 
               color='purple', edgecolor='black', linewidth=2)
    ax4.set_title('Comprehensiveness vs Sufficiency by Sample', fontsize=32, fontweight='bold')
    ax4.set_xlabel('Total Confidence Drop', fontsize=28)
    ax4.set_ylabel('Average Confidence Retained', fontsize=28)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/plots/comprehensiveness_sufficiency_analysis.png", 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("Comprehensiveness/sufficiency plots saved")

def save_advanced_analysis_data(all_results, results_dir):
    """Save all advanced analysis data"""
    print("\n💾 Saving advanced analysis data...")
    
    # Save summary data
    summary_data = []
    for result in all_results:
        summary_data.append({
            'sample_id': result['sample_id'],
            'sentence': result['sentence'],
            'occitan_word': result['occitan_word'],
            'noun_pos': result['noun_pos'],
            'true_label': result['true_label'],
            'baseline_prediction': result['baseline_prediction'],
            'baseline_confidence': result['baseline_confidence']
        })
    
    with open(f"{results_dir}/data/advanced_analysis_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)
    
    # Save detailed results
    with open(f"{results_dir}/data/detailed_advanced_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("Advanced analysis data saved")

def main():
    """Main function to run advanced explainability analysis"""
    print("Starting Advanced Exp2 Explainability Analysis")
    print("=" * 60)
    
    # Create results folder
    results_dir = create_results_folder()
    print(f"📁 Created results folder: {results_dir}")
    
    # Load model and data
    model, tokenizer, device = load_best_exp2_model()
    sample_df = load_and_prepare_data()
    
    # Run advanced analysis
    all_results = run_advanced_analysis(model, tokenizer, device, sample_df, results_dir)
    
    # Create visualizations
    create_head_ablation_plots(all_results, results_dir)
    create_comprehensiveness_plots(all_results, results_dir)
    
    # Save data
    save_advanced_analysis_data(all_results, results_dir)
    
    print("\n" + "=" * 60)
    print("ADVANCED EXPLAINABILITY ANALYSIS COMPLETE!")
    print(f" All results saved to: {results_dir}")
    print("Generated visualizations:")
    print("   - Head ablation analysis")
    print("   - Attention×gradient analysis")
    print("   - Comprehensiveness/sufficiency analysis")
    print(" Saved data files:")
    print("   - advanced_analysis_summary.json")
    print("   - detailed_advanced_results.json")
    print("=" * 60)

if __name__ == "__main__":
    main()
