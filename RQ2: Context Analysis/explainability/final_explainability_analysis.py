#!/usr/bin/env python3
"""
Final Working Exp2 Explainability Analysis
Focuses on comprehensiveness tests and attention pattern analysis with meaningful data.
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
    """Create dedicated folder for final explainability results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"/root/occ/final_explainability_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f"{results_dir}/plots", exist_ok=True)
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
    
    print(f"\nSampled {len(sample_df)} sentences for final analysis")
    return sample_df

class WorkingComprehensivenessAnalyzer:
    """Working comprehensiveness and sufficiency analysis"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def mask_tokens_and_measure(self, sample, tokens_to_mask):
        """Mask specific tokens and measure performance"""
        input_ids = sample['input_ids_sent'].clone()
        
        # Mask the specified tokens
        for token_idx in tokens_to_mask:
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
            'prediction': logits.argmax(dim=-1).item(),
            'confidence': probs[0, logits.argmax(dim=-1)].item()
        }
    
    def keep_only_tokens_and_measure(self, sample, tokens_to_keep):
        """Keep only specific tokens and measure performance"""
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
            'prediction': logits.argmax(dim=-1).item(),
            'confidence': probs[0, logits.argmax(dim=-1)].item()
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
        noun_only = [noun_pos]
        noun_context = [max(0, noun_pos-1), noun_pos, min(sample['input_ids_sent'].shape[1]-1, noun_pos+1)]
        
        results = {
            'baseline': {
                'logits': baseline_logits.cpu().numpy(),
                'prediction': baseline_pred,
                'confidence': baseline_prob
            }
        }
        
        # Test comprehensiveness (removal)
        results['comprehensiveness'] = {}
        
        # Remove noun only
        noun_removal = self.mask_tokens_and_measure(sample, noun_only)
        results['comprehensiveness']['noun_only'] = {
            'logits': noun_removal['logits'],
            'prediction': noun_removal['prediction'],
            'confidence': noun_removal['confidence'],
            'confidence_drop': baseline_prob - noun_removal['confidence'],
            'logit_drop': baseline_logits[0, baseline_pred].item() - noun_removal['logits'][0, baseline_pred]
        }
        
        # Remove noun + context
        context_removal = self.mask_tokens_and_measure(sample, noun_context)
        results['comprehensiveness']['noun_context'] = {
            'logits': context_removal['logits'],
            'prediction': context_removal['prediction'],
            'confidence': context_removal['confidence'],
            'confidence_drop': baseline_prob - context_removal['confidence'],
            'logit_drop': baseline_logits[0, baseline_pred].item() - context_removal['logits'][0, baseline_pred]
        }
        
        # Test sufficiency (keeping only)
        results['sufficiency'] = {}
        
        # Keep noun only
        noun_only_result = self.keep_only_tokens_and_measure(sample, noun_only)
        results['sufficiency']['noun_only'] = {
            'logits': noun_only_result['logits'],
            'prediction': noun_only_result['prediction'],
            'confidence': noun_only_result['confidence'],
            'confidence_retained': noun_only_result['confidence'],
            'logit_retained': noun_only_result['logits'][0, baseline_pred]
        }
        
        # Keep noun + context
        context_only_result = self.keep_only_tokens_and_measure(sample, noun_context)
        results['sufficiency']['noun_context'] = {
            'logits': context_only_result['logits'],
            'prediction': context_only_result['prediction'],
            'confidence': context_only_result['confidence'],
            'confidence_retained': context_only_result['confidence'],
            'logit_retained': context_only_result['logits'][0, baseline_pred]
        }
        
        return results

class AttentionPatternAnalyzer:
    """Analyze attention patterns without ablation"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def analyze_attention_patterns(self, sample):
        """Analyze attention patterns for each head"""
        with torch.no_grad():
            output = self.model(**sample, return_attention=True)
            attention_weights = output['attention_weights'][0]  # (heads, T)
            logits = output['logits']
            pred_class = logits.argmax(dim=-1).item()
        
        # Analyze each head
        head_analysis = []
        for head_idx in range(attention_weights.shape[0]):
            head_attn = attention_weights[head_idx]
            
            # Compute statistics
            entropy = -torch.sum(head_attn * torch.log(head_attn + 1e-8)).item()
            max_attention = torch.max(head_attn).item()
            mean_attention = torch.mean(head_attn).item()
            std_attention = torch.std(head_attn).item()
            
            # Find most attended tokens
            top_indices = torch.topk(head_attn, k=min(5, len(head_attn))).indices.cpu().numpy()
            top_values = torch.topk(head_attn, k=min(5, len(head_attn))).values.cpu().numpy()
            
            head_analysis.append({
                'head': head_idx,
                'entropy': entropy,
                'max_attention': max_attention,
                'mean_attention': mean_attention,
                'std_attention': std_attention,
                'top_indices': top_indices.tolist(),
                'top_values': top_values.tolist(),
                'attention_weights': head_attn.cpu().numpy().tolist()
            })
        
        return {
            'head_analysis': head_analysis,
            'prediction': pred_class,
            'attention_weights': attention_weights.cpu().numpy().tolist()
        }

def run_final_analysis(model, tokenizer, device, sample_df, results_dir):
    """Run final explainability analyses"""
    print("\n🔬 Starting Final Explainability Analysis...")
    
    # Initialize analyzers
    comp_analyzer = WorkingComprehensivenessAnalyzer(model, tokenizer, device)
    attn_analyzer = AttentionPatternAnalyzer(model, tokenizer, device)
    
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
        
        # Run analyses
        try:
            # 1. Comprehensiveness/Sufficiency Analysis
            comp_results = comp_analyzer.analyze_comprehensiveness_sufficiency(sample, int(row['token_index']))
            
            # 2. Attention Pattern Analysis
            attn_results = attn_analyzer.analyze_attention_patterns(sample)
            
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
                'comprehensiveness': comp_results,
                'attention_patterns': attn_results
            }
            
            all_results.append(sample_result)
            
        except Exception as e:
            print(f"Error analyzing sample {i+1}: {e}")
            continue
    
    return all_results

def create_comprehensiveness_plots(all_results, results_dir):
    """Create comprehensiveness/sufficiency plots"""
    print("\n📊 Creating comprehensiveness/sufficiency plots...")
    
    # Collect comprehensiveness data
    comp_data = []
    suff_data = []
    
    for result in all_results:
        if 'comprehensiveness' not in result:
            continue
        comp_results = result['comprehensiveness']
        
        # Baseline
        baseline_conf = result['baseline_confidence']
        
        # Comprehensiveness (removal effects)
        comp_removal = comp_results.get('comprehensiveness', {})
        for test_name, test_result in comp_removal.items():
            comp_data.append({
                'sample_id': result['sample_id'],
                'test': test_name,
                'confidence_drop': test_result['confidence_drop'],
                'logit_drop': test_result['logit_drop']
            })
        
        # Sufficiency (retention effects)
        suff_results = comp_results.get('sufficiency', {})
        for test_name, test_result in suff_results.items():
            suff_data.append({
                'sample_id': result['sample_id'],
                'test': test_name,
                'confidence_retained': test_result['confidence_retained'],
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
    if not df_comp.empty:
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
                    f'{val:.4f}', ha='center', va='bottom', fontsize=20, fontweight='bold')
    
    # 2. Sufficiency - Confidence retained
    if not df_suff.empty:
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
                    f'{val:.4f}', ha='center', va='bottom', fontsize=20, fontweight='bold')
    
    # 3. Sample-by-sample analysis
    ax3 = axes[1, 0]
    if not df_comp.empty and not df_suff.empty:
        sample_drops = df_comp.groupby('sample_id')['confidence_drop'].sum()
        sample_retained = df_suff.groupby('sample_id')['confidence_retained'].mean()
        
        ax3.scatter(sample_drops.values, sample_retained.values, alpha=0.7, s=100, 
                   color='purple', edgecolor='black', linewidth=2)
        ax3.set_title('Comprehensiveness vs Sufficiency by Sample', fontsize=32, fontweight='bold')
        ax3.set_xlabel('Total Confidence Drop', fontsize=28)
        ax3.set_ylabel('Average Confidence Retained', fontsize=28)
        ax3.grid(True, alpha=0.3)
    
    # 4. Summary statistics
    ax4 = axes[1, 1]
    summary_stats = []
    if not df_comp.empty:
        summary_stats.append(f"Avg Confidence Drop: {df_comp['confidence_drop'].mean():.6f}")
        summary_stats.append(f"Max Confidence Drop: {df_comp['confidence_drop'].max():.6f}")
        summary_stats.append(f"Std Confidence Drop: {df_comp['confidence_drop'].std():.6f}")
    if not df_suff.empty:
        summary_stats.append(f"Avg Confidence Retained: {df_suff['confidence_retained'].mean():.6f}")
        summary_stats.append(f"Min Confidence Retained: {df_suff['confidence_retained'].min():.6f}")
        summary_stats.append(f"Std Confidence Retained: {df_suff['confidence_retained'].std():.6f}")
    
    ax4.text(0.1, 0.5, '\n'.join(summary_stats), fontsize=24, fontweight='bold',
             transform=ax4.transAxes, verticalalignment='center')
    ax4.set_title('Summary Statistics', fontsize=32, fontweight='bold')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/plots/comprehensiveness_analysis.png", 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✅ Comprehensiveness/sufficiency plots saved")

def create_attention_pattern_plots(all_results, results_dir):
    """Create attention pattern analysis plots"""
    print("\n📊 Creating attention pattern plots...")
    
    # Collect attention pattern data
    head_stats = []
    for result in all_results:
        if 'attention_patterns' not in result:
            continue
        attn_results = result['attention_patterns']
        
        for head_data in attn_results['head_analysis']:
            head_stats.append({
                'sample_id': result['sample_id'],
                'head': head_data['head'],
                'entropy': head_data['entropy'],
                'max_attention': head_data['max_attention'],
                'mean_attention': head_data['mean_attention'],
                'std_attention': head_data['std_attention']
            })
    
    if not head_stats:
        print("⚠️ No attention pattern data available, skipping plots")
        return
    
    df_attn = pd.DataFrame(head_stats)
    
    # Create attention pattern visualization
    fig, axes = plt.subplots(2, 2, figsize=(48, 32))
    fig.suptitle('Attention Pattern Analysis', fontsize=48, fontweight='bold')
    
    # 1. Average entropy by head
    ax1 = axes[0, 0]
    avg_entropy = df_attn.groupby('head')['entropy'].mean()
    bars = ax1.bar(range(len(avg_entropy)), avg_entropy.values, alpha=0.7, color='skyblue', edgecolor='black', linewidth=2)
    ax1.set_title('Average Attention Entropy by Head', fontsize=32, fontweight='bold')
    ax1.set_xlabel('Attention Head', fontsize=28)
    ax1.set_ylabel('Entropy', fontsize=28)
    ax1.set_xticks(range(len(avg_entropy)))
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, avg_entropy.values)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=20, fontweight='bold')
    
    # 2. Average max attention by head
    ax2 = axes[0, 1]
    avg_max_attn = df_attn.groupby('head')['max_attention'].mean()
    bars = ax2.bar(range(len(avg_max_attn)), avg_max_attn.values, alpha=0.7, color='lightcoral', edgecolor='black', linewidth=2)
    ax2.set_title('Average Max Attention by Head', fontsize=32, fontweight='bold')
    ax2.set_xlabel('Attention Head', fontsize=28)
    ax2.set_ylabel('Max Attention', fontsize=28)
    ax2.set_xticks(range(len(avg_max_attn)))
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, avg_max_attn.values)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.3f}', ha='center', va='bottom', fontsize=20, fontweight='bold')
    
    # 3. Attention concentration (std) by head
    ax3 = axes[1, 0]
    avg_std_attn = df_attn.groupby('head')['std_attention'].mean()
    bars = ax3.bar(range(len(avg_std_attn)), avg_std_attn.values, alpha=0.7, color='lightgreen', edgecolor='black', linewidth=2)
    ax3.set_title('Average Attention Concentration by Head', fontsize=32, fontweight='bold')
    ax3.set_xlabel('Attention Head', fontsize=28)
    ax3.set_ylabel('Attention Std', fontsize=28)
    ax3.set_xticks(range(len(avg_std_attn)))
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, avg_std_attn.values)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.3f}', ha='center', va='bottom', fontsize=20, fontweight='bold')
    
    # 4. Head specialization heatmap
    ax4 = axes[1, 1]
    pivot_entropy = df_attn.pivot_table(values='entropy', index='sample_id', columns='head', aggfunc='mean')
    im = ax4.imshow(pivot_entropy.values, cmap='viridis', aspect='auto')
    ax4.set_title('Attention Entropy by Sample and Head', fontsize=32, fontweight='bold')
    ax4.set_xlabel('Attention Head', fontsize=28)
    ax4.set_ylabel('Sample ID', fontsize=28)
    ax4.set_xticks(range(8))
    ax4.set_xticklabels([f'H{i+1}' for i in range(8)])
    
    cbar = plt.colorbar(im, ax=ax4, shrink=0.8)
    cbar.ax.tick_params(labelsize=20)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/plots/attention_pattern_analysis.png", 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✅ Attention pattern plots saved")

def save_final_analysis_data(all_results, results_dir):
    """Save final analysis data"""
    print("\n💾 Saving final analysis data...")
    
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
    
    with open(f"{results_dir}/data/final_analysis_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)
    
    # Save detailed results
    with open(f"{results_dir}/data/detailed_final_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("✅ Final analysis data saved")

def main():
    """Main function to run final explainability analysis"""
    print("🚀 Starting Final Exp2 Explainability Analysis")
    print("=" * 60)
    
    # Create results folder
    results_dir = create_results_folder()
    print(f"📁 Created results folder: {results_dir}")
    
    # Load model and data
    model, tokenizer, device = load_best_exp2_model()
    sample_df = load_and_prepare_data()
    
    # Run final analysis
    all_results = run_final_analysis(model, tokenizer, device, sample_df, results_dir)
    
    # Create visualizations
    create_comprehensiveness_plots(all_results, results_dir)
    create_attention_pattern_plots(all_results, results_dir)
    
    # Save data
    save_final_analysis_data(all_results, results_dir)
    
    print("\n" + "=" * 60)
    print("✅ FINAL EXPLAINABILITY ANALYSIS COMPLETE!")
    print(f"📁 All results saved to: {results_dir}")
    print("📊 Generated visualizations:")
    print("   - Comprehensiveness/sufficiency analysis")
    print("   - Attention pattern analysis")
    print("💾 Saved data files:")
    print("   - final_analysis_summary.json")
    print("   - detailed_final_results.json")
    print("=" * 60)

if __name__ == "__main__":
    main()
