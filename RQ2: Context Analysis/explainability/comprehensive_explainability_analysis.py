#!/usr/bin/env python3
"""
Comprehensive Exp2 Explainability Analysis with Saliency Maps
Includes Switch metrics, head ablations, and word-by-word importance analysis.
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
    """Create dedicated folder for comprehensive explainability results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"/root/occ/comprehensive_explainability_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f"{results_dir}/plots", exist_ok=True)
    os.makedirs(f"{results_dir}/data", exist_ok=True)
    os.makedirs(f"{results_dir}/saliency_maps", exist_ok=True)
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
    
    print(f"\nSampled {len(sample_df)} sentences for comprehensive analysis")
    return sample_df

class SaliencyAnalyzer:
    """Analyze word-by-word importance using various saliency methods"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def compute_gradients(self, sample):
        """Compute gradients for saliency analysis"""
        # Enable gradient computation
        sample_grad = {}
        for key, value in sample.items():
            if isinstance(value, torch.Tensor) and value.dtype in [torch.float32, torch.float64]:
                sample_grad[key] = value.clone().detach().requires_grad_(True)
            else:
                sample_grad[key] = value
        
        # Forward pass
        output = self.model(**sample_grad)
        logits = output['logits']
        loss = self.ce_loss(logits, sample['label'])
        
        # Backward pass
        loss.backward()
        
        # Get gradients
        gradients = {}
        if 'input_ids_sent' in sample_grad and sample_grad['input_ids_sent'].grad is not None:
            gradients['input_ids_sent'] = sample_grad['input_ids_sent'].grad.cpu().numpy()
        
        return {
            'logits': logits.detach().cpu().numpy(),
            'loss': loss.detach().cpu().numpy(),
            'gradients': gradients
        }
    
    def compute_integrated_gradients(self, sample, steps=50):
        """Compute integrated gradients for word importance"""
        # Get baseline (all zeros)
        baseline_sample = sample.copy()
        baseline_sample['input_ids_sent'] = torch.zeros_like(sample['input_ids_sent'])
        
        # Get original sample
        original_sample = sample.copy()
        
        integrated_grads = []
        
        for i in range(steps + 1):
            alpha = i / steps
            
            # Interpolate between baseline and original
            interpolated_sample = original_sample.copy()
            interpolated_sample['input_ids_sent'] = (
                baseline_sample['input_ids_sent'] * (1 - alpha) + 
                original_sample['input_ids_sent'] * alpha
            )
            
            # Enable gradients
            interpolated_sample['input_ids_sent'] = interpolated_sample['input_ids_sent'].requires_grad_(True)
            
            # Forward pass
            output = self.model(**interpolated_sample)
            logits = output['logits']
            loss = self.ce_loss(logits, sample['label'])
            
            # Backward pass
            loss.backward()
            
            # Get gradients
            if interpolated_sample['input_ids_sent'].grad is not None:
                integrated_grads.append(interpolated_sample['input_ids_sent'].grad.cpu().numpy())
        
        # Average gradients
        if integrated_grads:
            avg_grads = np.mean(integrated_grads, axis=0)
            # Multiply by input difference
            input_diff = original_sample['input_ids_sent'].cpu().numpy() - baseline_sample['input_ids_sent'].cpu().numpy()
            integrated_grads_final = avg_grads * input_diff
        else:
            integrated_grads_final = np.zeros_like(sample['input_ids_sent'].cpu().numpy())
        
        return integrated_grads_final
    
    def compute_word_importance_by_occlusion(self, sample, tokens):
        """Compute word importance by occluding each word"""
        original_output = self.model(**sample)
        original_logits = original_output['logits']
        original_pred = original_logits.argmax(dim=-1).item()
        original_true_logit = original_logits[0, sample['label'].item()].item()
        
        word_importance = []
        
        for i, token in enumerate(tokens):
            if token in ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]']:
                continue
            
            # Create occluded sample
            occluded_sample = sample.copy()
            occluded_sample['input_ids_sent'] = sample['input_ids_sent'].clone()
            occluded_sample['input_ids_sent'][0, i] = self.tokenizer.mask_token_id
            
            # Get prediction
            with torch.no_grad():
                occluded_output = self.model(**occluded_sample)
                occluded_logits = occluded_output['logits']
                occluded_pred = occluded_logits.argmax(dim=-1).item()
                occluded_true_logit = occluded_logits[0, sample['label'].item()].item()
            
            # Compute importance
            logit_change = original_true_logit - occluded_true_logit
            prediction_change = 1 if original_pred != occluded_pred else 0
            
            word_importance.append({
                'token': token,
                'position': i,
                'logit_change': logit_change,
                'prediction_change': prediction_change,
                'original_logit': original_true_logit,
                'occluded_logit': occluded_true_logit
            })
        
        return word_importance
    
    def analyze_saliency(self, sample, tokens):
        """Comprehensive saliency analysis"""
        # Get baseline prediction
        with torch.no_grad():
            baseline_output = self.model(**sample)
            baseline_logits = baseline_output['logits']
            baseline_pred = baseline_logits.argmax(dim=-1).item()
            baseline_true_logit = baseline_logits[0, sample['label'].item()].item()
        
        # Compute word importance by occlusion
        word_importance = self.compute_word_importance_by_occlusion(sample, tokens)
        
        # Compute integrated gradients
        try:
            integrated_grads = self.compute_integrated_gradients(sample)
            integrated_grads_flat = integrated_grads.flatten()
        except Exception as e:
            print(f"Integrated gradients failed: {e}")
            integrated_grads_flat = np.zeros(len(tokens))
        
        # Compute gradients
        try:
            grad_results = self.compute_gradients(sample)
            gradients_flat = grad_results['gradients'].get('input_ids_sent', np.zeros(len(tokens))).flatten()
        except Exception as e:
            print(f"Gradients failed: {e}")
            gradients_flat = np.zeros(len(tokens))
        
        return {
            'baseline': {
                'logits': baseline_logits.cpu().numpy(),
                'prediction': baseline_pred,
                'true_logit': baseline_true_logit
            },
            'word_importance': word_importance,
            'integrated_gradients': integrated_grads_flat.tolist(),
            'gradients': gradients_flat.tolist()
        }

class SwitchMetricAnalyzer:
    """Fixed Switch metrics analyzer"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def mask_tokens_and_measure(self, sample, tokens_to_mask):
        """Mask specific tokens and measure logits and loss"""
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
            loss = self.ce_loss(logits, sample['label'])
        
        return {
            'logits': logits.cpu().numpy(),
            'loss': loss.item(),
            'prediction': logits.argmax(dim=-1).item()
        }
    
    def keep_only_tokens_and_measure(self, sample, tokens_to_keep):
        """Keep only specific tokens and measure logits and loss"""
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
            loss = self.ce_loss(logits, sample['label'])
        
        return {
            'logits': logits.cpu().numpy(),
            'loss': loss.item(),
            'prediction': logits.argmax(dim=-1).item()
        }
    
    def compute_switch_metrics(self, sample, noun_pos, true_label):
        """Compute proper Switch metrics for comprehensiveness and sufficiency"""
        # Get baseline prediction
        with torch.no_grad():
            baseline_output = self.model(**sample)
            baseline_logits = baseline_output['logits']
            baseline_pred = baseline_logits.argmax(dim=-1).item()
            baseline_loss = self.ce_loss(baseline_logits, sample['label']).item()
        
        # Get true class logit
        true_class = 1 if true_label == 'f' else 0
        baseline_true_logit = baseline_logits[0, true_class].item()
        
        # Define token sets to test
        noun_only = [noun_pos]
        noun_context = [max(0, noun_pos-1), noun_pos, min(sample['input_ids_sent'].shape[1]-1, noun_pos+1)]
        
        results = {
            'baseline': {
                'logits': baseline_logits.cpu().numpy(),
                'true_class_logit': baseline_true_logit,
                'loss': baseline_loss,
                'prediction': baseline_pred
            }
        }
        
        # Test comprehensiveness (removal) - Δ_logit = z_y(x) - z_y(x \ S)
        results['comprehensiveness'] = {}
        
        # Remove noun only
        noun_removal = self.mask_tokens_and_measure(sample, noun_only)
        noun_removal_true_logit = noun_removal['logits'][0, true_class]
        delta_logit_noun = baseline_true_logit - noun_removal_true_logit
        delta_ce_noun = noun_removal['loss'] - baseline_loss
        
        results['comprehensiveness']['noun_only'] = {
            'logits': noun_removal['logits'],
            'true_class_logit': noun_removal_true_logit,
            'loss': noun_removal['loss'],
            'delta_logit': delta_logit_noun,
            'delta_ce': delta_ce_noun,
            'prediction': noun_removal['prediction']
        }
        
        # Remove noun + context
        context_removal = self.mask_tokens_and_measure(sample, noun_context)
        context_removal_true_logit = context_removal['logits'][0, true_class]
        delta_logit_context = baseline_true_logit - context_removal_true_logit
        delta_ce_context = context_removal['loss'] - baseline_loss
        
        results['comprehensiveness']['noun_context'] = {
            'logits': context_removal['logits'],
            'true_class_logit': context_removal_true_logit,
            'loss': context_removal['loss'],
            'delta_logit': delta_logit_context,
            'delta_ce': delta_ce_context,
            'prediction': context_removal['prediction']
        }
        
        # Test sufficiency (keeping only) - z_y(S) or -l(S)
        results['sufficiency'] = {}
        
        # Keep noun only
        noun_only_result = self.keep_only_tokens_and_measure(sample, noun_only)
        noun_only_true_logit = noun_only_result['logits'][0, true_class]
        noun_only_loss = noun_only_result['loss']
        
        results['sufficiency']['noun_only'] = {
            'logits': noun_only_result['logits'],
            'true_class_logit': noun_only_true_logit,
            'loss': noun_only_loss,
            'neg_loss': -noun_only_loss,
            'prediction': noun_only_result['prediction']
        }
        
        # Keep noun + context
        context_only_result = self.keep_only_tokens_and_measure(sample, noun_context)
        context_only_true_logit = context_only_result['logits'][0, true_class]
        context_only_loss = context_only_result['loss']
        
        results['sufficiency']['noun_context'] = {
            'logits': context_only_result['logits'],
            'true_class_logit': context_only_true_logit,
            'loss': context_only_loss,
            'neg_loss': -context_only_loss,
            'prediction': context_only_result['prediction']
        }
        
        return results

def run_comprehensive_analysis(model, tokenizer, device, sample_df, results_dir):
    """Run comprehensive explainability analyses"""
    print("\n🔬 Starting Comprehensive Explainability Analysis...")
    
    # Initialize analyzers
    saliency_analyzer = SaliencyAnalyzer(model, tokenizer, device)
    switch_analyzer = SwitchMetricAnalyzer(model, tokenizer, device)
    
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
        
        # Add true label to sample for loss computation
        true_label = 1 if row['Genus_ok'] == 'f' else 0
        sample['label'] = torch.tensor([true_label], dtype=torch.long).to(device)
        
        # Get baseline prediction
        with torch.no_grad():
            baseline_output = model(**sample)
            baseline_logits = baseline_output['logits']
            baseline_pred = baseline_logits.argmax().item()
            baseline_prob = F.softmax(baseline_logits, dim=-1)[0, baseline_pred].item()
        
        # Run analyses
        try:
            # 1. Saliency Analysis
            saliency_results = saliency_analyzer.analyze_saliency(sample, sample['tokens'])
            
            # 2. Switch Metric Analysis
            switch_results = switch_analyzer.compute_switch_metrics(sample, int(row['token_index']), row['Genus_ok'])
            
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
                'saliency': saliency_results,
                'switch_metrics': switch_results
            }
            
            all_results.append(sample_result)
            
        except Exception as e:
            print(f"Error analyzing sample {i+1}: {e}")
            continue
    
    return all_results

def create_saliency_plots(all_results, results_dir):
    """Create saliency map plots"""
    print("\n📊 Creating saliency map plots...")
    
    # Create individual saliency maps for each sample
    for result in all_results[:5]:  # Show first 5 samples
        if 'saliency' not in result:
            continue
        
        saliency_data = result['saliency']
        tokens = clean_tokens(result['tokens'])
        
        # Create saliency visualization
        fig, axes = plt.subplots(2, 2, figsize=(48, 32))
        fig.suptitle(f'Saliency Analysis - Sample {result["sample_id"]}: "{result["sentence"]}"', 
                    fontsize=48, fontweight='bold')
        
        # 1. Word importance by occlusion
        ax1 = axes[0, 0]
        word_importance = saliency_data['word_importance']
        if word_importance:
            words = [w['token'] for w in word_importance]
            logit_changes = [w['logit_change'] for w in word_importance]
            
            bars = ax1.bar(range(len(words)), logit_changes, alpha=0.7, 
                          color='lightcoral', edgecolor='black', linewidth=2)
            ax1.set_title('Word Importance by Occlusion', fontsize=32, fontweight='bold')
            ax1.set_xlabel('Token Position', fontsize=28)
            ax1.set_ylabel('Logit Change', fontsize=28)
            ax1.set_xticks(range(len(words)))
            ax1.set_xticklabels(words, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3)
            
            # Highlight noun position
            noun_pos = result['noun_pos']
            if noun_pos < len(bars):
                bars[noun_pos].set_color('red')
                bars[noun_pos].set_edgecolor('darkred')
                bars[noun_pos].set_linewidth(4)
        
        # 2. Integrated gradients
        ax2 = axes[0, 1]
        integrated_grads = saliency_data['integrated_gradients']
        if integrated_grads and len(integrated_grads) >= len(tokens):
            grad_values = integrated_grads[:len(tokens)]
            
            bars = ax2.bar(range(len(tokens)), grad_values, alpha=0.7, 
                          color='lightblue', edgecolor='black', linewidth=2)
            ax2.set_title('Integrated Gradients', fontsize=32, fontweight='bold')
            ax2.set_xlabel('Token Position', fontsize=28)
            ax2.set_ylabel('Gradient Value', fontsize=28)
            ax2.set_xticks(range(len(tokens)))
            ax2.set_xticklabels(tokens, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
            
            # Highlight noun position
            noun_pos = result['noun_pos']
            if noun_pos < len(bars):
                bars[noun_pos].set_color('blue')
                bars[noun_pos].set_edgecolor('darkblue')
                bars[noun_pos].set_linewidth(4)
        
        # 3. Prediction changes
        ax3 = axes[1, 0]
        if word_importance:
            words = [w['token'] for w in word_importance]
            pred_changes = [w['prediction_change'] for w in word_importance]
            
            bars = ax3.bar(range(len(words)), pred_changes, alpha=0.7, 
                          color='lightgreen', edgecolor='black', linewidth=2)
            ax3.set_title('Prediction Changes by Occlusion', fontsize=32, fontweight='bold')
            ax3.set_xlabel('Token Position', fontsize=28)
            ax3.set_ylabel('Prediction Change', fontsize=28)
            ax3.set_xticks(range(len(words)))
            ax3.set_xticklabels(words, rotation=45, ha='right')
            ax3.grid(True, alpha=0.3)
            
            # Highlight noun position
            noun_pos = result['noun_pos']
            if noun_pos < len(bars):
                bars[noun_pos].set_color('green')
                bars[noun_pos].set_edgecolor('darkgreen')
                bars[noun_pos].set_linewidth(4)
        
        # 4. Summary statistics
        ax4 = axes[1, 1]
        summary_text = []
        summary_text.append(f"True Label: {result['true_label']}")
        summary_text.append(f"Predicted: {result['baseline_prediction']}")
        summary_text.append(f"Confidence: {result['baseline_confidence']:.4f}")
        summary_text.append(f"Noun Position: {result['noun_pos']}")
        
        if word_importance:
            max_change_word = max(word_importance, key=lambda x: abs(x['logit_change']))
            summary_text.append(f"Most Important: {max_change_word['token']}")
            summary_text.append(f"Max Logit Change: {max_change_word['logit_change']:.4f}")
        
        ax4.text(0.1, 0.5, '\n'.join(summary_text), fontsize=24, fontweight='bold',
                transform=ax4.transAxes, verticalalignment='center')
        ax4.set_title('Summary Statistics', fontsize=32, fontweight='bold')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{results_dir}/saliency_maps/saliency_sample_{result['sample_id']}.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    # Create aggregate saliency analysis
    create_aggregate_saliency_plots(all_results, results_dir)
    
    print("✅ Saliency map plots saved")

def create_aggregate_saliency_plots(all_results, results_dir):
    """Create aggregate saliency analysis plots"""
    print("\n📊 Creating aggregate saliency plots...")
    
    # Collect all word importance data
    all_word_importance = []
    all_integrated_grads = []
    
    for result in all_results:
        if 'saliency' not in result:
            continue
        
        saliency_data = result['saliency']
        tokens = clean_tokens(result['tokens'])
        
        # Word importance
        word_importance = saliency_data['word_importance']
        for w in word_importance:
            all_word_importance.append({
                'sample_id': result['sample_id'],
                'token': w['token'],
                'position': w['position'],
                'logit_change': w['logit_change'],
                'prediction_change': w['prediction_change'],
                'is_noun': w['position'] == result['noun_pos']
            })
        
        # Integrated gradients
        integrated_grads = saliency_data['integrated_gradients']
        if integrated_grads and len(integrated_grads) >= len(tokens):
            for i, grad_val in enumerate(integrated_grads[:len(tokens)]):
                all_integrated_grads.append({
                    'sample_id': result['sample_id'],
                    'token': tokens[i] if i < len(tokens) else f'token_{i}',
                    'position': i,
                    'gradient': grad_val,
                    'is_noun': i == result['noun_pos']
                })
    
    if not all_word_importance:
        print("⚠️ No word importance data available, skipping aggregate plots")
        return
    
    df_word_imp = pd.DataFrame(all_word_importance)
    df_integrated = pd.DataFrame(all_integrated_grads) if all_integrated_grads else pd.DataFrame()
    
    # Create aggregate visualization
    fig, axes = plt.subplots(2, 2, figsize=(48, 32))
    fig.suptitle('Aggregate Saliency Analysis', fontsize=48, fontweight='bold')
    
    # 1. Average logit change by token type
    ax1 = axes[0, 0]
    noun_changes = df_word_imp[df_word_imp['is_noun']]['logit_change']
    context_changes = df_word_imp[~df_word_imp['is_noun']]['logit_change']
    
    categories = ['Noun', 'Context']
    avg_changes = [noun_changes.mean(), context_changes.mean()]
    std_changes = [noun_changes.std(), context_changes.std()]
    
    bars = ax1.bar(categories, avg_changes, alpha=0.7, color=['red', 'blue'], 
                   edgecolor='black', linewidth=2, yerr=std_changes, capsize=10)
    ax1.set_title('Average Logit Change by Token Type', fontsize=32, fontweight='bold')
    ax1.set_xlabel('Token Type', fontsize=28)
    ax1.set_ylabel('Average Logit Change', fontsize=28)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, avg_changes):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=20, fontweight='bold')
    
    # 2. Prediction change frequency
    ax2 = axes[0, 1]
    pred_change_counts = df_word_imp['prediction_change'].value_counts()
    labels = ['No Change', 'Prediction Change']
    values = [pred_change_counts.get(0, 0), pred_change_counts.get(1, 0)]
    
    ax2.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, 
            colors=['lightgreen', 'lightcoral'], textprops={'fontsize': 20})
    ax2.set_title('Prediction Change Frequency', fontsize=32, fontweight='bold')
    
    # 3. Token importance distribution
    ax3 = axes[1, 0]
    ax3.hist(df_word_imp['logit_change'], bins=20, alpha=0.7, color='skyblue', 
             edgecolor='black', linewidth=2)
    ax3.set_title('Distribution of Logit Changes', fontsize=32, fontweight='bold')
    ax3.set_xlabel('Logit Change', fontsize=28)
    ax3.set_ylabel('Frequency', fontsize=28)
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary statistics
    ax4 = axes[1, 1]
    summary_stats = []
    summary_stats.append(f"Total Samples: {len(all_results)}")
    summary_stats.append(f"Total Tokens Analyzed: {len(df_word_imp)}")
    summary_stats.append(f"Noun Tokens: {len(df_word_imp[df_word_imp['is_noun']])}")
    summary_stats.append(f"Context Tokens: {len(df_word_imp[~df_word_imp['is_noun']])}")
    summary_stats.append(f"Avg Noun Logit Change: {noun_changes.mean():.4f}")
    summary_stats.append(f"Avg Context Logit Change: {context_changes.mean():.4f}")
    summary_stats.append(f"Prediction Changes: {pred_change_counts.get(1, 0)}")
    
    ax4.text(0.1, 0.5, '\n'.join(summary_stats), fontsize=24, fontweight='bold',
            transform=ax4.transAxes, verticalalignment='center')
    ax4.set_title('Summary Statistics', fontsize=32, fontweight='bold')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/plots/aggregate_saliency_analysis.png", 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✅ Aggregate saliency plots saved")

def create_switch_metric_plots(all_results, results_dir):
    """Create Switch metric plots"""
    print("\n📊 Creating Switch metric plots...")
    
    # Collect Switch metric data
    comp_data = []
    suff_data = []
    
    for result in all_results:
        if 'switch_metrics' not in result:
            continue
        switch_results = result['switch_metrics']
        
        # Comprehensiveness data
        comp_removal = switch_results.get('comprehensiveness', {})
        for test_name, test_result in comp_removal.items():
            comp_data.append({
                'sample_id': result['sample_id'],
                'test': test_name,
                'delta_logit': test_result['delta_logit'],
                'delta_ce': test_result['delta_ce']
            })
        
        # Sufficiency data
        suff_results = switch_results.get('sufficiency', {})
        for test_name, test_result in suff_results.items():
            suff_data.append({
                'sample_id': result['sample_id'],
                'test': test_name,
                'true_class_logit': test_result['true_class_logit'],
                'neg_loss': test_result['neg_loss']
            })
    
    if not comp_data and not suff_data:
        print("⚠️ No Switch metric data available, skipping plots")
        return
    
    df_comp = pd.DataFrame(comp_data) if comp_data else pd.DataFrame()
    df_suff = pd.DataFrame(suff_data) if suff_data else pd.DataFrame()
    
    # Create Switch metric visualization
    fig, axes = plt.subplots(2, 2, figsize=(48, 32))
    fig.suptitle('Switch Metrics: Comprehensiveness and Sufficiency Analysis', fontsize=48, fontweight='bold')
    
    # 1. Comprehensiveness - Δ_logit = z_y(x) - z_y(x \ S)
    if not df_comp.empty:
        ax1 = axes[0, 0]
        test_names = df_comp['test'].unique()
        avg_delta_logit = [df_comp[df_comp['test'] == test]['delta_logit'].mean() for test in test_names]
        
        bars = ax1.bar(test_names, avg_delta_logit, alpha=0.7, color='lightcoral', edgecolor='black', linewidth=2)
        ax1.set_title('Comprehensiveness: Δ_logit = z_y(x) - z_y(x \\ S)', fontsize=32, fontweight='bold')
        ax1.set_xlabel('Token Set Removed', fontsize=28)
        ax1.set_ylabel('Δ Logit', fontsize=28)
        ax1.tick_params(axis='x', rotation=45, labelsize=20)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, avg_delta_logit):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=20, fontweight='bold')
    
    # 2. Comprehensiveness - Δ_CE = l(x \ S) - l(x)
    if not df_comp.empty:
        ax2 = axes[0, 1]
        test_names = df_comp['test'].unique()
        avg_delta_ce = [df_comp[df_comp['test'] == test]['delta_ce'].mean() for test in test_names]
        
        bars = ax2.bar(test_names, avg_delta_ce, alpha=0.7, color='orange', edgecolor='black', linewidth=2)
        ax2.set_title('Comprehensiveness: Δ_CE = l(x \\ S) - l(x)', fontsize=32, fontweight='bold')
        ax2.set_xlabel('Token Set Removed', fontsize=28)
        ax2.set_ylabel('Δ Cross-Entropy', fontsize=28)
        ax2.tick_params(axis='x', rotation=45, labelsize=20)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, avg_delta_ce):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=20, fontweight='bold')
    
    # 3. Sufficiency - z_y(S)
    if not df_suff.empty:
        ax3 = axes[1, 0]
        test_names = df_suff['test'].unique()
        avg_true_logit = [df_suff[df_suff['test'] == test]['true_class_logit'].mean() for test in test_names]
        
        bars = ax3.bar(test_names, avg_true_logit, alpha=0.7, color='lightgreen', edgecolor='black', linewidth=2)
        ax3.set_title('Sufficiency: z_y(S)', fontsize=32, fontweight='bold')
        ax3.set_xlabel('Token Set Kept', fontsize=28)
        ax3.set_ylabel('True Class Logit', fontsize=28)
        ax3.tick_params(axis='x', rotation=45, labelsize=20)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, avg_true_logit):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=20, fontweight='bold')
    
    # 4. Sufficiency - -l(S)
    if not df_suff.empty:
        ax4 = axes[1, 1]
        test_names = df_suff['test'].unique()
        avg_neg_loss = [df_suff[df_suff['test'] == test]['neg_loss'].mean() for test in test_names]
        
        bars = ax4.bar(test_names, avg_neg_loss, alpha=0.7, color='lightblue', edgecolor='black', linewidth=2)
        ax4.set_title('Sufficiency: -l(S)', fontsize=32, fontweight='bold')
        ax4.set_xlabel('Token Set Kept', fontsize=28)
        ax4.set_ylabel('Negative Loss', fontsize=28)
        ax4.tick_params(axis='x', rotation=45, labelsize=20)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, avg_neg_loss):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=20, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/plots/switch_metrics_analysis.png", 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✅ Switch metric plots saved")

def save_comprehensive_data(all_results, results_dir):
    """Save comprehensive analysis data"""
    print("\n💾 Saving comprehensive analysis data...")
    
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
    
    with open(f"{results_dir}/data/comprehensive_analysis_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)
    
    # Save detailed results
    with open(f"{results_dir}/data/detailed_comprehensive_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("✅ Comprehensive analysis data saved")

def main():
    """Main function to run comprehensive explainability analysis"""
    print("🚀 Starting Comprehensive Exp2 Explainability Analysis")
    print("=" * 60)
    
    # Create results folder
    results_dir = create_results_folder()
    print(f"📁 Created results folder: {results_dir}")
    
    # Load model and data
    model, tokenizer, device = load_best_exp2_model()
    sample_df = load_and_prepare_data()
    
    # Run comprehensive analysis
    all_results = run_comprehensive_analysis(model, tokenizer, device, sample_df, results_dir)
    
    # Create visualizations
    create_saliency_plots(all_results, results_dir)
    create_switch_metric_plots(all_results, results_dir)
    
    # Save data
    save_comprehensive_data(all_results, results_dir)
    
    print("\n" + "=" * 60)
    print("✅ COMPREHENSIVE EXPLAINABILITY ANALYSIS COMPLETE!")
    print(f"📁 All results saved to: {results_dir}")
    print("📊 Generated visualizations:")
    print("   - Individual saliency maps (word-by-word importance)")
    print("   - Aggregate saliency analysis")
    print("   - Switch metrics analysis (Comprehensiveness & Sufficiency)")
    print("💾 Saved data files:")
    print("   - comprehensive_analysis_summary.json")
    print("   - detailed_comprehensive_results.json")
    print("=" * 60)

if __name__ == "__main__":
    main()
