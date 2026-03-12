#!/usr/bin/env python3
"""
Comprehensive Exp2 Explainability Analysis
Analyzes the best Exp2 model weights with attention visualizations, 
gradient-based explanations, and detailed attention head analysis.
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
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')

# Import our custom modules
import sys
import os
import importlib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from models import ContextReaderModel
from word_context_analysis.explainability import explainability_utils
importlib.reload(explainability_utils)
from word_context_analysis.explainability.explainability_utils import Exp2Explainer, visualize_attention_heads

# Set style for VERY large, clear plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (40, 24)  # Much larger default size
plt.rcParams['font.size'] = 24  # Much larger font
plt.rcParams['axes.titlesize'] = 32  # Much larger titles
plt.rcParams['axes.labelsize'] = 28  # Much larger axis labels
plt.rcParams['xtick.labelsize'] = 20  # Much larger x-axis labels
plt.rcParams['ytick.labelsize'] = 20  # Much larger y-axis labels
plt.rcParams['legend.fontsize'] = 20  # Much larger legend
plt.rcParams['figure.titlesize'] = 36  # Much larger figure titles

def clean_tokens(tokens):
    """Remove special tokens and clean token names for better readability"""
    cleaned_tokens = []
    for token in tokens:
        # Remove common special tokens
        if token in ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]']:
            continue
        # Remove BERT subword prefixes
        if token.startswith('##'):
            token = token[2:]  # Remove ## prefix
        # Skip very short tokens that are likely artifacts
        if len(token.strip()) < 2:
            continue
        cleaned_tokens.append(token)
    return cleaned_tokens

def create_results_folder():
    """Create dedicated folder for explainability results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"/root/occ/explainability_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f"{results_dir}/plots", exist_ok=True)
    os.makedirs(f"{results_dir}/attention_analysis", exist_ok=True)
    os.makedirs(f"{results_dir}/gradient_analysis", exist_ok=True)
    os.makedirs(f"{results_dir}/head_analysis", exist_ok=True)
    os.makedirs(f"{results_dir}/data", exist_ok=True)
    return results_dir

def load_best_exp2_model():
    """Load the best Exp2 model weights"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the best model (fold 3 based on exp2_summary.json)
    model_path = "/root/occ/outputs/exp2_only_20251010_123308/Exp2_ContextReader_fold_3.pth"
    
    # Initialize model with same architecture as training
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
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    
    print("✅ Best Exp2 model loaded successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Number of attention heads: {model.heads}")
    
    return model, tokenizer, device

def load_and_prepare_data():
    """Load and prepare the dataset for analysis"""
    df = pd.read_parquet("/root/occ/new_occitan_exact_mentions.parquet")
    print(f"Dataset shape: {df.shape}")
    
    # Filter for valid samples (noun position >= 0 AND gender in [m,f])
    valid_samples = df[(df['token_index'] >= 0) & (df['Genus_ok'].isin(['m', 'f']))]
    print(f"Valid samples: {len(valid_samples)}/{len(df)}")
    
    # Sample diverse sentences for analysis
    np.random.seed(42)
    sample_size = min(50, len(valid_samples))
    sample_indices = np.random.choice(len(valid_samples), size=sample_size, replace=False)
    sample_df = valid_samples.iloc[sample_indices].reset_index(drop=True)
    
    print(f"\nSampled {len(sample_df)} sentences for analysis")
    print(f"Gender distribution in sample:")
    print(sample_df['Genus_ok'].value_counts())
    
    return sample_df

def analyze_attention_patterns(model, tokenizer, device, sample_df, results_dir):
    """Comprehensive attention pattern analysis"""
    print("\n🔍 Starting comprehensive attention analysis...")
    
    explainer = Exp2Explainer(model, tokenizer, device)
    
    # Analyze multiple samples
    attention_results = []
    for i in range(min(20, len(sample_df))):
        row = sample_df.iloc[i]
        
        print(f"Analyzing sample {i+1}/{min(20, len(sample_df))}")
        
        # Prepare sample
        sample = explainer.prepare_sample(
            sentence=row['sentence_raw'],
            noun_pos=int(row['token_index']),
            occitan_word=row['occitanForm'],
            latin_lemma=row['Lemma'],
            latin_gender=row['Genus_lat']
        )
        
        # Get attention analysis
        attention_results_sample = explainer.get_attention_analysis(sample)
        
        # Get head importance analysis
        head_importance_results = explainer.analyze_head_importance(sample)
        
        # Get prediction
        with torch.no_grad():
            output = model(**sample)
            logits = output['logits'][0]
            probs = torch.softmax(logits, dim=0)
            pred_class = logits.argmax().item()
        
        attention_results.append({
            'sample': sample,
            'attention': attention_results_sample,
            'head_analysis': head_importance_results,
            'tokens': sample['tokens'],
            'noun_pos': int(row['token_index']),
            'prediction': pred_class,
            'confidence': probs[pred_class].item(),
            'true_label': row['Genus_ok'],
            'sentence': row['sentence_raw'],
            'occitan_word': row['occitanForm'],
            'latin_lemma': row['Lemma'],
            'latin_gender': row['Genus_lat']
        })
    
    return attention_results

def create_comprehensive_attention_visualization(attention_results, results_dir):
    """Create comprehensive attention visualizations"""
    print("\n📊 Creating comprehensive attention visualizations...")
    
    for i, result in enumerate(attention_results[:10]):  # Analyze first 10 samples
        tokens = result['tokens']
        cleaned_tokens = clean_tokens(tokens)
        attention_data = result['attention']
        raw_attention = attention_data["raw_attention"]  # (heads, T)
        noun_pos = result['noun_pos']
        
        # Adjust attention weights to match cleaned tokens
        cleaned_attention = []
        cleaned_noun_pos = None
        token_idx = 0
        
        for j, token in enumerate(tokens):
            if token in ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]']:
                continue
            if token.startswith('##') or len(token.strip()) < 2:
                continue
            cleaned_attention.append(raw_attention[:, j])  # Keep attention for this token
            if j == noun_pos:
                cleaned_noun_pos = token_idx
            token_idx += 1
        
        if cleaned_noun_pos is None:
            cleaned_noun_pos = 0  # Fallback to first token
        
        cleaned_attention = np.array(cleaned_attention).T  # (heads, cleaned_T)
        
        # Create VERY large figure with subplots
        fig = plt.figure(figsize=(48, 32))  # Much larger figure
        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.4)  # More spacing
        
        # Main title
        fig.suptitle(f'Sample {i+1}: {result["sentence"][:80]}...\n'
                    f'Noun: {result["occitan_word"]} | Pred: {"F" if result["prediction"] == 1 else "M"} '
                    f'({result["confidence"]:.3f}) | True: {result["true_label"].upper()}', 
                    fontsize=20, y=0.95)
        
        # Plot attention for each head
        for head_idx in range(min(8, cleaned_attention.shape[0])):
            row = head_idx // 4
            col = head_idx % 4
            ax = fig.add_subplot(gs[row, col])
            
            # Create attention heatmap
            attention_matrix = cleaned_attention[head_idx].reshape(1, -1)
            
            # Create heatmap
            im = ax.imshow(attention_matrix, cmap='Blues', aspect='auto')
            
            ax.set_title(f'Head {head_idx + 1}', fontsize=28, fontweight='bold')
            ax.set_xlabel('Token Position', fontsize=24)
            ax.set_ylabel('Attention Weight', fontsize=24)
            
            # Set x-axis labels
            ax.set_xticks(range(len(cleaned_tokens)))
            ax.set_xticklabels(cleaned_tokens, rotation=45, ha='right', fontsize=18)
            ax.set_yticks([])
            
            # Highlight noun position
            ax.axvline(x=cleaned_noun_pos, color='red', linestyle='--', alpha=0.8, linewidth=2)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.ax.tick_params(labelsize=18)
        
        # Add attention statistics subplot
        ax_stats = fig.add_subplot(gs[2, :2])
        
        # Compute attention statistics
        stats = analyze_attention_patterns_single(attention_data, tokens, noun_pos)
        
        # Plot attention statistics
        heads = list(range(len(stats["head_entropy"])))
        ax_stats.bar(heads, stats["head_entropy"], alpha=0.7, color='skyblue', label='Entropy')
        ax_stats.set_title('Attention Head Statistics', fontsize=28, fontweight='bold')
        ax_stats.set_xlabel('Attention Head', fontsize=24)
        ax_stats.set_ylabel('Entropy', fontsize=24)
        ax_stats.set_xticks(heads)
        ax_stats.legend(fontsize=20)
        ax_stats.grid(True, alpha=0.3)
        
        # Add attention rollout subplot
        ax_rollout = fig.add_subplot(gs[2, 2:])
        
        # Clean rollout data to match cleaned tokens
        rollout = attention_data["attention_rollout"]
        cleaned_rollout = []
        for j, token in enumerate(tokens):
            if token in ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]']:
                continue
            if token.startswith('##') or len(token.strip()) < 2:
                continue
            cleaned_rollout.append(rollout[j])
        
        bars = ax_rollout.bar(range(len(cleaned_rollout)), cleaned_rollout, alpha=0.7, color='lightgreen')
        ax_rollout.set_title('Attention Rollout', fontsize=28, fontweight='bold')
        ax_rollout.set_xlabel('Token Position', fontsize=24)
        ax_rollout.set_ylabel('Rollout Weight', fontsize=24)
        ax_rollout.set_xticks(range(len(cleaned_tokens)))
        ax_rollout.set_xticklabels(cleaned_tokens, rotation=45, ha='right', fontsize=18)
        ax_rollout.axvline(x=cleaned_noun_pos, color='red', linestyle='--', alpha=0.8, linewidth=2)
        
        # Save plot
        plt.savefig(f"{results_dir}/plots/attention_sample_{i+1}.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    print(f"✅ Saved {min(10, len(attention_results))} attention visualizations")

def analyze_attention_patterns_single(attention_data, tokens, noun_pos):
    """Analyze patterns in attention weights for a single sample"""
    raw_attention = attention_data["raw_attention"]  # (heads, T)
    n_heads = raw_attention.shape[0]
    
    # Compute statistics
    stats = {
        "head_entropy": [],
        "head_max_attention": [],
        "head_noun_attention": [],
        "head_concentration": []
    }
    
    for head_idx in range(n_heads):
        head_attn = raw_attention[head_idx]
        
        # Entropy (measure of attention spread)
        entropy = -np.sum(head_attn * np.log(head_attn + 1e-8))
        stats["head_entropy"].append(entropy)
        
        # Maximum attention weight
        max_attn = np.max(head_attn)
        stats["head_max_attention"].append(max_attn)
        
        # Attention to noun position
        noun_attn = head_attn[noun_pos] if noun_pos < len(head_attn) else 0
        stats["head_noun_attention"].append(noun_attn)
        
        # Concentration (how focused the attention is)
        concentration = np.sum(head_attn**2)
        stats["head_concentration"].append(concentration)
    
    return stats

def create_head_importance_analysis(attention_results, results_dir):
    """Create comprehensive head importance analysis"""
    print("\n🎯 Creating head importance analysis...")
    
    # Collect head importance data from all samples
    all_head_importance = []
    all_head_stats = []
    
    for result in attention_results:
        head_importance = result['head_analysis']['head_importance']
        all_head_importance.append(head_importance)
        
        # Get attention statistics for this sample
        stats = analyze_attention_patterns_single(result['attention'], result['tokens'], result['noun_pos'])
        all_head_stats.append(stats)
    
    # Convert to numpy arrays
    all_head_importance = np.array(all_head_importance)  # (n_samples, n_heads)
    n_samples, n_heads = all_head_importance.shape
    
    # Create comprehensive head analysis visualization
    fig, axes = plt.subplots(2, 3, figsize=(48, 32))  # Much larger figure
    fig.suptitle('Comprehensive Head Importance Analysis', fontsize=48, fontweight='bold')
    
    # 1. Average head importance across samples
    ax1 = axes[0, 0]
    avg_importance = np.mean(all_head_importance, axis=0)
    std_importance = np.std(all_head_importance, axis=0)
    
    bars = ax1.bar(range(n_heads), avg_importance, yerr=std_importance, 
                   alpha=0.7, color='skyblue', capsize=5, edgecolor='black', linewidth=2)
    ax1.set_title('Average Head Importance Across Samples', fontsize=32, fontweight='bold')
    ax1.set_xlabel('Attention Head', fontsize=28)
    ax1.set_ylabel('Importance Score', fontsize=28)
    ax1.set_xticks(range(n_heads))
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, avg_importance)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_importance[i] + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=20, fontweight='bold')
    
    # 2. Head importance distribution
    ax2 = axes[0, 1]
    box_data = [all_head_importance[:, i] for i in range(n_heads)]
    bp = ax2.boxplot(box_data, labels=[f'Head {i+1}' for i in range(n_heads)], patch_artist=True)
    
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, n_heads))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_title('Head Importance Distribution', fontsize=32, fontweight='bold')
    ax2.set_xlabel('Attention Head', fontsize=28)
    ax2.set_ylabel('Importance Score', fontsize=28)
    ax2.tick_params(axis='x', rotation=45, labelsize=20)
    ax2.grid(True, alpha=0.3)
    
    # 3. Head attention entropy (average across samples)
    ax3 = axes[0, 2]
    avg_entropy = np.mean([stats['head_entropy'] for stats in all_head_stats], axis=0)
    
    bars = ax3.bar(range(n_heads), avg_entropy, alpha=0.7, color='lightgreen', edgecolor='black', linewidth=2)
    ax3.set_title('Average Attention Entropy by Head', fontsize=32, fontweight='bold')
    ax3.set_xlabel('Attention Head', fontsize=28)
    ax3.set_ylabel('Entropy', fontsize=28)
    ax3.set_xticks(range(n_heads))
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, avg_entropy)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=20, fontweight='bold')
    
    # 4. Head noun attention (average across samples)
    ax4 = axes[1, 0]
    avg_noun_attention = np.mean([stats['head_noun_attention'] for stats in all_head_stats], axis=0)
    
    bars = ax4.bar(range(n_heads), avg_noun_attention, alpha=0.7, color='lightcoral', edgecolor='black', linewidth=2)
    ax4.set_title('Average Attention to Noun Position', fontsize=32, fontweight='bold')
    ax4.set_xlabel('Attention Head', fontsize=28)
    ax4.set_ylabel('Noun Attention Weight', fontsize=28)
    ax4.set_xticks(range(n_heads))
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, avg_noun_attention)):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.3f}', ha='center', va='bottom', fontsize=20, fontweight='bold')
    
    # 5. Head concentration (average across samples)
    ax5 = axes[1, 1]
    avg_concentration = np.mean([stats['head_concentration'] for stats in all_head_stats], axis=0)
    
    bars = ax5.bar(range(n_heads), avg_concentration, alpha=0.7, color='gold', edgecolor='black', linewidth=2)
    ax5.set_title('Average Attention Concentration by Head', fontsize=32, fontweight='bold')
    ax5.set_xlabel('Attention Head', fontsize=28)
    ax5.set_ylabel('Concentration', fontsize=28)
    ax5.set_xticks(range(n_heads))
    ax5.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, avg_concentration)):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.3f}', ha='center', va='bottom', fontsize=20, fontweight='bold')
    
    # 6. Correlation heatmap
    ax6 = axes[1, 2]
    
    # Compute correlation between different head metrics
    metrics = {
        'Importance': avg_importance,
        'Entropy': avg_entropy,
        'Noun Attention': avg_noun_attention,
        'Concentration': avg_concentration
    }
    
    metric_names = list(metrics.keys())
    correlation_matrix = np.zeros((len(metric_names), len(metric_names)))
    
    for i, metric1 in enumerate(metric_names):
        for j, metric2 in enumerate(metric_names):
            correlation = np.corrcoef(metrics[metric1], metrics[metric2])[0, 1]
            correlation_matrix[i, j] = correlation
    
    # Plot correlation heatmap
    im = ax6.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax6.set_xticks(range(len(metric_names)))
    ax6.set_yticks(range(len(metric_names)))
    ax6.set_xticklabels(metric_names, rotation=45, ha='right', fontsize=20)
    ax6.set_yticklabels(metric_names, fontsize=20)
    ax6.set_title('Correlation Between Head Metrics', fontsize=32, fontweight='bold')
    
    # Add correlation values as text
    for i in range(len(metric_names)):
        for j in range(len(metric_names)):
            text = ax6.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=20, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax6, shrink=0.8)
    cbar.ax.tick_params(labelsize=20)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/plots/head_importance_analysis.png", 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Print head ranking
    print("\n=== HEAD RANKING BY IMPORTANCE ===")
    head_ranking = np.argsort(avg_importance)[::-1]
    
    print(f"{'Rank':<6} {'Head':<6} {'Importance':<12} {'Entropy':<10} {'Noun Attn':<12} {'Concentration':<12}")
    print("-" * 70)
    
    for rank, head_idx in enumerate(head_ranking):
        importance = avg_importance[head_idx]
        entropy = avg_entropy[head_idx]
        noun_attn = avg_noun_attention[head_idx]
        concentration = avg_concentration[head_idx]
        print(f"{rank+1:<6} {head_idx+1:<6} {importance:<12.4f} {entropy:<10.4f} {noun_attn:<12.4f} {concentration:<12.4f}")
    
    print(f"✅ Saved comprehensive head importance analysis")

def create_prediction_analysis(attention_results, results_dir):
    """Create prediction accuracy and confidence analysis"""
    print("\n📈 Creating prediction analysis...")
    
    # Extract prediction data
    predictions = [r['prediction'] for r in attention_results]
    confidences = [r['confidence'] for r in attention_results]
    true_labels = [1 if r['true_label'] == 'f' else 0 for r in attention_results]
    correct = [p == t for p, t in zip(predictions, true_labels)]
    
    # Create prediction analysis visualization
    fig, axes = plt.subplots(2, 2, figsize=(40, 32))  # Much larger figure
    fig.suptitle('Prediction Analysis', fontsize=48, fontweight='bold')
    
    # 1. Prediction accuracy
    ax1 = axes[0, 0]
    accuracy = np.mean(correct)
    colors = ['lightcoral', 'lightgreen']
    labels = ['Incorrect', 'Correct']
    sizes = [1-accuracy, accuracy]
    
    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                                      startangle=90, textprops={'fontsize': 24})
    ax1.set_title(f'Overall Accuracy: {accuracy:.3f}', fontsize=32, fontweight='bold')
    
    # 2. Confidence distribution
    ax2 = axes[0, 1]
    ax2.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black', linewidth=2)
    ax2.set_title('Confidence Distribution', fontsize=32, fontweight='bold')
    ax2.set_xlabel('Confidence Score', fontsize=28)
    ax2.set_ylabel('Frequency', fontsize=28)
    ax2.grid(True, alpha=0.3)
    
    # Add mean line
    mean_conf = np.mean(confidences)
    ax2.axvline(mean_conf, color='red', linestyle='--', linewidth=3, 
                label=f'Mean: {mean_conf:.3f}')
    ax2.legend(fontsize=20)
    
    # 3. Confidence by correctness
    ax3 = axes[1, 0]
    correct_conf = [c for c, corr in zip(confidences, correct) if corr]
    incorrect_conf = [c for c, corr in zip(confidences, correct) if not corr]
    
    ax3.hist(correct_conf, bins=15, alpha=0.7, color='lightgreen', label='Correct', edgecolor='black', linewidth=2)
    ax3.hist(incorrect_conf, bins=15, alpha=0.7, color='lightcoral', label='Incorrect', edgecolor='black', linewidth=2)
    ax3.set_title('Confidence by Prediction Correctness', fontsize=32, fontweight='bold')
    ax3.set_xlabel('Confidence Score', fontsize=28)
    ax3.set_ylabel('Frequency', fontsize=28)
    ax3.legend(fontsize=20)
    ax3.grid(True, alpha=0.3)
    
    # 4. Gender distribution
    ax4 = axes[1, 1]
    gender_counts = pd.Series([r['true_label'] for r in attention_results]).value_counts()
    colors = ['lightblue', 'lightpink']
    bars = ax4.bar(gender_counts.index, gender_counts.values, color=colors, edgecolor='black', linewidth=2)
    ax4.set_title('Gender Distribution in Sample', fontsize=32, fontweight='bold')
    ax4.set_xlabel('Gender', fontsize=28)
    ax4.set_ylabel('Count', fontsize=28)
    
    # Add value labels on bars
    for bar, val in zip(bars, gender_counts.values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(val), ha='center', va='bottom', fontsize=24, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/plots/prediction_analysis.png", 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Saved prediction analysis")
    print(f"Overall accuracy: {accuracy:.3f}")
    print(f"Mean confidence: {np.mean(confidences):.3f}")

def save_analysis_data(attention_results, results_dir):
    """Save all analysis data to JSON files"""
    print("\n💾 Saving analysis data...")
    
    # Save attention results summary
    summary_data = []
    for i, result in enumerate(attention_results):
        summary_data.append({
            'sample_id': i,
            'sentence': result['sentence'],
            'occitan_word': result['occitan_word'],
            'latin_lemma': result['latin_lemma'],
            'latin_gender': result['latin_gender'],
            'noun_position': result['noun_pos'],
            'prediction': 'F' if result['prediction'] == 1 else 'M',
            'confidence': result['confidence'],
            'true_label': result['true_label'],
            'correct': result['prediction'] == (1 if result['true_label'] == 'f' else 0)
        })
    
    with open(f"{results_dir}/data/analysis_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)
    
    # Save detailed attention data
    detailed_data = []
    for i, result in enumerate(attention_results):
        detailed_data.append({
            'sample_id': i,
            'tokens': result['tokens'],
            'attention_weights': result['attention']['raw_attention'].tolist(),
            'attention_rollout': result['attention']['attention_rollout'].tolist(),
            'attention_flow': result['attention']['attention_flow'].tolist(),
            'head_importance': result['head_analysis']['head_importance'].tolist()
        })
    
    with open(f"{results_dir}/data/detailed_attention_data.json", "w") as f:
        json.dump(detailed_data, f, indent=2)
    
    print(f"✅ Saved analysis data to {results_dir}/data/")

def main():
    """Main function to run comprehensive explainability analysis"""
    print("🚀 Starting Comprehensive Exp2 Explainability Analysis")
    print("=" * 60)
    
    # Create results folder
    results_dir = create_results_folder()
    print(f"📁 Created results folder: {results_dir}")
    
    # Load best model
    model, tokenizer, device = load_best_exp2_model()
    
    # Load and prepare data
    sample_df = load_and_prepare_data()
    
    # Run comprehensive analysis
    attention_results = analyze_attention_patterns(model, tokenizer, device, sample_df, results_dir)
    
    # Create visualizations
    create_comprehensive_attention_visualization(attention_results, results_dir)
    create_head_importance_analysis(attention_results, results_dir)
    create_prediction_analysis(attention_results, results_dir)
    
    # Save data
    save_analysis_data(attention_results, results_dir)
    
    print("\n" + "=" * 60)
    print("✅ COMPREHENSIVE EXPLAINABILITY ANALYSIS COMPLETE!")
    print(f"📁 All results saved to: {results_dir}")
    print("📊 Generated visualizations:")
    print("   - Individual attention heatmaps for 10 samples")
    print("   - Comprehensive head importance analysis")
    print("   - Prediction accuracy and confidence analysis")
    print("💾 Saved data files:")
    print("   - analysis_summary.json")
    print("   - detailed_attention_data.json")
    print("=" * 60)

if __name__ == "__main__":
    main()
