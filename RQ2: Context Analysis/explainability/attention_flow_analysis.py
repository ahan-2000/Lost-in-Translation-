#!/usr/bin/env python3
"""
Attention Flow Analysis for Exp2 Model
Creates attention flow visualizations showing how words attend to each other.
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
    """Create dedicated folder for attention flow results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"/root/occ/attention_flow_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f"{results_dir}/plots", exist_ok=True)
    os.makedirs(f"{results_dir}/data", exist_ok=True)
    os.makedirs(f"{results_dir}/attention_flows", exist_ok=True)
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
    sample_size = min(15, len(valid_samples))
    sample_indices = np.random.choice(len(valid_samples), size=sample_size, replace=False)
    sample_df = valid_samples.iloc[sample_indices].reset_index(drop=True)
    
    print(f"\nSampled {len(sample_df)} sentences for attention flow analysis")
    return sample_df

class AttentionFlowAnalyzer:
    """Analyze attention flow patterns in the Exp2 model"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def extract_attention_weights(self, sample):
        """Extract attention weights from the model"""
        with torch.no_grad():
            output = self.model(**sample, return_attention=True)
            attention_weights = output['attention_weights'][0]  # (heads, seq_len)
            logits = output['logits']
        
        return {
            'attention_weights': attention_weights.cpu().numpy(),
            'logits': logits.cpu().numpy(),
            'prediction': logits.argmax(dim=-1).item()
        }
    
    def analyze_attention_patterns(self, sample, tokens, noun_pos):
        """Analyze attention patterns for the sample"""
        attention_data = self.extract_attention_weights(sample)
        attention_weights = attention_data['attention_weights']  # (heads, seq_len)
        
        # Clean tokens for display
        clean_tokens_list = clean_tokens(tokens)
        
        # Analyze each head
        head_analyses = []
        for head_idx in range(attention_weights.shape[0]):
            head_attn = attention_weights[head_idx]  # (seq_len,)
            
            # Find most attended positions
            top_indices = np.argsort(head_attn)[-5:]  # Top 5 most attended
            top_values = head_attn[top_indices]
            
            # Analyze attention to noun position
            noun_attention = head_attn[noun_pos] if noun_pos < len(head_attn) else 0
            
            head_analyses.append({
                'head': head_idx,
                'attention_weights': head_attn.tolist(),
                'top_indices': top_indices.tolist(),
                'top_values': top_values.tolist(),
                'noun_attention': noun_attention,
                'entropy': -np.sum(head_attn * np.log(head_attn + 1e-8))
            })
        
        # Compute attention flow patterns
        flow_patterns = self.compute_attention_flows(attention_weights, tokens, noun_pos)
        
        return {
            'tokens': tokens,
            'clean_tokens': clean_tokens_list,
            'noun_pos': noun_pos,
            'head_analyses': head_analyses,
            'flow_patterns': flow_patterns,
            'prediction': attention_data['prediction']
        }
    
    def compute_attention_flows(self, attention_weights, tokens, noun_pos):
        """Compute attention flow patterns"""
        flows = []
        
        # For each head, analyze attention patterns
        for head_idx in range(attention_weights.shape[0]):
            head_attn = attention_weights[head_idx]
            
            # Find attention from noun to other positions
            noun_to_others = []
            for i, attn_val in enumerate(head_attn):
                if i != noun_pos and attn_val > 0.01:  # Threshold for significant attention
                    noun_to_others.append({
                        'from_pos': noun_pos,
                        'to_pos': i,
                        'attention': attn_val,
                        'from_token': tokens[noun_pos] if noun_pos < len(tokens) else 'UNK',
                        'to_token': tokens[i] if i < len(tokens) else 'UNK'
                    })
            
            # Find attention to noun from other positions
            others_to_noun = []
            for i, attn_val in enumerate(head_attn):
                if i != noun_pos and attn_val > 0.01:
                    others_to_noun.append({
                        'from_pos': i,
                        'to_pos': noun_pos,
                        'attention': attn_val,
                        'from_token': tokens[i] if i < len(tokens) else 'UNK',
                        'to_token': tokens[noun_pos] if noun_pos < len(tokens) else 'UNK'
                    })
            
            flows.append({
                'head': head_idx,
                'noun_to_others': noun_to_others,
                'others_to_noun': others_to_noun
            })
        
        return flows

def create_attention_flow_visualization(attention_data, sample_info, results_dir):
    """Create attention flow visualization similar to the reference image"""
    
    tokens = attention_data['clean_tokens']
    noun_pos = attention_data['noun_pos']
    flow_patterns = attention_data['flow_patterns']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(48, 32))
    fig.suptitle(f'Attention Flow Analysis - Sample {sample_info["sample_id"]}: "{sample_info["sentence"]}"', 
                fontsize=48, fontweight='bold')
    
    # Select 4 different heads to show different patterns
    selected_heads = [0, 2, 4, 6] if len(flow_patterns) >= 7 else [0, 1, 2, 3]
    
    for plot_idx, head_idx in enumerate(selected_heads):
        if head_idx >= len(flow_patterns):
            continue
            
        ax = axes[plot_idx // 2, plot_idx % 2]
        flow_data = flow_patterns[head_idx]
        
        # Create attention flow diagram
        create_single_attention_flow(ax, flow_data, tokens, noun_pos, head_idx)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/attention_flows/attention_flow_sample_{sample_info['sample_id']}.png", 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_single_attention_flow(ax, flow_data, tokens, noun_pos, head_idx):
    """Create a single attention flow diagram"""
    
    # Set up the plot
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(tokens) + 2)
    ax.set_aspect('equal')
    
    # Draw tokens as boxes
    token_boxes = []
    for i, token in enumerate(tokens):
        # Position tokens vertically
        y_pos = len(tokens) - i
        x_pos = 2 if i == noun_pos else (1 if i < noun_pos else 3)
        
        # Draw token box
        if i == noun_pos:
            # Highlight noun in red
            box = plt.Rectangle((x_pos-0.4, y_pos-0.3), 0.8, 0.6, 
                              facecolor='red', edgecolor='darkred', linewidth=2)
            ax.add_patch(box)
            ax.text(x_pos, y_pos, token, ha='center', va='center', 
                   fontsize=16, fontweight='bold', color='white')
        else:
            # Regular token boxes
            box = plt.Rectangle((x_pos-0.4, y_pos-0.3), 0.8, 0.6, 
                              facecolor='lightblue', edgecolor='black', linewidth=1)
            ax.add_patch(box)
            ax.text(x_pos, y_pos, token, ha='center', va='center', 
                   fontsize=14, fontweight='normal')
        
        token_boxes.append((x_pos, y_pos))
    
    # Draw attention flows
    # From noun to others
    noun_pos_clean = min(noun_pos, len(token_boxes)-1)
    noun_x, noun_y = token_boxes[noun_pos_clean]
    
    for flow in flow_data['noun_to_others']:
        to_pos = min(flow['to_pos'], len(token_boxes)-1)
        to_x, to_y = token_boxes[to_pos]
        
        # Draw red line from noun to other token
        line_width = max(1, flow['attention'] * 10)  # Scale line width
        ax.plot([noun_x, to_x], [noun_y, to_y], 'r-', linewidth=line_width, alpha=0.7)
        
        # Add arrow
        ax.annotate('', xy=(to_x, to_y), xytext=(noun_x, noun_y),
                   arrowprops=dict(arrowstyle='->', color='red', lw=line_width, alpha=0.7))
    
    # From others to noun
    for flow in flow_data['others_to_noun']:
        from_pos = min(flow['from_pos'], len(token_boxes)-1)
        from_x, from_y = token_boxes[from_pos]
        
        # Draw blue line from other token to noun
        line_width = max(1, flow['attention'] * 10)  # Scale line width
        ax.plot([from_x, noun_x], [from_y, noun_y], 'b-', linewidth=line_width, alpha=0.7)
        
        # Add arrow
        ax.annotate('', xy=(noun_x, noun_y), xytext=(from_x, from_y),
                   arrowprops=dict(arrowstyle='->', color='blue', lw=line_width, alpha=0.7))
    
    # Set title and labels
    ax.set_title(f'Head {head_idx + 1} Attention Flow', fontsize=32, fontweight='bold')
    ax.set_xlabel('Token Position', fontsize=28)
    ax.set_ylabel('Attention Flow', fontsize=28)
    
    # Remove ticks and labels for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add legend
    ax.text(0.02, 0.98, 'Red: From Noun\nBlue: To Noun', 
           transform=ax.transAxes, fontsize=20, fontweight='bold',
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def create_attention_head_comparison(all_results, results_dir):
    """Create comparison of attention patterns across heads"""
    print("\n📊 Creating attention head comparison...")
    
    # Collect attention data
    head_data = []
    for result in all_results:
        if 'attention_analysis' not in result:
            continue
        
        attention_data = result['attention_analysis']
        for head_analysis in attention_data['head_analyses']:
            head_data.append({
                'sample_id': result['sample_id'],
                'head': head_analysis['head'],
                'noun_attention': head_analysis['noun_attention'],
                'entropy': head_analysis['entropy'],
                'max_attention': max(head_analysis['attention_weights']),
                'mean_attention': np.mean(head_analysis['attention_weights'])
            })
    
    if not head_data:
        print("⚠️ No attention data available, skipping head comparison")
        return
    
    df_heads = pd.DataFrame(head_data)
    
    # Create head comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(48, 32))
    fig.suptitle('Attention Head Comparison Analysis', fontsize=48, fontweight='bold')
    
    # 1. Average noun attention by head
    ax1 = axes[0, 0]
    avg_noun_attn = df_heads.groupby('head')['noun_attention'].mean()
    bars = ax1.bar(range(len(avg_noun_attn)), avg_noun_attn.values, alpha=0.7, 
                   color='red', edgecolor='black', linewidth=2)
    ax1.set_title('Average Attention to Noun by Head', fontsize=32, fontweight='bold')
    ax1.set_xlabel('Attention Head', fontsize=28)
    ax1.set_ylabel('Average Noun Attention', fontsize=28)
    ax1.set_xticks(range(len(avg_noun_attn)))
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, avg_noun_attn.values)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.3f}', ha='center', va='bottom', fontsize=20, fontweight='bold')
    
    # 2. Attention entropy by head
    ax2 = axes[0, 1]
    avg_entropy = df_heads.groupby('head')['entropy'].mean()
    bars = ax2.bar(range(len(avg_entropy)), avg_entropy.values, alpha=0.7, 
                   color='blue', edgecolor='black', linewidth=2)
    ax2.set_title('Average Attention Entropy by Head', fontsize=32, fontweight='bold')
    ax2.set_xlabel('Attention Head', fontsize=28)
    ax2.set_ylabel('Average Entropy', fontsize=28)
    ax2.set_xticks(range(len(avg_entropy)))
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, avg_entropy.values)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=20, fontweight='bold')
    
    # 3. Max attention by head
    ax3 = axes[1, 0]
    avg_max_attn = df_heads.groupby('head')['max_attention'].mean()
    bars = ax3.bar(range(len(avg_max_attn)), avg_max_attn.values, alpha=0.7, 
                   color='green', edgecolor='black', linewidth=2)
    ax3.set_title('Average Max Attention by Head', fontsize=32, fontweight='bold')
    ax3.set_xlabel('Attention Head', fontsize=28)
    ax3.set_ylabel('Average Max Attention', fontsize=28)
    ax3.set_xticks(range(len(avg_max_attn)))
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, avg_max_attn.values)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.3f}', ha='center', va='bottom', fontsize=20, fontweight='bold')
    
    # 4. Head specialization heatmap
    ax4 = axes[1, 1]
    pivot_noun_attn = df_heads.pivot_table(values='noun_attention', index='sample_id', columns='head', aggfunc='mean')
    im = ax4.imshow(pivot_noun_attn.values, cmap='Reds', aspect='auto')
    ax4.set_title('Noun Attention Heatmap by Sample and Head', fontsize=32, fontweight='bold')
    ax4.set_xlabel('Attention Head', fontsize=28)
    ax4.set_ylabel('Sample ID', fontsize=28)
    ax4.set_xticks(range(8))
    ax4.set_xticklabels([f'H{i+1}' for i in range(8)])
    
    cbar = plt.colorbar(im, ax=ax4, shrink=0.8)
    cbar.ax.tick_params(labelsize=20)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/plots/attention_head_comparison.png", 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✅ Attention head comparison plots saved")

def run_attention_flow_analysis(model, tokenizer, device, sample_df, results_dir):
    """Run attention flow analysis"""
    print("\n🔬 Starting Attention Flow Analysis...")
    
    # Initialize analyzer
    attention_analyzer = AttentionFlowAnalyzer(model, tokenizer, device)
    
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
        
        # Run attention analysis
        try:
            attention_results = attention_analyzer.analyze_attention_patterns(
                sample, sample['tokens'], int(row['token_index'])
            )
            
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
                'attention_analysis': attention_results
            }
            
            all_results.append(sample_result)
            
            # Create individual attention flow visualization
            create_attention_flow_visualization(attention_results, sample_result, results_dir)
            
        except Exception as e:
            print(f"Error analyzing sample {i+1}: {e}")
            continue
    
    return all_results

def save_attention_flow_data(all_results, results_dir):
    """Save attention flow analysis data"""
    print("\n💾 Saving attention flow analysis data...")
    
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
    
    with open(f"{results_dir}/data/attention_flow_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)
    
    # Save detailed results
    with open(f"{results_dir}/data/detailed_attention_flow_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("✅ Attention flow analysis data saved")

def main():
    """Main function to run attention flow analysis"""
    print("🚀 Starting Attention Flow Analysis for Exp2 Model")
    print("=" * 60)
    
    # Create results folder
    results_dir = create_results_folder()
    print(f"📁 Created results folder: {results_dir}")
    
    # Load model and data
    model, tokenizer, device = load_best_exp2_model()
    sample_df = load_and_prepare_data()
    
    # Run attention flow analysis
    all_results = run_attention_flow_analysis(model, tokenizer, device, sample_df, results_dir)
    
    # Create visualizations
    create_attention_head_comparison(all_results, results_dir)
    
    # Save data
    save_attention_flow_data(all_results, results_dir)
    
    print("\n" + "=" * 60)
    print("✅ ATTENTION FLOW ANALYSIS COMPLETE!")
    print(f"📁 All results saved to: {results_dir}")
    print("📊 Generated visualizations:")
    print("   - Individual attention flow diagrams")
    print("   - Attention head comparison analysis")
    print("💾 Saved data files:")
    print("   - attention_flow_summary.json")
    print("   - detailed_attention_flow_results.json")
    print("=" * 60)

if __name__ == "__main__":
    main()
