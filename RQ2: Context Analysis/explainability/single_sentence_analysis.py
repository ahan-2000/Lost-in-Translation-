#!/usr/bin/env python3
"""
Single Sentence Explainability Analysis for Exp2
Analyzes one Occitan sentence in detail with all explainability methods
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from models import ContextReaderModel
from transformers import AutoTokenizer
from word_context_analysis.explainability.explainability_utils import Exp2Explainer
import warnings
warnings.filterwarnings('ignore')

def analyze_single_sentence():
    """Analyze a single sentence with all explainability methods"""
    
    print("🔍 Single Sentence Explainability Analysis")
    print("=" * 50)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    model_path = "/root/occ/outputs/exp2_only_20251008_223808/Exp2_ContextReader_fold_1.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    explainer = Exp2Explainer(model, tokenizer, device)
    
    # Load data and select one interesting sentence
    df = pd.read_parquet("/root/occ/occitan_exact_mentions_cleaned.parquet")
    valid_samples = df[(df['token_index'] >= 0) & (df['Genus_ok'].isin(['m', 'f']))]
    
    # Use a predefined shorter sentence for clearer analysis
    sentence = "La casa es bella."
    noun = "casa"
    noun_pos = 1  # Position of "casa" in the sentence (0-indexed)
    occitan_word = "casa"
    latin_lemma = "casa"
    latin_gender = "f"  # feminine
    
    print(f"\n📝 SENTENCE ANALYSIS")
    print(f"Sentence: {sentence}")
    print(f"Noun: {noun} (position: {noun_pos})")
    print(f"Occitan form: {occitan_word}")
    print(f"Latin lemma: {latin_lemma}")
    print(f"True gender: {latin_gender}")
    
    # Prepare sample
    sample = explainer.prepare_sample(
        sentence=sentence,
        noun_pos=noun_pos,
        occitan_word=occitan_word,
        latin_lemma=latin_lemma,
        latin_gender=latin_gender
    )
    
    tokens = sample['tokens']
    noun_pos = sample['noun_pos'].item()
    
    print(f"\n🔤 TOKENS: {tokens}")
    print(f"🎯 NOUN POSITION: {noun_pos} (token: '{tokens[noun_pos]}')")
    
    # Get all explanations
    print(f"\n🧠 COMPUTING EXPLANATIONS...")
    
    # Gradient-based methods
    saliency_results = explainer.get_saliency_maps(sample)
    smooth_grad_results = explainer.get_smooth_grad(sample, n_samples=20)
    ig_results = explainer.get_integrated_gradients(sample, steps=30)
    
    # Attention-based methods
    attention_results = explainer.get_attention_analysis(sample)
    head_importance_results = explainer.analyze_head_importance(sample)
    
    # Get prediction
    with torch.no_grad():
        output = model(**sample)
        logits = output['logits'][0]
        probs = torch.softmax(logits, dim=0)
        pred_class = logits.argmax().item()
    
    print(f"\n🎯 PREDICTION")
    print(f"Predicted: {'Feminine' if pred_class == 1 else 'Masculine'}")
    print(f"Confidence: {probs[pred_class]:.3f}")
    print(f"True label: {'Feminine' if latin_gender == 'f' else 'Masculine'}")
    print(f"Correct: {'✓' if (pred_class == 1 and latin_gender == 'f') or (pred_class == 0 and latin_gender == 'm') else '✗'}")
    
    # Analyze results
    print(f"\n📊 DETAILED ANALYSIS")
    
    # 1. Gradient-based attributions
    print(f"\n1️⃣ GRADIENT-BASED ATTRIBUTIONS")
    saliency_scores = saliency_results['saliency'].sum(axis=-1)
    smooth_grad_scores = smooth_grad_results['smooth_grad'].sum(axis=-1)
    ig_scores = ig_results['integrated_gradients'].sum(axis=-1)
    
    print(f"Top tokens by Saliency:")
    top_saliency = np.argsort(np.abs(saliency_scores))[-5:]
    for i, idx in enumerate(reversed(top_saliency)):
        token = tokens[idx]
        score = saliency_scores[idx]
        is_noun = " (NOUN)" if idx == noun_pos else ""
        print(f"  {i+1}. {token:15} | {score:8.4f}{is_noun}")
    
    print(f"Top tokens by SmoothGrad:")
    top_smooth = np.argsort(np.abs(smooth_grad_scores))[-5:]
    for i, idx in enumerate(reversed(top_smooth)):
        token = tokens[idx]
        score = smooth_grad_scores[idx]
        is_noun = " (NOUN)" if idx == noun_pos else ""
        print(f"  {i+1}. {token:15} | {score:8.4f}{is_noun}")
    
    print(f"Top tokens by Integrated Gradients:")
    top_ig = np.argsort(np.abs(ig_scores))[-5:]
    for i, idx in enumerate(reversed(top_ig)):
        token = tokens[idx]
        score = ig_scores[idx]
        is_noun = " (NOUN)" if idx == noun_pos else ""
        print(f"  {i+1}. {token:15} | {score:8.4f}{is_noun}")
    
    # 2. Attention analysis
    print(f"\n2️⃣ ATTENTION ANALYSIS")
    raw_attention = attention_results['raw_attention']  # (heads, seq_len)
    attention_rollout = attention_results['attention_rollout']
    attention_flow = attention_results['attention_flow']
    
    print(f"Average attention across heads:")
    avg_attention = raw_attention.mean(axis=0)
    top_attention = np.argsort(avg_attention)[-5:]
    for i, idx in enumerate(reversed(top_attention)):
        token = tokens[idx]
        score = avg_attention[idx]
        is_noun = " (NOUN)" if idx == noun_pos else ""
        print(f"  {i+1}. {token:15} | {score:8.4f}{is_noun}")
    
    print(f"Attention rollout:")
    top_rollout = np.argsort(attention_rollout)[-5:]
    for i, idx in enumerate(reversed(top_rollout)):
        token = tokens[idx]
        score = attention_rollout[idx]
        is_noun = " (NOUN)" if idx == noun_pos else ""
        print(f"  {i+1}. {token:15} | {score:8.4f}{is_noun}")
    
    # 3. Head importance
    print(f"\n3️⃣ HEAD IMPORTANCE")
    head_importance = head_importance_results['head_importance']
    print(f"Head importance scores:")
    for i, importance in enumerate(head_importance):
        print(f"  Head {i}: {importance:.4f}")
    
    # 4. Head-by-head attention
    print(f"\n4️⃣ ATTENTION BY HEAD")
    print(f"Attention to noun position by head:")
    for head_idx in range(raw_attention.shape[0]):
        noun_attention = raw_attention[head_idx, noun_pos]
        print(f"  Head {head_idx}: {noun_attention:.4f}")
    
    # Create visualizations
    create_visualizations(tokens, noun_pos, saliency_scores, smooth_grad_scores, ig_scores, 
                         raw_attention, attention_rollout, attention_flow, head_importance)
    
    return {
        'tokens': tokens,
        'noun_pos': noun_pos,
        'saliency': saliency_scores,
        'smooth_grad': smooth_grad_scores,
        'integrated_gradients': ig_scores,
        'attention': raw_attention,
        'attention_rollout': attention_rollout,
        'attention_flow': attention_flow,
        'head_importance': head_importance,
        'prediction': pred_class,
        'confidence': probs[pred_class].item(),
        'true_label': latin_gender
    }

def create_visualizations(tokens, noun_pos, saliency_scores, smooth_grad_scores, ig_scores,
                         raw_attention, attention_rollout, attention_flow, head_importance):
    """Create comprehensive visualizations for the single sentence"""
    
    print(f"\n📈 CREATING VISUALIZATIONS...")
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Filter out special tokens and get only meaningful words
    def filter_meaningful_tokens(tokens, scores):
        """Remove special tokens and return only meaningful words with their scores"""
        meaningful_tokens = []
        meaningful_scores = []
        meaningful_indices = []
        
        for i, token in enumerate(tokens):
            if token not in ['[CLS]', '[SEP]', '[PAD]']:
                meaningful_tokens.append(token)
                meaningful_scores.append(scores[i])
                meaningful_indices.append(i)
        
        return meaningful_tokens, meaningful_scores, meaningful_indices
    
    # Filter all data to remove special tokens
    meaningful_tokens, meaningful_saliency, saliency_indices = filter_meaningful_tokens(tokens, saliency_scores)
    _, meaningful_smooth_grad, _ = filter_meaningful_tokens(tokens, smooth_grad_scores)
    _, meaningful_ig, _ = filter_meaningful_tokens(tokens, ig_scores)
    _, meaningful_attention, _ = filter_meaningful_tokens(tokens, raw_attention.mean(axis=0))
    _, meaningful_rollout, _ = filter_meaningful_tokens(tokens, attention_rollout)
    _, meaningful_flow, _ = filter_meaningful_tokens(tokens, attention_flow)
    
    # Find the noun position in the filtered tokens
    filtered_noun_pos = None
    for i, idx in enumerate(saliency_indices):
        if idx == noun_pos:
            filtered_noun_pos = i
            break
    
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # 1. Gradient-based methods (top row)
    gradient_methods = [
        ('Saliency', meaningful_saliency, 'Reds'),
        ('SmoothGrad', meaningful_smooth_grad, 'Blues'),
        ('Integrated Gradients', meaningful_ig, 'Greens')
    ]
    
    for i, (method_name, scores, cmap) in enumerate(gradient_methods):
        ax = fig.add_subplot(gs[0, i])
        
        # Create bar plot with simple colors
        colors = ['red', 'blue', 'green'][i]
        bars = ax.bar(range(len(scores)), scores, alpha=0.7, color=colors)
        
        ax.set_title(f'{method_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Words in Sentence')
        ax.set_ylabel('Attribution Score')
        ax.set_xticks(range(len(meaningful_tokens)))
        ax.set_xticklabels(meaningful_tokens, rotation=45, ha='right', fontsize=10)
        
        # Highlight noun position
        if filtered_noun_pos is not None:
            ax.axvline(x=filtered_noun_pos, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Target Noun')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 2. Attention methods (second row)
    attention_methods = [
        ('Average Attention', meaningful_attention, 'purple'),
        ('Attention Rollout', meaningful_rollout, 'orange'),
        ('Attention Flow', meaningful_flow, 'pink')
    ]
    
    for i, (method_name, scores, color) in enumerate(attention_methods):
        ax = fig.add_subplot(gs[1, i])
        
        bars = ax.bar(range(len(scores)), scores, alpha=0.7, color=color)
        
        ax.set_title(f'{method_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Words in Sentence')
        ax.set_ylabel('Attention Weight')
        ax.set_xticks(range(len(meaningful_tokens)))
        ax.set_xticklabels(meaningful_tokens, rotation=45, ha='right', fontsize=10)
        
        # Highlight noun position
        if filtered_noun_pos is not None:
            ax.axvline(x=filtered_noun_pos, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Target Noun')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 3. Head importance (third row, left)
    ax = fig.add_subplot(gs[2, 0])
    bars = ax.bar(range(len(head_importance)), head_importance, alpha=0.7, color='gold')
    ax.set_title('Head Importance', fontsize=14, fontweight='bold')
    ax.set_xlabel('Attention Head')
    ax.set_ylabel('Importance Score')
    ax.set_xticks(range(len(head_importance)))
    ax.grid(True, alpha=0.3)
    
    # 4. Attention heatmap (third row, center and right)
    ax = fig.add_subplot(gs[2, 1:])
    
    # Filter attention matrix to remove special tokens
    filtered_attention = raw_attention[:, saliency_indices]
    im = ax.imshow(filtered_attention, cmap='Blues', aspect='auto')
    ax.set_title('Attention Heatmap Across All Heads', fontsize=14, fontweight='bold')
    ax.set_xlabel('Words in Sentence')
    ax.set_ylabel('Attention Head')
    ax.set_xticks(range(len(meaningful_tokens)))
    ax.set_xticklabels(meaningful_tokens, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(filtered_attention.shape[0]))
    ax.set_yticklabels([f'Head {i}' for i in range(filtered_attention.shape[0])])
    
    # Highlight noun position
    if filtered_noun_pos is not None:
        ax.axvline(x=filtered_noun_pos, color='red', linestyle='--', alpha=0.8, linewidth=2)
    plt.colorbar(im, ax=ax)
    
    # 5. Method comparison (bottom row)
    ax = fig.add_subplot(gs[3, :])
    
    # Compare top tokens across methods
    methods_data = []
    method_names = []
    
    for method_name, scores, _ in gradient_methods:
        top_idx = np.argmax(np.abs(scores))
        methods_data.append(scores[top_idx])
        method_names.append(method_name)
    
    bars = ax.bar(method_names, methods_data, alpha=0.7, color=['red', 'blue', 'green'])
    ax.set_title('Method Comparison (Max Attribution Score)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Attribution Score')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, methods_data):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Single Sentence Explainability Analysis', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()
    
    # Create a summary table
    create_summary_table(tokens, noun_pos, saliency_scores, smooth_grad_scores, ig_scores, 
                        raw_attention.mean(axis=0), attention_rollout, attention_flow)

def create_summary_table(tokens, noun_pos, saliency_scores, smooth_grad_scores, ig_scores,
                        avg_attention, attention_rollout, attention_flow):
    """Create a summary table of all methods"""
    
    print(f"\n📋 SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Token':<15} {'Saliency':<10} {'SmoothGrad':<12} {'Int.Grad':<10} {'Avg.Attn':<10} {'Rollout':<10} {'Flow':<10} {'Noun':<6}")
    print("-" * 80)
    
    for i, token in enumerate(tokens):
        is_noun = "YES" if i == noun_pos else "NO"
        print(f"{token:<15} {saliency_scores[i]:<10.4f} {smooth_grad_scores[i]:<12.4f} "
              f"{ig_scores[i]:<10.4f} {avg_attention[i]:<10.4f} {attention_rollout[i]:<10.4f} "
              f"{attention_flow[i]:<10.4f} {is_noun:<6}")
    
    print("=" * 80)
    
    # Find most important tokens across methods
    print(f"\n🏆 TOP INFLUENTIAL TOKENS")
    
    # Combine all methods (normalized)
    all_scores = np.array([
        np.abs(saliency_scores),
        np.abs(smooth_grad_scores), 
        np.abs(ig_scores),
        avg_attention,
        attention_rollout,
        attention_flow
    ])
    
    # Average across methods
    combined_scores = all_scores.mean(axis=0)
    top_tokens = np.argsort(combined_scores)[-5:]
    
    print("Ranked by average importance across all methods:")
    for i, idx in enumerate(reversed(top_tokens)):
        token = tokens[idx]
        score = combined_scores[idx]
        is_noun = " (NOUN)" if idx == noun_pos else ""
        print(f"  {i+1}. {token:15} | {score:.4f}{is_noun}")

if __name__ == "__main__":
    results = analyze_single_sentence()
    print(f"\n✅ Analysis complete!")
    print(f"📊 Generated comprehensive visualizations and summary table")
    print(f"🎯 Focused analysis of one sentence with all explainability methods")
