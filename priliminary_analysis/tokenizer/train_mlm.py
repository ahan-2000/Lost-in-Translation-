#!/usr/bin/env python3
"""
MLM Fine-Tuning Script with WandB Logging
==========================================
This script:
1. Fine-tunes mBERT on Occitan text using MLM
2. Compares traditional mBERT tokenizer vs hybrid (mBERT + BPE) tokenizer
3. Evaluates perplexity on original mBERT and both fine-tuned models
4. Logs training curves to WandB for 10 epochs
"""

import os
import random
import json
import logging
from glob import glob
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Any
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from tqdm.auto import tqdm
import wandb
from transformers import (
    BertTokenizerFast,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling
)
import sentencepiece as spm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Paths and Hyperparameters
CORPUS_DIR = "../data"  # Data folder (relative to preliminary_analysis)
BPE_MODEL = "occitan_bpe_600.model"  # BPE model file (in same folder)
OUTPUT_DIR = Path("mlm_finetune_results")  # Output in same folder
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32  # Increased from 8 for faster training (GPU has 46GB, can handle larger batches)
LEARNING_RATE = 5e-5
EPOCHS = 10  # 10 epochs as requested
MLM_PROB = 0.15
MAX_SEQ_LENGTH = 512
VALIDATION_SPLIT = 0.1  # 10% for validation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Log detailed GPU information if available
if torch.cuda.is_available():
    logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    logger.info(f"Using device: {DEVICE} (GPU)")
else:
    logger.warning("No GPU detected. Training will be slower on CPU.")
    logger.info(f"Using device: {DEVICE} (CPU)")


def train_bpe_model(corpus_dir: str, output_model: str, vocab_size: int = 600) -> None:
    """
    Train a BPE SentencePiece model from text files.
    
    Args:
        corpus_dir: Directory containing text files
        output_model: Path to save the trained BPE model
        vocab_size: Vocabulary size for the BPE model (default: 600)
    
    Raises:
        ValueError: If no text files are found in corpus_dir
        OSError: If model training fails
    """
    logger.info(f"Training BPE model with vocab_size={vocab_size}...")
    
    # Collect all text files
    files = glob(os.path.join(corpus_dir, "*.txt"))
    if not files:
        raise ValueError(f"No text files found in {corpus_dir}")
    
    logger.info(f"Found {len(files)} text files")
    
    # Create a temporary input file for SentencePiece training
    temp_input = "temp_corpus.txt"
    try:
        with open(temp_input, 'w', encoding='utf-8') as f:
            for file_path in tqdm(files, desc="Collecting text"):
                try:
                    text = Path(file_path).read_text(encoding='utf-8')
                    f.write(text)
                except Exception as e:
                    logger.warning(f"Could not read {file_path}: {e}")
        
        # Train SentencePiece model
        # Note: <unk>, <s>, </s> are control symbols, only <mask> needs to be user-defined
        spm.SentencePieceTrainer.train(
            input=temp_input,
            model_prefix=output_model.replace('.model', ''),
            vocab_size=vocab_size,
            model_type='bpe',
            character_coverage=0.9995,
            user_defined_symbols=['<mask>']
        )
        
        logger.info(f"BPE model saved to {output_model}")
    finally:
        # Clean up temp file
        if os.path.exists(temp_input):
            os.remove(temp_input)
            logger.debug(f"Removed temporary file: {temp_input}")


def create_hybrid_tokenizer(bpe_model_path: str) -> Tuple[BertTokenizerFast, int]:
    """
    Create hybrid tokenizer: mBERT + BPE pieces.
    
    Args:
        bpe_model_path: Path to the trained BPE model file
    
    Returns:
        Tuple of (hybrid_tokenizer, number_of_added_tokens)
    
    Raises:
        FileNotFoundError: If BPE model file doesn't exist
    """
    # Load base mBERT tokenizer
    base_tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")
    
    # Load BPE model
    if not os.path.exists(bpe_model_path):
        raise FileNotFoundError(f"BPE model not found: {bpe_model_path}")
    
    sp = spm.SentencePieceProcessor(model_file=bpe_model_path)
    
    # Collect BPE pieces not in mBERT vocab
    occ_pieces = {sp.id_to_piece(i) for i in range(sp.get_piece_size())}
    new_pieces = [
        p for p in occ_pieces 
        if p not in base_tokenizer.get_vocab() and not p.startswith("<")
    ]
    
    # Create hybrid tokenizer
    hybrid_tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")
    added = hybrid_tokenizer.add_tokens(new_pieces)
    logger.info(f"Added {added} Occitan BPE pieces to hybrid tokenizer")
    logger.info(f"Hybrid vocab size: {len(hybrid_tokenizer)}")
    
    return hybrid_tokenizer, added


class OccitanTextDataset(Dataset):
    """Dataset for Occitan text files."""
    
    def __init__(self, files: List[str], tokenizer: BertTokenizerFast) -> None:
        """
        Initialize dataset from text files.
        
        Args:
            files: List of file paths to load
            tokenizer: Tokenizer to use for encoding text
        """
        self.tokenizer = tokenizer
        self.examples: List[torch.Tensor] = []
        
        for fp in tqdm(files, desc="Loading dataset"):
            try:
                text = Path(fp).read_text(encoding="utf-8")
                for line in text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    enc = tokenizer(
                        line,
                        truncation=True,
                        max_length=MAX_SEQ_LENGTH,
                        return_special_tokens_mask=False,
                        padding=False
                    )
                    self.examples.append(torch.tensor(enc['input_ids'], dtype=torch.long))
            except Exception as e:
                logger.warning(f"Error processing {fp}: {e}")
        
        if not self.examples:
            raise ValueError("No valid examples loaded from files")
        
        logger.info(f"Loaded {len(self.examples)} training examples.")
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.examples[idx]


def compute_perplexity(
    model: AutoModelForMaskedLM,
    tokenizer: BertTokenizerFast,
    data_loader: DataLoader,
    device: torch.device,
    max_samples: Optional[int] = None
) -> float:
    """
    Compute perplexity on a dataset.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer used (for type checking, not used in computation)
        data_loader: DataLoader for the evaluation dataset
        device: Device to run computation on
        max_samples: Maximum number of samples to evaluate (None for all)
    
    Returns:
        Perplexity score (float('inf') if no valid tokens found)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Computing perplexity"):
            if max_samples and n_samples >= max_samples:
                break
            
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            
            # Count non-padding and non-masked tokens (only count labels != -100)
            if 'labels' in batch:
                mask = (batch['labels'] != -100)
                n_tokens = mask.sum().item()
            else:
                # Fallback: count all tokens
                n_tokens = batch['input_ids'].numel()
            
            if n_tokens > 0:
                total_loss += loss.item() * n_tokens
                total_tokens += n_tokens
                n_samples += 1
    
    if total_tokens == 0:
        logger.warning("No valid tokens found for perplexity computation")
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return perplexity


def finetune_with_wandb(
    train_loader: DataLoader,
    val_loader: DataLoader,
    tokenizer: BertTokenizerFast,
    model_name: str,
    wandb_project: str = "mlm-finetuning"
) -> Tuple[AutoModelForMaskedLM, Dict[str, Any]]:
    """
    Fine-tune model with WandB logging.
    
    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        tokenizer: Tokenizer to use
        model_name: Name identifier for the model (e.g., "traditional", "hybrid")
        wandb_project: WandB project name
    
    Returns:
        Tuple of (trained_model, results_dict)
    """
    # Initialize WandB
    run = wandb.init(
        project=wandb_project,
        name=f"mlm-{model_name}",
        config={
            "model_name": model_name,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "mlm_prob": MLM_PROB,
            "max_seq_length": MAX_SEQ_LENGTH,
            "validation_split": VALIDATION_SPLIT,
        },
        reinit=True
    )
    
    # Load base mBERT model
    model = AutoModelForMaskedLM.from_pretrained("bert-base-multilingual-cased").to(DEVICE)
    
    # Resize embeddings if hybrid tokenizer
    if model_name == "hybrid":
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Resized model embeddings to {len(tokenizer)}")
    
    # Log GPU memory usage if available
    if torch.cuda.is_available():
        logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # Add learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=EPOCHS
    )
    
    # Use mixed precision training for faster GPU training (if available)
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        logger.info("Using mixed precision training (AMP) for faster GPU training")
    
    # Training loop
    for epoch in range(1, EPOCHS + 1):
        # Training phase
        model.train()
        total_train_loss = 0.0
        train_steps = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            
            # Use mixed precision training if available
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
                    loss = outputs.loss
                
                scaler.scale(loss).backward()
                # Gradient clipping for training stability
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                # Gradient clipping for training stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            optimizer.zero_grad()
            
            total_train_loss += loss.item()
            train_steps += 1
            
            # Log batch loss (use step for continuous plotting)
            step = train_steps + (epoch - 1) * len(train_loader)
            wandb.log({
                f"{model_name}/train/batch_loss": loss.item(),
                f"{model_name}/train/epoch": epoch,
                f"{model_name}/train/learning_rate": scheduler.get_last_lr()[0],
            }, step=step)
        
        # Update learning rate
        scheduler.step()
        
        avg_train_loss = total_train_loss / train_steps
        
        # Validation phase
        model.eval()
        total_val_loss = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]"):
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                
                total_val_loss += loss.item()
                val_steps += 1
        
        avg_val_loss = total_val_loss / val_steps if val_steps > 0 else 0.0
        
        # Log epoch metrics (use global step for proper plotting)
        global_step = epoch * len(train_loader)
        wandb.log({
            f"{model_name}/train/epoch_loss": avg_train_loss,
            f"{model_name}/val/epoch_loss": avg_val_loss,
            f"{model_name}/epoch": epoch,
            "global_step": global_step
        }, step=epoch)  # Use epoch as step for cleaner plotting
        
        logger.info(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    
    # Save model
    out_dir = OUTPUT_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        logger.info(f"Saved model to {out_dir}")
    except Exception as e:
        logger.error(f"Failed to save model to {out_dir}: {e}")
        raise
    
    wandb.finish()
    
    return model, {
        'model_name': model_name,
        'model': model,
        'tokenizer': tokenizer,
        'output_dir': out_dir
    }


def main() -> None:
    """Main function to run MLM fine-tuning pipeline."""
    # Step 1: Check/create BPE model
    if not os.path.exists(BPE_MODEL):
        logger.info("BPE model not found. Training new BPE model...")
        train_bpe_model(CORPUS_DIR, BPE_MODEL, vocab_size=600)
    else:
        logger.info(f"Using existing BPE model: {BPE_MODEL}")
    
    # Step 2: Prepare data
    file_list = glob(os.path.join(CORPUS_DIR, "*.txt"))
    logger.info(f"Found {len(file_list)} text files in {CORPUS_DIR}")
    
    if len(file_list) == 0:
        raise ValueError(f"No text files found in {CORPUS_DIR}")
    
    # Step 3: Create tokenizers
    logger.info("=== Creating Tokenizers ===")
    trad_tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")
    logger.info(f"Traditional tokenizer vocab size: {len(trad_tokenizer)}")
    
    hybrid_tokenizer, n_added = create_hybrid_tokenizer(BPE_MODEL)
    
    # Step 4: Create datasets and splits
    logger.info("=== Preparing Datasets ===")
    trad_dataset = OccitanTextDataset(file_list, trad_tokenizer)
    hybrid_dataset = OccitanTextDataset(file_list, hybrid_tokenizer)
    
    # Split datasets
    def split_dataset(dataset: Dataset) -> Tuple[Dataset, Dataset]:
        """Split dataset into train and validation sets."""
        val_size = int(len(dataset) * VALIDATION_SPLIT)
        train_size = len(dataset) - val_size
        return random_split(
            dataset, 
            [train_size, val_size], 
            generator=torch.Generator().manual_seed(SEED)
        )
    
    trad_train, trad_val = split_dataset(trad_dataset)
    hybrid_train, hybrid_val = split_dataset(hybrid_dataset)
    
    logger.info(f"Train/Val sizes (traditional): {len(trad_train)}/{len(trad_val)}")
    logger.info(f"Train/Val sizes (hybrid): {len(hybrid_train)}/{len(hybrid_val)}")
    
    # Step 5: Create data loaders
    trad_collator = DataCollatorForLanguageModeling(
        tokenizer=trad_tokenizer, mlm=True, mlm_probability=MLM_PROB
    )
    hybrid_collator = DataCollatorForLanguageModeling(
        tokenizer=hybrid_tokenizer, mlm=True, mlm_probability=MLM_PROB
    )
    
    # Use multiple workers for faster data loading
    num_workers = min(4, os.cpu_count() or 1)  # Use up to 4 workers
    
    trad_train_loader = DataLoader(
        trad_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=trad_collator,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    trad_val_loader = DataLoader(
        trad_val, batch_size=BATCH_SIZE, shuffle=False, collate_fn=trad_collator,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    hybrid_train_loader = DataLoader(
        hybrid_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=hybrid_collator,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    hybrid_val_loader = DataLoader(
        hybrid_val, batch_size=BATCH_SIZE, shuffle=False, collate_fn=hybrid_collator,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    
    # Step 6: Evaluate original mBERT perplexity
    logger.info("=== Evaluating Original mBERT ===")
    original_model = AutoModelForMaskedLM.from_pretrained("bert-base-multilingual-cased").to(DEVICE)
    original_model.eval()
    
    # Use validation set for perplexity evaluation
    orig_perplexity = compute_perplexity(
        original_model, trad_tokenizer, trad_val_loader, DEVICE, max_samples=100
    )
    logger.info(f"Original mBERT Perplexity: {orig_perplexity:.2f}")
    
    # Step 7: Fine-tune hybrid model (first)
    logger.info("=== Fine-tuning Hybrid mBERT+BPE ===")
    hybrid_model, hybrid_results = finetune_with_wandb(
        hybrid_train_loader, hybrid_val_loader, hybrid_tokenizer, "hybrid"
    )
    
    # Step 8: Fine-tune traditional mBERT (second)
    logger.info("=== Fine-tuning Traditional mBERT ===")
    trad_model, trad_results = finetune_with_wandb(
        trad_train_loader, trad_val_loader, trad_tokenizer, "traditional"
    )
    
    # Step 9: Evaluate perplexity on fine-tuned models
    logger.info("=== Evaluating Fine-tuned Models ===")
    
    # Traditional fine-tuned model
    trad_perplexity = compute_perplexity(
        trad_model, trad_tokenizer, trad_val_loader, DEVICE, max_samples=100
    )
    logger.info(f"Fine-tuned Traditional mBERT Perplexity: {trad_perplexity:.2f}")
    
    # Hybrid fine-tuned model
    hybrid_perplexity = compute_perplexity(
        hybrid_model, hybrid_tokenizer, hybrid_val_loader, DEVICE, max_samples=100
    )
    logger.info(f"Fine-tuned Hybrid mBERT+BPE Perplexity: {hybrid_perplexity:.2f}")
    
    # Step 10: Log final results to WandB (create summary run)
    logger.info("=== Logging Final Results to WandB ===")
    
    summary_run = wandb.init(
        project="mlm-finetuning",
        name="perplexity-comparison",
        reinit=True
    )
    wandb.log({
        "perplexity/original_mbert": orig_perplexity,
        "perplexity/traditional_finetuned": trad_perplexity,
        "perplexity/hybrid_finetuned": hybrid_perplexity,
        "improvement/traditional_absolute": orig_perplexity - trad_perplexity,
        "improvement/traditional_percentage": ((orig_perplexity - trad_perplexity) / orig_perplexity) * 100,
        "improvement/hybrid_absolute": orig_perplexity - hybrid_perplexity,
        "improvement/hybrid_percentage": ((orig_perplexity - hybrid_perplexity) / orig_perplexity) * 100,
    })
    wandb.finish()
    
    # Step 11: Save results
    results = {
        "original_mbert_perplexity": orig_perplexity,
        "traditional_finetuned_perplexity": trad_perplexity,
        "hybrid_finetuned_perplexity": hybrid_perplexity,
        "improvements": {
            "traditional": {
                "absolute": orig_perplexity - trad_perplexity,
                "percentage": ((orig_perplexity - trad_perplexity) / orig_perplexity) * 100
            },
            "hybrid": {
                "absolute": orig_perplexity - hybrid_perplexity,
                "percentage": ((orig_perplexity - hybrid_perplexity) / orig_perplexity) * 100
            }
        },
        "config": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "mlm_prob": MLM_PROB,
        }
    }
    
    results_path = OUTPUT_DIR / "perplexity_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    logger.info("=== Results Summary ===")
    logger.info(f"Original mBERT Perplexity: {orig_perplexity:.2f}")
    logger.info(f"Fine-tuned Traditional mBERT Perplexity: {trad_perplexity:.2f}")
    logger.info(f"  → Improvement: {results['improvements']['traditional']['absolute']:.2f} ({results['improvements']['traditional']['percentage']:.2f}%)")
    logger.info(f"Fine-tuned Hybrid mBERT+BPE Perplexity: {hybrid_perplexity:.2f}")
    logger.info(f"  → Improvement: {results['improvements']['hybrid']['absolute']:.2f} ({results['improvements']['hybrid']['percentage']:.2f}%)")
    logger.info(f"Results saved to {results_path}")
    logger.info("Training complete!")


if __name__ == "__main__":
    main()

