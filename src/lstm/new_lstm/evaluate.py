"""
Evaluation utilities for LSTM model.
Compare different checkpoints and analyze performance.
"""

import torch
import pandas as pd
import numpy as np
import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from transformers import AutoTokenizer
from src.preprocessing import load_data, split_dataset, preprocess_mc_batch
from src.lstm.new_lstm.lstm_model import LSTMMultipleChoice
from torch.utils.data import DataLoader
from datasets import Dataset


def get_device():
    """Detect and return the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class MCDataset(torch.utils.data.Dataset):
    """Custom Dataset for Multiple Choice data."""
    
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(item['labels'], dtype=torch.long),
        }


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }


def evaluate_checkpoint(checkpoint_path, data_loader, device):
    """Evaluate a single checkpoint."""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = checkpoint['args']
    
    # Initialize tokenizer (needed for vocab_size)
    tokenizer = AutoTokenizer.from_pretrained(model_args.get('tokenizer_name', 'roberta-base'))
    
    # Initialize model
    model = LSTMMultipleChoice(
        vocab_size=tokenizer.vocab_size,
        d_model=model_args.get('d_model', 256),
        hidden_size=model_args.get('hidden_size', 256),
        num_layers=model_args.get('num_layers', 2),
        dropout=model_args.get('dropout', 0.1),
        max_len=model_args.get('max_length', 128),
        pad_token_id=tokenizer.pad_token_id,
        bidirectional=model_args.get('bidirectional', True),
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Evaluate
    correct = 0
    total = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            
            logits = outputs['logits']
            preds = torch.argmax(logits, dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    accuracy = 100.0 * correct / total
    
    # Per-class accuracy
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    for pred, true in zip(predictions, true_labels):
        class_total[true] += 1
        if pred == true:
            class_correct[true] += 1
    
    class_accuracies = {
        cls: 100.0 * class_correct[cls] / class_total[cls]
        for cls in class_total.keys()
    }
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'class_accuracies': class_accuracies,
        'epoch': checkpoint.get('epoch', -1) + 1,
        'stored_val_acc': checkpoint.get('val_acc', -1),
    }


def compare_checkpoints(checkpoint_dir, data_loader, device):
    """Compare all checkpoints in a directory."""
    checkpoints = sorted([
        f for f in os.listdir(checkpoint_dir)
        if f.endswith('.pt')
    ])
    
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return
    
    print(f"\n{'='*80}")
    print(f"Comparing {len(checkpoints)} checkpoints")
    print(f"{'='*80}\n")
    
    results = []
    
    for ckpt_name in checkpoints:
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
        print(f"Evaluating {ckpt_name}...")
        
        try:
            result = evaluate_checkpoint(ckpt_path, data_loader, device)
            result['checkpoint'] = ckpt_name
            results.append(result)
            
            print(f"  Epoch: {result['epoch']}")
            print(f"  Accuracy: {result['accuracy']:.2f}%")
            print(f"  Stored Val Acc: {result['stored_val_acc']:.2f}%")
            print()
        except Exception as e:
            print(f"  Error: {e}\n")
            continue
    
    # Sort by accuracy
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY (sorted by accuracy)")
    print(f"{'='*80}\n")
    
    print(f"{'Rank':<6} {'Checkpoint':<30} {'Epoch':<8} {'Accuracy':<12}")
    print("-" * 80)
    
    for i, result in enumerate(results, 1):
        print(f"{i:<6} {result['checkpoint']:<30} {result['epoch']:<8} {result['accuracy']:.2f}%")
    
    # Best model details
    if results:
        best = results[0]
        print(f"\n{'='*80}")
        print("BEST MODEL DETAILS")
        print(f"{'='*80}\n")
        print(f"Checkpoint: {best['checkpoint']}")
        print(f"Epoch: {best['epoch']}")
        print(f"Overall Accuracy: {best['accuracy']:.2f}%")
        print(f"\nPer-class Accuracy:")
        for cls, acc in sorted(best['class_accuracies'].items()):
            print(f"  Choice {cls}: {acc:.2f}%")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate and compare LSTM model checkpoints"
    )
    
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="src/lstm/new_lstm/checkpoints",
        help="Directory containing checkpoints"
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default="data/train.csv",
        help="Path to training CSV (for validation split)"
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.2,
        help="Validation split size (should match training)"
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="roberta-base",
        help="Tokenizer name"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--single_checkpoint",
        type=str,
        help="Evaluate only a single checkpoint"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    
    print(f"Using device: {device}")
    
    # Load validation data
    print(f"\n📂 Loading data from {args.train_path} ...")
    df = load_data(args.train_path)
    _, eval_dataset = split_dataset(df, test_size=args.val_size)
    
    # Tokenize
    print("🧪 Tokenizing validation dataset ...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    encoded_eval = eval_dataset.map(
        lambda batch: preprocess_mc_batch(
            batch, tokenizer=tokenizer, max_length=args.max_length
        ),
        batched=True,
        remove_columns=eval_dataset.column_names,
    )
    
    # Create DataLoader
    eval_loader = DataLoader(
        MCDataset(encoded_eval),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    print(f"✅ Validation samples: {len(encoded_eval)}\n")
    
    # Evaluate
    if args.single_checkpoint:
        print(f"Evaluating single checkpoint: {args.single_checkpoint}")
        result = evaluate_checkpoint(args.single_checkpoint, eval_loader, device)
        print(f"\nResults:")
        print(f"  Accuracy: {result['accuracy']:.2f}%")
        print(f"  Correct: {result['correct']}/{result['total']}")
        print(f"\n  Per-class Accuracy:")
        for cls, acc in sorted(result['class_accuracies'].items()):
            print(f"    Choice {cls}: {acc:.2f}%")
    else:
        compare_checkpoints(args.checkpoint_dir, eval_loader, device)


if __name__ == "__main__":
    main()
