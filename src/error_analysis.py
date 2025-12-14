"""
Error Analysis Script for Transformer Multiple Choice Model.
Analyzes where the model makes mistakes and why.
"""

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
from datasets import Dataset
import json
from collections import defaultdict

from preprocessing import load_data, split_dataset
from tokenizer_trainonly import TrainOnlyVocab
from preprocessing_scratch import preprocess_mc_batch_scratch
from transformer_model import TransformerMCQModel
from config import TRAIN_CSV, EMBED_MAX_LEN


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model_and_vocab(checkpoint_path, device):
    """Load model and vocab from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Reconstruct vocab
    itos = ckpt["vocab"]
    stoi = {t: i for i, t in enumerate(itos)}
    vocab = TrainOnlyVocab(stoi=stoi, itos=itos)
    
    # Get config
    cfg = ckpt.get("config", None)
    if cfg is None:
        raise RuntimeError("Checkpoint missing config")
    
    # Build model
    model = TransformerMCQModel(
        vocab_size=vocab.vocab_size,
        pad_token_id=vocab.pad_id,
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_ff=cfg["dim_ff"],
        dropout=cfg["dropout"],
        pooling=cfg["pooling"],
    ).to(device)
    
    model.load_state_dict(ckpt["model"])
    model.eval()
    
    return model, vocab, cfg


def analyze_predictions(model, dataloader, val_df, device):
    """Generate predictions and analyze errors."""
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Get predictions
            preds = torch.argmax(logits, dim=1)
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_logits)


def calculate_metrics(predictions, labels):
    """Calculate various metrics."""
    correct = predictions == labels
    accuracy = correct.mean()
    
    # Per-class accuracy
    class_accuracies = {}
    for i in range(4):  # 4 answer choices
        mask = labels == i
        if mask.sum() > 0:
            class_accuracies[i] = correct[mask].mean()
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'class_accuracies': class_accuracies
    }


def analyze_confidence(logits, predictions, labels):
    """Analyze model confidence."""
    # Get softmax probabilities
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()
    
    # Confidence = probability of predicted class
    confidence = probs[np.arange(len(predictions)), predictions]
    
    # Analyze confidence for correct vs incorrect
    correct = predictions == labels
    
    return {
        'probs': probs,
        'confidence': confidence,
        'avg_confidence_correct': confidence[correct].mean() if correct.any() else 0.0,
        'avg_confidence_incorrect': confidence[~correct].mean() if (~correct).any() else 0.0,
    }


def find_worst_examples(val_df, predictions, labels, probs, n=10):
    """Find examples where model was most confident but wrong."""
    # Get confidence in predicted class
    confidence = probs[np.arange(len(predictions)), predictions]
    
    # Find incorrect predictions
    incorrect_mask = predictions != labels
    incorrect_indices = np.where(incorrect_mask)[0]
    
    if len(incorrect_indices) == 0:
        print("No incorrect predictions found!")
        return []
    
    # Sort by confidence (high confidence but wrong)
    incorrect_confidence = confidence[incorrect_indices]
    sorted_indices = np.argsort(-incorrect_confidence)[:n]
    worst_indices = incorrect_indices[sorted_indices]
    
    worst_examples = []
    for idx in worst_indices:
        # Convert answers to list if it's an array or other type
        answers = val_df.iloc[idx]['answers']
        if isinstance(answers, np.ndarray):
            answers = answers.tolist()
        elif not isinstance(answers, list):
            answers = list(answers)
        
        example = {
            'id': str(val_df.iloc[idx]['id']),
            'context': str(val_df.iloc[idx]['context']),
            'question': str(val_df.iloc[idx]['question']),
            'answers': answers,
            'true_label': int(labels[idx]),
            'predicted_label': int(predictions[idx]),
            'confidence': float(confidence[idx]),
            'probabilities': probs[idx].tolist(),
        }
        worst_examples.append(example)
    
    return worst_examples


def analyze_error_patterns(val_df, predictions, labels):
    """Analyze patterns in errors."""
    # Confusion matrix
    confusion = np.zeros((4, 4), dtype=int)
    for true_label, pred_label in zip(labels, predictions):
        confusion[true_label, pred_label] += 1
    
    # Length analysis
    incorrect_mask = predictions != labels
    incorrect_indices = np.where(incorrect_mask)[0]
    
    # Calculate average context length for correct vs incorrect
    def get_text_length(text):
        return len(str(text).split())
    
    incorrect_context_lengths = [
        get_text_length(val_df.iloc[idx]['context'])
        for idx in incorrect_indices
    ]
    
    correct_indices = np.where(~incorrect_mask)[0]
    correct_context_lengths = [
        get_text_length(val_df.iloc[idx]['context'])
        for idx in correct_indices
    ]
    
    return {
        'confusion_matrix': confusion,
        'avg_context_length_incorrect': np.mean(incorrect_context_lengths) if incorrect_context_lengths else 0,
        'avg_context_length_correct': np.mean(correct_context_lengths) if correct_context_lengths else 0,
    }


def print_examples(examples, n=2):
    """Print detailed examples."""
    print(f"\n{'='*80}")
    print(f"DETAILED ERROR EXAMPLES (showing {min(n, len(examples))} examples)")
    print(f"{'='*80}\n")
    
    for i, example in enumerate(examples[:n], 1):
        print(f"\n{'─'*80}")
        print(f"Example {i}: {example['id']}")
        print(f"{'─'*80}")
        
        print(f"\n📝 CONTEXT:")
        context = example['context']
        if len(context) > 500:
            print(f"{context[:500]}...")
        else:
            print(context)
        
        print(f"\n❓ QUESTION:")
        print(f"{example['question']}")
        
        print(f"\n📋 ANSWER CHOICES:")
        for j, answer in enumerate(example['answers']):
            marker = ""
            if j == example['true_label']:
                marker = " ✅ (CORRECT)"
            elif j == example['predicted_label']:
                marker = f" ❌ (MODEL PREDICTED - {example['confidence']:.2%} confidence)"
            print(f"  {j}. {answer}{marker}")
        
        print(f"\n📊 MODEL PROBABILITIES:")
        for j, prob in enumerate(example['probabilities']):
            marker = "←" if j == example['predicted_label'] else ""
            print(f"  Choice {j}: {prob:.4f} ({prob*100:.2f}%) {marker}")
        
        print(f"\n💡 ANALYSIS:")
        print(f"  • Model predicted: Choice {example['predicted_label']}")
        print(f"  • Correct answer: Choice {example['true_label']}")
        print(f"  • Confidence: {example['confidence']:.2%}")
        
        # Analyze why it might be wrong
        pred_prob = example['probabilities'][example['predicted_label']]
        true_prob = example['probabilities'][example['true_label']]
        diff = pred_prob - true_prob
        
        if diff < 0.1:
            print(f"  • The model was uncertain (only {diff:.2%} difference)")
        else:
            print(f"  • The model was confident but wrong ({diff:.2%} difference)")
        
        # Check if there's a close second choice
        sorted_probs = sorted(enumerate(example['probabilities']), key=lambda x: -x[1])
        second_choice, second_prob = sorted_probs[1]
        if second_choice == example['true_label']:
            print(f"  • The correct answer was the second choice ({second_prob:.2%})")
        
        print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Error Analysis for Transformer Model")
    
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="error_analysis",
        help="Directory to save error analysis results"
    )
    parser.add_argument(
        "--n_examples",
        type=int,
        default=10,
        help="Number of error examples to save"
    )
    parser.add_argument(
        "--show_examples",
        type=int,
        default=2,
        help="Number of error examples to display"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("TRANSFORMER MODEL ERROR ANALYSIS")
    print("="*80)
    
    # Set seed
    set_seed(args.seed)
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n📁 Loading model from: {args.checkpoint_path}")
    print(f"🖥️  Device: {device}")
    
    model, vocab, cfg = load_model_and_vocab(args.checkpoint_path, device)
    print(f"✅ Model loaded successfully")
    print(f"   Vocab size: {vocab.vocab_size}")
    print(f"   Config: {cfg}")
    
    # Load validation data
    print(f"\n📊 Loading validation data...")
    df = load_data(TRAIN_CSV)
    train_set, val_set = split_dataset(df)
    val_df = val_set.to_pandas()
    print(f"   Validation samples: {len(val_df)}")
    
    # Preprocess validation data
    print(f"\n🔄 Preprocessing validation data...")
    val_set = val_set.map(
        lambda b: preprocess_mc_batch_scratch(b, vocab, max_length=cfg.get("max_len", EMBED_MAX_LEN)),
        batched=True
    )
    val_set.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    
    # Generate predictions and analyze
    print("\n🔍 Analyzing model predictions...")
    predictions, labels, logits = analyze_predictions(model, val_loader, val_df, device)
    
    # Calculate metrics
    print("\n📈 Calculating metrics...")
    metrics = calculate_metrics(predictions, labels)
    
    print(f"\n{'='*80}")
    print("OVERALL RESULTS")
    print(f"{'='*80}")
    print(f"\n✓ Overall Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"\n📊 Per-Class Accuracy:")
    for choice, acc in metrics['class_accuracies'].items():
        print(f"   Choice {choice}: {acc:.4f} ({acc*100:.2f}%)")
    
    # Analyze confidence
    confidence_metrics = analyze_confidence(logits, predictions, labels)
    print(f"\n🎯 Confidence Analysis:")
    print(f"   Avg confidence (correct predictions): {confidence_metrics['avg_confidence_correct']:.4f}")
    print(f"   Avg confidence (incorrect predictions): {confidence_metrics['avg_confidence_incorrect']:.4f}")
    
    # Analyze error patterns
    error_patterns = analyze_error_patterns(val_df, predictions, labels)
    print(f"\n📏 Length Analysis:")
    print(f"   Avg context length (correct): {error_patterns['avg_context_length_correct']:.1f} words")
    print(f"   Avg context length (incorrect): {error_patterns['avg_context_length_incorrect']:.1f} words")
    
    print(f"\n🎭 Confusion Matrix:")
    print("     Predicted →")
    print("   ", "  ".join([f"  {i}" for i in range(4)]))
    for i in range(4):
        row_str = f" {i} "
        for j in range(4):
            row_str += f" {error_patterns['confusion_matrix'][i][j]:3d}"
        print(row_str)
    print("   ↑")
    print("   True")
    
    # Find worst examples
    print(f"\n🔎 Finding worst error examples...")
    worst_examples = find_worst_examples(
        val_df, predictions, labels, 
        confidence_metrics['probs'], 
        n=args.n_examples
    )
    
    # Print detailed examples
    if worst_examples:
        print_examples(worst_examples, n=args.show_examples)
    
    # Save results
    results = {
        'checkpoint': args.checkpoint_path,
        'config': cfg,
        'accuracy': float(metrics['accuracy']),
        'class_accuracies': {str(k): float(v) for k, v in metrics['class_accuracies'].items()},
        'confidence_correct': float(confidence_metrics['avg_confidence_correct']),
        'confidence_incorrect': float(confidence_metrics['avg_confidence_incorrect']),
        'avg_context_length_correct': float(error_patterns['avg_context_length_correct']),
        'avg_context_length_incorrect': float(error_patterns['avg_context_length_incorrect']),
        'confusion_matrix': error_patterns['confusion_matrix'].tolist(),
        'worst_examples': worst_examples,
    }
    
    output_file = os.path.join(args.output_dir, 'error_analysis_transformer.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✅ Error analysis complete!")
    print(f"📄 Results saved to: {output_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()