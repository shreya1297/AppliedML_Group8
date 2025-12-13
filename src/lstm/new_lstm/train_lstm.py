"""
Training script for LSTM-based Multiple Choice Model.
Trains the model from scratch without using pretrained weights.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from transformers import AutoTokenizer, set_seed
from src.preprocessing import load_data, split_dataset, preprocess_mc_batch
from src.lstm.new_lstm.lstm_model import LSTMMultipleChoice


def get_device():
    """Detect and return the best available device."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using Mac GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU")
    return device


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


def train_epoch(model, dataloader, optimizer, device, epoch, total_epochs):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs['loss']
        logits = outputs['logits']
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        total_loss += loss.item()
        
        # Print progress
        if (batch_idx + 1) % 50 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            accuracy = 100.0 * correct / total
            print(f"Epoch [{epoch+1}/{total_epochs}] "
                  f"Batch [{batch_idx+1}/{len(dataloader)}] "
                  f"Loss: {avg_loss:.4f} "
                  f"Accuracy: {accuracy:.2f}%")
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def evaluate(model, dataloader, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            logits = outputs['logits']
            
            # Calculate accuracy
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train LSTM model from scratch for multiple-choice QA"
    )
    
    parser.add_argument(
        "--train_path",
        type=str,
        default="data/train.csv",
        help="Path to training CSV file"
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="roberta-base",
        help="Tokenizer to use (we only use its vocab, not pretrained weights)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="src/lstm/new_lstm/checkpoints",
        help="Directory to save model checkpoints"
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.2,
        help="Validation split size"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=256,
        help="Embedding dimension"
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=256,
        help="LSTM hidden size"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="Number of LSTM layers"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Training batch size"
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=16,
        help="Evaluation batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        default=True,
        help="Use bidirectional LSTM"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seeds for reproducibility
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Get device
    device = get_device()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- 1. Load & Split Data ---
    print(f"\n📂 Loading training data from {args.train_path} ...")
    df = load_data(args.train_path)
    
    print("🔀 Performing stratified train/validation split ...")
    train_dataset, eval_dataset = split_dataset(df, test_size=args.val_size)
    
    # --- 2. Tokenizer & Preprocessing ---
    print(f"\n🔧 Loading tokenizer: {args.tokenizer_name}")
    print("   (Note: Using tokenizer vocab only, NO pretrained weights)")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    print("🧪 Tokenizing train and validation datasets ...")
    encoded_train = train_dataset.map(
        lambda batch: preprocess_mc_batch(
            batch, tokenizer=tokenizer, max_length=args.max_length
        ),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    
    encoded_eval = eval_dataset.map(
        lambda batch: preprocess_mc_batch(
            batch, tokenizer=tokenizer, max_length=args.max_length
        ),
        batched=True,
        remove_columns=eval_dataset.column_names,
    )
    
    # --- 3. Create DataLoaders ---
    train_loader = DataLoader(
        MCDataset(encoded_train),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    eval_loader = DataLoader(
        MCDataset(encoded_eval),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    print(f"✅ Train samples: {len(encoded_train)}, Val samples: {len(encoded_eval)}")
    
    # --- 4. Initialize Model from Scratch ---
    print(f"\n🏗️  Initializing LSTM model from scratch...")
    print(f"   - Vocab size: {tokenizer.vocab_size}")
    print(f"   - Embedding dim: {args.d_model}")
    print(f"   - Hidden size: {args.hidden_size}")
    print(f"   - Num layers: {args.num_layers}")
    print(f"   - Bidirectional: {args.bidirectional}")
    
    model = LSTMMultipleChoice(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_len=args.max_length,
        pad_token_id=tokenizer.pad_token_id,
        bidirectional=args.bidirectional,
    )
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}")
    
    # --- 5. Optimizer ---
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # --- 6. Training Loop ---
    print(f"\n🚀 Starting training for {args.num_epochs} epochs...")
    print("=" * 80)
    
    best_val_acc = 0.0
    best_epoch = 0
    
    for epoch in range(args.num_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print(f"{'='*80}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device, epoch, args.num_epochs
        )
        
        print(f"\n📈 Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        
        # Evaluate
        val_loss, val_acc = evaluate(model, eval_loader, device)
        print(f"📊 Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            
            checkpoint_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'args': vars(args),
            }, checkpoint_path)
            
            print(f"💾 Saved best model to {checkpoint_path}")
        
        # Save checkpoint every epoch
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
            'args': vars(args),
        }, checkpoint_path)
    
    # --- 7. Final Results ---
    print(f"\n{'='*80}")
    print(f"🎉 Training Complete!")
    print(f"{'='*80}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"Model saved to: {args.output_dir}")
    

if __name__ == "__main__":
    main()
