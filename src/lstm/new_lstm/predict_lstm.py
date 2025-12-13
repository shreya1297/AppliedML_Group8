"""
Inference script for LSTM Multiple Choice Model.
Generates predictions on test data.
"""

import argparse
import torch
import pandas as pd
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from transformers import AutoTokenizer
from src.preprocessing import preprocess_mc_batch
from src.lstm.new_lstm.lstm_model import LSTMMultipleChoice
from datasets import Dataset


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


def load_test_data(test_path):
    """Load test data from CSV."""
    df = pd.read_csv(test_path)
    # Convert string representation of list to actual list
    import ast
    df["answers"] = df["answers"].apply(ast.literal_eval)
    return df


def predict(model, dataloader, device):
    """Generate predictions."""
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            
            logits = outputs['logits']
            
            # Get predictions
            preds = torch.argmax(logits, dim=1)
            all_predictions.extend(preds.cpu().numpy())
    
    return all_predictions


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate predictions using trained LSTM model"
    )
    
    parser.add_argument(
        "--test_path",
        type=str,
        default="data/test.csv",
        help="Path to test CSV file"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="src/lstm/new_lstm/checkpoints/best_model.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="roberta-base",
        help="Tokenizer name (should match training)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="src/lstm/new_lstm/submission.csv",
        help="Path to save predictions"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for inference"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Get device
    device = get_device()
    
    # --- 1. Load checkpoint ---
    print(f"\n📂 Loading model checkpoint from {args.checkpoint_path} ...")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    
    # Get model arguments from checkpoint
    model_args = checkpoint['args']
    print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")
    print(f"   Validation Accuracy: {checkpoint['val_acc']:.2f}%")
    
    # --- 2. Load tokenizer ---
    print(f"\n🔧 Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    # --- 3. Load test data ---
    print(f"\n📂 Loading test data from {args.test_path} ...")
    test_df = load_test_data(args.test_path)
    print(f"✅ Loaded {len(test_df)} test samples")
    
    # Convert to HF Dataset
    test_dataset = Dataset.from_pandas(test_df)
    
    # Tokenize
    print("🧪 Tokenizing test dataset ...")
    encoded_test = test_dataset.map(
        lambda batch: preprocess_mc_batch(
            batch, 
            tokenizer=tokenizer, 
            max_length=model_args.get('max_length', 128)
        ),
        batched=True,
        remove_columns=test_dataset.column_names,
    )
    
    # --- 4. Create DataLoader ---
    from torch.utils.data import DataLoader
    
    class MCDataset(torch.utils.data.Dataset):
        def __init__(self, hf_dataset):
            self.dataset = hf_dataset
            
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            item = self.dataset[idx]
            return {
                'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long),
            }
    
    def collate_fn(batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
    
    test_loader = DataLoader(
        MCDataset(encoded_test),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    # --- 5. Initialize model ---
    print(f"\n🏗️  Initializing LSTM model...")
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
    
    print("✅ Model loaded successfully")
    
    # --- 6. Generate predictions ---
    print(f"\n🔮 Generating predictions...")
    predictions = predict(model, test_loader, device)
    
    # --- 7. Create submission file ---
    print(f"\n💾 Saving predictions to {args.output_path} ...")
    
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'label': predictions
    })
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    submission_df.to_csv(args.output_path, index=False)
    
    print(f"✅ Submission file saved!")
    print(f"\nPrediction distribution:")
    print(submission_df['label'].value_counts().sort_index())
    print(f"\n🎉 Inference complete!")


if __name__ == "__main__":
    main()
