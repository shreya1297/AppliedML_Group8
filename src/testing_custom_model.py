"""
Inference script for TinyTransformerMCQ.

What it does:
- Loads saved checkpoint (models/custom/best.pt)
- Loads test.csv
- Applies same preprocess_mc_batch(...) tokenization
- Produces logits [N,4], converts to predicted label via argmax
- Writes submission CSV: id,label

Output:
- submission/custom_transformer_submission.csv
"""

import os
import torch
import pandas as pd
from transformers import AutoTokenizer

from preprocessing import load_data, preprocess_mc_batch
from custom_model import TinyTransformerMCQ
from config import TEST_CSV, MODEL_DIR


@torch.no_grad()
def main():
    # Choose device
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint saved by train_custom_model.py
    ckpt_path = os.path.join(MODEL_DIR, "best.pt")
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}. Train first."

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Load tokenizer used during training
    tokenizer = AutoTokenizer.from_pretrained(ckpt["tokenizer_name"], use_fast=True)
    max_len = ckpt["max_len"]

    # Reconstruct model and load weights
    model = TinyTransformerMCQ(
        vocab_size=ckpt["vocab_size"],
        max_len=ckpt["max_len"],
        d_model=ckpt["d_model"],
        num_heads=ckpt["num_heads"],
        num_layers=ckpt["num_layers"],
        d_ff=ckpt["d_ff"],
        dropout=ckpt["dropout"],
        pad_token_id=ckpt["pad_token_id"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load test data
    test_df = load_data(TEST_CSV)

    # We will do manual batching to avoid extra dependencies and keep it simple.
    batch_size = 16
    preds = []

    for start in range(0, len(test_df), batch_size):
        batch_df = test_df.iloc[start:start + batch_size]

        # preprocess_mc_batch expects a dict of lists (HF datasets style)
        batch = {
            "context": batch_df["context"].tolist(),
            "question": batch_df["question"].tolist(),
            "answers": batch_df["answers"].tolist(),
        }

        enc = preprocess_mc_batch(batch, tokenizer, max_length=max_len)

        # Convert to tensors
        input_ids = torch.tensor(enc["input_ids"], dtype=torch.long).to(device)        # [B,4,L]
        attn = torch.tensor(enc["attention_mask"], dtype=torch.long).to(device)        # [B,4,L]

        # Forward pass -> logits [B,4]
        logits = model(input_ids, attn)

        # Choose best option
        batch_pred = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
        preds.extend(batch_pred)

    # Write submission file
    os.makedirs("submission", exist_ok=True)
    out_path = "submission/custom_transformer_submission.csv"
    pd.DataFrame({"id": test_df["id"], "label": preds}).to_csv(out_path, index=False)

    print(f"✅ Saved: {out_path}")


if __name__ == "__main__":
    main()
