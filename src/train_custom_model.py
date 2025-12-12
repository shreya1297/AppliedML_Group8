"""
Train script for TinyTransformerMCQ.

What it does:
- Loads train.csv
- Splits into train/val (stratified)
- Uses your preprocess_mc_batch(...) to build MCQ tensors:
    input_ids: [B, 4, L]
    attention_mask: [B, 4, L]
    labels: [B]
- Trains with CrossEntropyLoss on logits [B,4]
- Prints train + val accuracy each epoch
- Saves best checkpoint as models/custom/best.pt
"""

import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from preprocessing import load_data, split_dataset, preprocess_mc_batch
from custom_model import TinyTransformerMCQ
from config import TRAIN_CSV, MODEL_DIR, EMBED_D_MODEL, EMBED_MAX_LEN, EMBED_DROPOUT


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    logits: [B,4]
    labels: [B]
    """
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Runs evaluation on the validation loader.
    Returns average loss and accuracy.
    """
    model.eval()
    ce = torch.nn.CrossEntropyLoss()

    losses = []
    accs = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)          # [B,4,L]
        attn = batch["attention_mask"].to(device)          # [B,4,L]
        labels = batch["labels"].to(device)                # [B]

        logits = model(input_ids, attn)                    # [B,4]
        loss = ce(logits, labels)

        losses.append(loss.item())
        accs.append(accuracy_from_logits(logits, labels))

    return float(np.mean(losses)), float(np.mean(accs))


def main():
    # ---- hyperparameters (safe defaults for scratch training) ----
    tokenizer_name = "distilroberta-base"  # tokenizer ONLY (we do NOT use pretrained encoder)
    batch_size = 8
    lr = 3e-4
    epochs = 8
    weight_decay = 0.01

    # Tiny Transformer config
    num_heads = 4
    num_layers = 2
    d_ff = 512

    # Max sequence length used in preprocessing
    max_len = EMBED_MAX_LEN

    # Choose device
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    best_path = os.path.join(MODEL_DIR, "best.pt")

    # ---- Load and split dataset ----
    df = load_data(TRAIN_CSV)
    train_ds, val_ds = split_dataset(df, test_size=0.1)  # stratified split inside preprocessing.py

    # ---- Tokenizer ----
    # We only need tokenizer to map text -> token IDs.
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    # ---- Preprocess into MCQ features ----
    # preprocess_mc_batch creates 4 candidates per example and tokenizes them.
    train_ds = train_ds.map(lambda x: preprocess_mc_batch(x, tokenizer, max_length=max_len), batched=True)
    val_ds = val_ds.map(lambda x: preprocess_mc_batch(x, tokenizer, max_length=max_len), batched=True)

    # Tell HF datasets to output torch tensors for these columns
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # ---- Build model ----
    model = TinyTransformerMCQ(
        vocab_size=tokenizer.vocab_size,
        max_len=max_len,
        d_model=EMBED_D_MODEL,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=EMBED_DROPOUT,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
    ).to(device)

    # ---- Optimizer + loss ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ce = torch.nn.CrossEntropyLoss()

    best_val_acc = -1.0

    # ---- Training loop ----
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()

        train_losses = []
        train_accs = []

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)          # [B,4,L]
            attn = batch["attention_mask"].to(device)          # [B,4,L]
            labels = batch["labels"].to(device)                # [B]

            optimizer.zero_grad(set_to_none=True)

            logits = model(input_ids, attn)                    # [B,4]
            loss = ce(logits, labels)                          # scalar

            loss.backward()

            # Gradient clipping improves stability (especially for scratch Transformers)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            train_losses.append(loss.item())
            train_accs.append(accuracy_from_logits(logits, labels))

        # ---- Validate ----
        val_loss, val_acc = evaluate(model, val_loader, device)
        dt = time.time() - t0

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={np.mean(train_losses):.4f} train_acc={np.mean(train_accs):.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"time={dt:.1f}s"
        )

        # ---- Save best checkpoint by val accuracy ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    # We save everything needed to reconstruct the model for inference
                    "model_state_dict": model.state_dict(),
                    "tokenizer_name": tokenizer_name,
                    "max_len": max_len,
                    "d_model": EMBED_D_MODEL,
                    "num_heads": num_heads,
                    "num_layers": num_layers,
                    "d_ff": d_ff,
                    "dropout": EMBED_DROPOUT,
                    "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
                    "vocab_size": tokenizer.vocab_size,
                },
                best_path,
            )
            print(f"  ✅ Saved new best to: {best_path} (val_acc={best_val_acc:.4f})")

    print(f"Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
