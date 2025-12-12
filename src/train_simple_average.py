import torch
from torch.utils.data import DataLoader

from preprocessing import load_data, split_dataset, preprocess_mc_batch
from simple_average import SimpleAverageModel, compute_most_common_label
from config import TRAIN_CSV, TEST_CSV, SAMPLE_SUBMISSION_CSV

import pandas as pd
from transformers import AutoTokenizer


# -------------------------
# Evaluation helper
# -------------------------
def compute_accuracy(preds, labels):
    correct = (preds == labels).sum().item()
    total = len(labels)
    return correct / total


# -------------------------
# Main pipeline
# -------------------------
def main():
    print("Loading data...")

    # Load train CSV
    df = load_data(TRAIN_CSV)

    # Split into train/validation dataset
    train_set, val_set = split_dataset(df)

    print("Computing most common label...")
    mcl = compute_most_common_label(train_set)
    print("Most common label =", mcl)

    # Create the simple average model
    model = SimpleAverageModel(mcl)

    # -------------------------
    # Tokenizer
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

    # Preprocess datasets using your preprocessing function
    print("Tokenizing train & val datasets...")

    train_set = train_set.map(
    lambda batch: preprocess_mc_batch(batch, tokenizer),
    batched=True
)

    val_set = val_set.map(
        lambda batch: preprocess_mc_batch(batch, tokenizer),
        batched=True
    )


    # Convert HF dataset outputs to PyTorch tensors
    train_set.set_format(type="torch")
    val_set.set_format(type="torch")

    # Use PyTorch DataLoader
    train_loader = DataLoader(train_set, batch_size=16)
    val_loader = DataLoader(val_set, batch_size=16)

    # -------------------------
    # “Training” phase (no updates)
    # -------------------------
    print("Running training loop (no real training for this model)...")
    for batch in train_loader:
        # Forward pass → logits
        logits = model(input_ids=batch["input_ids"])
        # No loss.backward(), no optimizer.step()

    # -------------------------
    # Evaluate on validation split
    # -------------------------
    print("Evaluating on validation set...")
    all_preds = []
    all_labels = []

    for batch in val_loader:
        logits = model(input_ids=batch["input_ids"])
        preds = torch.argmax(logits, dim=-1)

        all_preds.append(preds)
        all_labels.append(batch["labels"])

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    val_acc = compute_accuracy(all_preds, all_labels)
    print(f"\nValidation Accuracy (Simple Average Model): {val_acc:.4f}\n")

    # -------------------------
    # Generate submission for test.csv
    # -------------------------
    print("Generating submission file...")

    test_df = pd.read_csv(TEST_CSV)
    test_predictions = [mcl] * len(test_df)

    # Keep the same format as sample_submission.csv
    submission = pd.DataFrame({
        "id": test_df["id"],       # keep ID order from test.csv
        "label": test_predictions  # simple average predictions
    })

    submission.to_csv("submission/simple_average_submission.csv", index=False)
    print("Saved: submission/simple_average_submission.csv")


if __name__ == "__main__":
    main()
