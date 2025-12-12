# testing baseline.py

import argparse
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMultipleChoice,
    TrainingArguments,
    Trainer,
    DataCollatorForMultipleChoice,
    set_seed,
)

from preprocessing import load_data, preprocess_mc_batch


def get_device():
    """
    Detect and log the available device (MPS, CUDA, or CPU).
    Trainer will automatically use this device.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using Mac GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU")
    print(f"Device: {device}")
    return device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference and create submission for the baseline multiple-choice model"
    )

    parser.add_argument(
        "--test_path",
        type=str,
        default="../../data/test.csv",
        help="Path to the test CSV file.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./results/baseline",
        help="Path to the trained model directory (same as baseline.py --output_dir).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="../../submission/submission.csv",
        help="Path to save the submission CSV.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,  # ⚠️ Must match the max_length used during training
        help="Maximum sequence length for tokenization (must match training).",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=4,
        help="Per-device batch size for inference.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    get_device()  # Just prints; Trainer uses the appropriate device automatically

    # --- 1. Load Test Data ---
    print(f"📂 Loading test data from {args.test_path} ...")
    # Reuse load_data so 'answers' is parsed with ast.literal_eval
    test_df = load_data(args.test_path)

    if "id" not in test_df.columns:
        raise ValueError("Test file must contain an 'id' column.")

    test_ids = test_df["id"].tolist()

    # Create Hugging Face Dataset
    test_dataset = Dataset.from_pandas(test_df)

    # --- 2. Load Model & Tokenizer ---
    print(f"🔧 Loading model and tokenizer from {args.model_path} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForMultipleChoice.from_pretrained(args.model_path)
    except OSError:
        print(f"❌ Error: Could not load model/tokenizer from '{args.model_path}'.")
        print("   Make sure you have trained and saved the baseline model (baseline.py).")
        return

    # Safety for predict as well (avoids weird graph issues)
    model.config.use_cache = False

    # Use fp16 only if CUDA (not for MPS/CPU)
    use_fp16 = torch.cuda.is_available() and not torch.backends.mps.is_available()

    # --- 3. Preprocessing (must match training logic) ---
    print("🧪 Tokenizing test dataset ...")
    encoded_test = test_dataset.map(
        lambda batch: preprocess_mc_batch(
            batch, tokenizer=tokenizer, max_length=args.max_length
        ),
        batched=True,
        remove_columns=test_dataset.column_names,  # keep only tokenized features
    )

    data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)

    # --- 4. Set Up Trainer for Inference ---
    inference_args = TrainingArguments(
        output_dir="./inference_results",
        per_device_eval_batch_size=args.eval_batch_size,
        fp16=use_fp16,
        dataloader_drop_last=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=inference_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # --- 5. Run Predictions ---
    print("🔮 Running predictions on test set ...")
    predictions = trainer.predict(encoded_test)

    # predictions.predictions: [num_examples, 4] logits → take argmax
    logits = predictions.predictions
    predicted_labels = np.argmax(logits, axis=1)

    if len(predicted_labels) != len(test_ids):
        raise RuntimeError(
            f"Length mismatch: {len(predicted_labels)} predictions vs {len(test_ids)} test ids."
        )

    # --- 6. Create Submission File ---
    submission_df = pd.DataFrame(
        {
            "id": test_ids,
            "label": predicted_labels,
        }
    )

    print(f"💾 Saving submission to {args.output_file} ...")
    submission_df.to_csv(args.output_file, index=False)

    print("✅ Submission file created.")
    print(submission_df.head())


if __name__ == "__main__":
    main()
