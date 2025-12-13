# a modular, cleaner, more reproducible version of deberta-v3-large from trail_1 using preprocessing.py

import argparse
import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForMultipleChoice,
    TrainingArguments,
    Trainer,
    DataCollatorForMultipleChoice,
    set_seed,
)

from preprocessing import load_data, split_dataset, preprocess_mc_batch


def get_device():
    """
    Detect and log the available device (MPS, CUDA, or CPU).
    Trainer will automatically use this device.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Mac GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("X Using CPU")
    print(f"Device: {device}")
    return device


def compute_metrics(eval_pred):
    """
    Compute accuracy for multiple-choice classification.
    eval_pred is (logits, labels).
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    accuracy = (preds == labels).mean()
    return {"accuracy": float(accuracy)}


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline multiple-choice model training")

    parser.add_argument(
        "--train_path",
        type=str,
        default="../../data/train.csv",
        help="Path to the training CSV file.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="distilroberta-base",
        help="HF model checkpoint (e.g. distilroberta-base, deberta-v3-base, etc.).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/baseline",
        help="Directory to save checkpoints and the final model.",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.2,
        help="Validation set size (fraction of the dataset).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="Per-device eval batch size.",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=16,
        help="Gradient accumulation steps to simulate larger batch size.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate.",
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
    get_device()  # just to print / confirm; Trainer picks it up automatically

    # --- 1. Load & Split Data ---
    print(f"Loading training data from {args.train_path} ...")
    df = load_data(args.train_path)

    print("Performing stratified train/validation split ...")
    train_dataset, eval_dataset = split_dataset(df, test_size=args.val_size)

    # --- 2. Tokenizer & Preprocessing ---
    print(f"Loading tokenizer and model from {args.model_name_or_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    print("Tokenizing train and validation datasets ...")
    encoded_train = train_dataset.map(
        lambda batch: preprocess_mc_batch(
            batch, tokenizer=tokenizer, max_length=args.max_length
        ),
        batched=True,
        remove_columns=train_dataset.column_names,  # drop raw text columns
    )

    encoded_eval = eval_dataset.map(
        lambda batch: preprocess_mc_batch(
            batch, tokenizer=tokenizer, max_length=args.max_length
        ),
        batched=True,
        remove_columns=eval_dataset.column_names,
    )

    # --- 3. Data Collator for Multiple Choice ---
    data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)

    # --- 4. Model ---
    model = AutoModelForMultipleChoice.from_pretrained(args.model_name_or_path)

    # Important for training with gradient checkpointing / Trainer
    model.config.use_cache = False

    # fp16 only if CUDA (not MPS / CPU)
    use_fp16 = torch.cuda.is_available() and not torch.backends.mps.is_available()

    # --- 5. Training Arguments ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        fp16=use_fp16,
        num_train_epochs=args.num_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        report_to="none",  # disable wandb etc. by default
    )

    # --- 6. Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_train,
        eval_dataset=encoded_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # --- 7. Train ---
    print("Starting training of the baseline model ...")
    trainer.train()

    # --- 8. Final Evaluation ---
    print("\n Evaluating on validation set ...")
    metrics = trainer.evaluate()
    print(f"Final Validation Accuracy: {metrics.get('eval_accuracy', 0.0):.2%}")

    # --- 9. Save Best Model & Tokenizer ---
    print(f"\n Saving best model and tokenizer to {args.output_dir} ...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Baseline training complete.")


if __name__ == "__main__":
    main()
