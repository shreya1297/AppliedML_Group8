import pandas as pd
import ast
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForMultipleChoice, 
    TrainingArguments, 
    Trainer
)

# Enable Mac GPU (MPS) if available
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

# --- 1. Load and Prepare Data ---
def load_data(file_path):
    df = pd.read_csv(file_path)
    # Parse the 'answers' column (string -> list)
    df['answers'] = df['answers'].apply(ast.literal_eval)
    
    # Map labels to 0-3 integers if needed (your data is already 0-3)
    return df

# Load data
df = load_data('../train.csv')

# Create a Hugging Face Dataset
full_dataset = Dataset.from_pandas(df)

# Split into Train (80%) and Validation (20%) to measure accuracy
dataset_split = full_dataset.train_test_split(test_size=0.2)
train_dataset = dataset_split["train"]
eval_dataset = dataset_split["test"]

# --- 2. Tokenization ---
# We use DeBERTa-v3-base for a good balance of speed and performance. 
# Switch to "microsoft/deberta-v3-large" for maximum accuracy if you have a powerful GPU (24GB+ VRAM).
model_checkpoint = "distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def preprocess_function(examples):
    # The context and question are repeated for each of the 4 options
    first_sentences = [[context] * 4 for context in examples["context"]]
    
    # We pair the question with each option
    # Structure: [CLS] Context [SEP] Question + Option [SEP]
    question_headers = examples["question"]
    second_sentences = [
        [f"{header} {option}" for option in options] 
        for header, options in zip(question_headers, examples["answers"])
    ]

    # Flatten for tokenization
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    tokenized_examples = tokenizer(
        first_sentences, 
        second_sentences, 
        truncation=True, 
        max_length=512,
        padding="max_length"
    )

    # Un-flatten
    return {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}

# Process datasets
encoded_train = train_dataset.map(preprocess_function, batched=True)
encoded_eval = eval_dataset.map(preprocess_function, batched=True)

# --- 3. Metrics (Accuracy) ---
def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    # The model outputs raw logits; we take the index of the highest score
    preds = np.argmax(predictions, axis=1)
    accuracy = (preds == labels).mean()
    return {"accuracy": accuracy}

# --- 4. Model and Training ---
# --- 4. Model and Training ---
model = AutoModelForMultipleChoice.from_pretrained(model_checkpoint)

# CRITICAL FIX: Disable cache to prevent "backward through graph" error
model.config.use_cache = False 

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    
    # --- MEMORY SETTINGS FOR MAC (MPS) ---
    per_device_train_batch_size=1,   # Keep small
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,  # Accumulate to simulate batch size 16
    gradient_checkpointing=True,     # Saves memory
    fp16=False,                      # Disable mixed precision for stability on Mac
    # -------------------------------------

    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_train,
    eval_dataset=encoded_eval,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()

# --- 6. Final Evaluation ---
print("\nCalculating final accuracy on validation set...")
metrics = trainer.evaluate()
print(f"Final Model Accuracy: {metrics['eval_accuracy']:.2%}")