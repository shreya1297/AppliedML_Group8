import pandas as pd
import ast
import torch
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForMultipleChoice, TrainingArguments, Trainer

# --- Configuration ---
TEST_FILE = "../test.csv"
MODEL_PATH = "./results/checkpoint-464"  # Path to your saved model folder
OUTPUT_FILE = "submission.csv"

# --- 1. Load Data ---
print(f"Loading {TEST_FILE}...")
test_df = pd.read_csv(TEST_FILE)

# Parse the 'answers' column (string -> list)
# e.g., "['Option A', 'Option B']" -> ['Option A', 'Option B']
test_df['answers'] = test_df['answers'].apply(ast.literal_eval)

# Create Hugging Face Dataset
test_dataset = Dataset.from_pandas(test_df)

# --- 2. Load Model & Tokenizer ---
print(f"Loading model from {MODEL_PATH}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForMultipleChoice.from_pretrained(MODEL_PATH)
except OSError:
    print(f"Error: Could not find model at {MODEL_PATH}.")
    print("Please make sure you have run the training script and saved the model.")
    exit()

# --- 3. Preprocessing (Must match training logic) ---
def preprocess_function(examples):
    # Repeat context for each of the 4 options
    first_sentences = [[context] * 4 for context in examples["context"]]
    
    # Pair Question with each Option
    question_headers = examples["question"]
    second_sentences = [
        [f"{header} {option}" for option in options] 
        for header, options in zip(question_headers, examples["answers"])
    ]

    # Flatten
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    # Tokenize
    tokenized_examples = tokenizer(
        first_sentences, 
        second_sentences, 
        truncation=True, 
        max_length=512,
        padding="max_length"
    )

    # Un-flatten
    return {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}

print("Tokenizing test data...")
encoded_test = test_dataset.map(preprocess_function, batched=True)

# --- 4. Run Inference ---
# We use the Trainer class just for prediction as it handles batching/GPU automatically
training_args = TrainingArguments(
    output_dir="./inference_results",
    per_device_eval_batch_size=4,  # Adjust if OOM occurs
    fp16=False, # Set True if using CUDA/GPU
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer
)

print("Running predictions...")
predictions = trainer.predict(encoded_test)

# The model returns logits (raw scores). We need the index of the highest score.
# predictions.predictions is the numpy array of logits
predicted_labels = np.argmax(predictions.predictions, axis=1)

# --- 5. Create Submission File ---
print(f"Saving results to {OUTPUT_FILE}...")
submission_df = pd.DataFrame({
    "id": test_df["id"],
    "label": predicted_labels
})

submission_df.to_csv(OUTPUT_FILE, index=False)
print("Done! Check submission.csv")
print(submission_df.head())