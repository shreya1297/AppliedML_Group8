import pandas as pd
import ast
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForMultipleChoice
from transformers import TrainingArguments, Trainer
import torch

# ============================================================
# STEP 1: LOAD CSV FILES
# ============================================================

train_df = pd.read_csv("../train.csv")
test_df = pd.read_csv("../test.csv")
sample_sub = pd.read_csv("../sample_submission.csv")

# Convert answers string → list
train_df["answers"] = train_df["answers"].apply(ast.literal_eval)
test_df["answers"] = test_df["answers"].apply(ast.literal_eval)

print("Train size:", len(train_df))
print("Test size:", len(test_df))


# ============================================================
# STEP 2: CONVERT INTO HF DATASET
# ============================================================

dataset = Dataset.from_pandas(train_df)
dataset = dataset.train_test_split(test_size=0.1)

test_dataset = Dataset.from_pandas(test_df)


# ============================================================
# STEP 3: TOKENIZER
# ============================================================

model_name = "microsoft/deberta-v3-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def preprocess(batch):
    contexts = batch["context"]
    questions = batch["question"]
    choices = batch["answers"]

    input_ids, attention_masks, labels = [], [], []

    for ctx, q, chs, label in zip(contexts, questions, choices, batch.get("label", [None] * len(contexts))):
        pair_inputs = [ctx + " " + q + " " + choice for choice in chs]

        tok = tokenizer(
            pair_inputs,
            truncation=True,
            padding="max_length",
            max_length=256,
        )

        input_ids.append(tok["input_ids"])
        attention_masks.append(tok["attention_mask"])

        if label is not None:
            labels.append(label)

    out = {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
    }

    if labels:
        out["labels"] = labels

    return out


# Map preprocessing to datasets
train_tokenized = dataset["train"].map(preprocess, batched=True)
val_tokenized = dataset["test"].map(preprocess, batched=True)
test_tokenized = test_dataset.map(preprocess, batched=True)


# ============================================================
# STEP 4: LOAD MODEL
# ============================================================

model = AutoModelForMultipleChoice.from_pretrained(model_name)


# ============================================================
# STEP 5: TRAINING CONFIG
# ============================================================

args = TrainingArguments(
    output_dir="output",
    eval_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    logging_steps=50,
    save_strategy="epoch",
    load_best_model_at_end=True,
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
)


# ============================================================
# STEP 6: TRAIN MODEL
# ============================================================

trainer.train()


# ============================================================
# STEP 7: PREDICT ON TEST SET
# ============================================================

predictions = trainer.predict(test_tokenized)
preds = predictions.predictions.argmax(-1)

test_df["label"] = preds


# ============================================================
# STEP 8: WRITE SUBMISSION FILE
# ============================================================

submission = sample_sub.copy()
submission["label"] = preds
submission.to_csv("submission.csv", index=False)

print("Submission file saved as submission.csv")
