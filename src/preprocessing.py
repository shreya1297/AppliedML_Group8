import pandas as pd
import ast
from datasets import Dataset


def load_data(path: str) -> pd.DataFrame:
    """
    Load CSV and convert the 'answers' column (string → list).
    e.g., "['Option A', 'Option B']" -> ['Option A', 'Option B']
    """
    df = pd.read_csv(path)
    df["answers"] = df["answers"].apply(ast.literal_eval)
    return df


#def split_dataset(df: pd.DataFrame, test_size: float = 0.1):
#    """
#    Convert DataFrame to HuggingFace Dataset and create train/validation split.
#    """
#    hf_dataset = Dataset.from_pandas(df)
#    split = hf_dataset.train_test_split(test_size=test_size)
#    return split["train"], split["test"]

'''
Because the dataset labels correspond to the correct answer index (0–3), and these labels are often imbalanced, using a stratified train/validation split is essential.
Stratification ensures that each label appears in the validation set with the same proportion as in the full dataset.
Without stratification, the validation set may become skewed toward easier labels, leading to misleading accuracy estimates, unfair model comparisons, unstable hyperparameter tuning results, and unreliable seed stability analysis.
Therefore, we use a stratified split (based on the label column) to guarantee that validation metrics reflect the true performance of the model.
'''
from sklearn.model_selection import train_test_split

def split_dataset(df, test_size=0.1):
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=42,
        stratify=df["label"]
    )
    return Dataset.from_pandas(train_df), Dataset.from_pandas(val_df)


def preprocess_mc_batch(batch, tokenizer, max_length=256):
    """
    Preprocess one batch for multiple-choice models.

    NOTE:
    Each question has 4 answer choices → model expects 4 separate inputs:
        (context, question + option_i) for i in 1..4.
    We create 4 tokenized sequences per question accordingly.
    """

    # Repeat context 4 times (once per answer choice)
    first_sentences = [[ctx] * 4 for ctx in batch["context"]]

    # Combine question with each answer option
    # Structure: [CLS] Context [SEP] Question + Option [SEP]
    second_sentences = [
        [f"{question} {option}" for option in options]
        for question, options in zip(batch["question"], batch["answers"])
    ]

    # Flatten lists for tokenizer
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    # Tokenize
    tokenized = tokenizer(
        first_sentences,
        second_sentences,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

    # TODO: CHECK
    # RoBERTa does not use token_type_ids (segment embeddings).
    # Some tokenizers still return this field for API consistency.
    # HF models safely ignore it, but our custom models (LSTM / Transformer)
    # may NOT accept unexpected inputs. Therefore we remove it here.
    if "token_type_ids" in tokenized:
        del tokenized["token_type_ids"]
            
    # Unflatten back to shape: [batch_size, 4, seq_len]
    result = {
        key: [val[i:i+4] for i in range(0, len(val), 4)]
        for key, val in tokenized.items()
    }

    # Add labels for training
    if "label" in batch:
        result["labels"] = batch["label"]

    return result
