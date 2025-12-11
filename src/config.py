
# Embedding 

# Dimension of token embeddings (d_model)
EMBED_D_MODEL = 256

# Maximum sequence length (used for positional encodings)
# Should match tokenizer max length used for training.
EMBED_MAX_LEN = 128

# Initialization for embeddings: tuple (method, kwargs)
# Supported (informal): ("normal", {"mean":0.0, "std":0.02}), ("xavier_uniform", {}), ("uniform", {"a":-0.1,"b":0.1})
EMBED_INIT = ("normal", {"mean": 0.0, "std": 0.02})

# Dropout applied to embeddings (before encoder)
EMBED_DROPOUT = 0.1


# Training config


# Model config 


# Data paths
TRAIN_CSV = "data/train.csv"
TEST_CSV = "data/test.csv"
SAMPLE_SUBMISSION_CSV = "data/sample_submission.csv"

# Model save dir
MODEL_DIR = "models/custom/"

