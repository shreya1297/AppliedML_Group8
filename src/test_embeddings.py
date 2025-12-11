import torch
from src.embeddings import InputEmbedding
from src.config import EMBED_D_MODEL, EMBED_MAX_LEN

# ----------------------------
# Fake setup
# ----------------------------

batch_size = 2
num_choices = 4
seq_len = 20

vocab_size = 30522       # pretend tokenizer size
pad_token_id = 0         # pretend pad ID

# Random fake token IDs (range 0..vocab_size-1)
fake_input_ids = torch.randint(
    low=0,
    high=vocab_size,
    size=(batch_size, num_choices, seq_len)
)

# ----------------------------
# Initialize embedding module
# ----------------------------
embed = InputEmbedding(
    vocab_size=vocab_size,
    pad_token_id=pad_token_id
)

# ----------------------------
# Run forward
# ----------------------------
output = embed(fake_input_ids)

# ----------------------------
# Print results
# ----------------------------
print("Input shape :", fake_input_ids.shape)
print("Output shape:", output.shape)
print("Expected    : (batch*4, seq_len, d_model)")
print("d_model     :", EMBED_D_MODEL)
