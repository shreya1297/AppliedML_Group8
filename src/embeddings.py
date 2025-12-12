"""
Defines:
  - TokenEmbedding: trainable token embeddings
  - PositionalEncoding: sinusoidal positional encodings
  - InputEmbedding: token + position + dropout block

Uses hyperparameters from src/config.py.
"""

import math
import torch
import torch.nn as nn

# Import embedding configuration
from src.config import (
    EMBED_D_MODEL,
    EMBED_MAX_LEN,
    EMBED_DROPOUT,
    EMBED_INIT,
)

# ============================================================
# Token Embedding
# ============================================================
class TokenEmbedding(nn.Module):
    """
    Trainable token embeddings.

    input_ids: [batch, num_choices, seq_len]
    output:    [batch*num_choices, seq_len, d_model]
    """

    def __init__(self, vocab_size, d_model, pad_token_id=0):
        super().__init__()

        # Embedding table
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_token_id,
        )

        # Initialize weights
        init_type, params = EMBED_INIT

        if init_type == "normal":
            nn.init.normal_(self.embedding.weight, **params)
        else:
            raise ValueError(f"Unknown EMBED_INIT type: {init_type}")

        self.d_model = d_model

    def forward(self, input_ids):
        """
        Flatten and embed:
            [B, 4, S] → [B*4, S]
        """
        batch_size, num_choices, seq_len = input_ids.shape

        flat_ids = input_ids.view(batch_size * num_choices, seq_len)

        # Lookup embeddings
        x = self.embedding(flat_ids)   # → [B*4, S, d_model]

        return x



# ============================================================
# Positional Encoding (Sinusoidal)
# ============================================================
class PositionalEncoding(nn.Module):
    """
    Adds sinusoidal positional encodings.
    Input:  [batch, seq_len, d_model]
    Output: [batch, seq_len, d_model]
    """

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Create positional encoding matrix: [max_len, d_model]
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not learnable)
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        return x


# ============================================================
# Combined Input Embedding Block
# ============================================================
class InputEmbedding(nn.Module):
    """
    token embeddings → positional encodings → dropout
    """

    def __init__(self, vocab_size, pad_token_id):
        super().__init__()

        self.token_embed = TokenEmbedding(
            vocab_size=vocab_size,
            d_model=EMBED_D_MODEL,
            pad_token_id=pad_token_id,
        )

        self.pos_enc = PositionalEncoding(
            d_model=EMBED_D_MODEL,
            max_len=EMBED_MAX_LEN,
        )

        self.dropout = nn.Dropout(EMBED_DROPOUT)

    def forward(self, input_ids):
        """
        input:  [batch, 4, seq_len]
        output: [batch*4, seq_len, EMBED_D_MODEL]
        """

        x = self.token_embed(input_ids)
        x = self.pos_enc(x)
        x = self.dropout(x)

        return x
