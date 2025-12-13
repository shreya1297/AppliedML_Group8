import torch
import torch.nn as nn
from embeddings import InputEmbedding
from config import EMBED_D_MODEL

class TransformerMCQModel(nn.Module):
    def __init__(self, vocab_size, pad_token_id, nhead=8, num_layers=4, dim_ff=1024, dropout=0.1, pooling="cls"):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.pooling = pooling

        self.input_embedding = InputEmbedding(vocab_size=vocab_size, pad_token_id=pad_token_id)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_D_MODEL,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(EMBED_D_MODEL, 1)

    def _pool(self, h, mask):
        # h: [B*4, L, D], mask: [B*4, L] bool True=token
        if self.pooling == "cls":
            return h[:, 0, :]
        # masked mean
        m = mask.unsqueeze(-1).float()
        return (h * m).sum(1) / m.sum(1).clamp(min=1.0)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        B, C, L = input_ids.shape
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id)

        flat_mask = attention_mask.view(B*C, L).bool()
        h = self.input_embedding(input_ids)                  # [B*4, L, D]
        h = self.encoder(h, src_key_padding_mask=~flat_mask) # pad=True

        pooled = self._pool(h, flat_mask)                    # [B*4, D]
        scores = self.head(self.dropout(pooled)).squeeze(-1) # [B*4]
        return scores.view(B, C)                             # [B,4]
