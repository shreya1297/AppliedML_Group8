"""
Tiny Transformer (from scratch) for 4-way multiple-choice reasoning.

Goal:
- Introduce SELF-ATTENTION so tokens can interact (semantic alignment) across:
  context + question + option.
- Works with your preprocessing pipeline which returns:
  input_ids:       [B, 4, L]
  attention_mask:  [B, 4, L]
- Outputs logits:  [B, 4]  (one score per answer option)

Important note:
- We intentionally do NOT use a pretrained encoder here.
- We DO reuse the tokenizer (e.g., distilroberta) only to turn text into token IDs.
"""

import math
import torch
import torch.nn as nn
from typing import Optional



class MultiHeadSelfAttention(nn.Module):
    """
    Standard scaled dot-product multi-head self-attention.

    Why it matters:
    - This is where "semantic similarity / alignment" happens:
      each token can attend to every other token (global interaction).
    - Without this, embeddings are static and tokens don't "talk" to each other.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # One projection to produce Q, K, V in a single matmul:
        # x -> [Q|K|V] each of size d_model
        self.qkv = nn.Linear(d_model, 3 * d_model)

        # Output projection after concatenating heads
        self.out = nn.Linear(d_model, d_model)

        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x:        [B, L, D]
        attn_mask:[B, L] where 1 = real token, 0 = padding token

        returns:  [B, L, D]
        """
        B, L, D = x.shape

        # Project to QKV
        qkv = self.qkv(x)  # [B, L, 3D]

        # Reshape into heads:
        # [B, L, 3D] -> [B, L, 3, H, Hd] -> [3, B, H, L, Hd]
        qkv = qkv.view(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each is [B, H, L, Hd]

        # Scaled dot-product attention scores:
        # scores[b,h,i,j] = dot(q[b,h,i], k[b,h,j])
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, L, L]

        # Mask out padding positions on the KEY side so we never attend to PAD tokens.
        if attn_mask is not None:
            # attn_mask: [B, L] -> [B, 1, 1, L]
            key_mask = attn_mask[:, None, None, :]
            scores = scores.masked_fill(key_mask == 0, -1e9)

        # Softmax converts scores to probabilities across j dimension
        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)

        # Weighted sum of values
        ctx = attn @ v  # [B, H, L, Hd]

        # Concatenate heads back:
        # [B, H, L, Hd] -> [B, L, H, Hd] -> [B, L, D]
        ctx = ctx.transpose(1, 2).contiguous().view(B, L, D)

        return self.out(ctx)


class TransformerEncoderBlock(nn.Module):
    """
    One Transformer encoder block:
    - Multi-head self-attention + residual + layernorm
    - Feedforward network (MLP) + residual + layernorm

    This is the standard building block of BERT-like encoders.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)

        # Position-wise feedforward network:
        # Each token vector is transformed independently by the same MLP.
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Attention sub-layer with residual connection
        x = self.norm1(x + self.drop1(self.attn(x, attn_mask)))

        # Feedforward sub-layer with residual connection
        x = self.norm2(x + self.drop2(self.ff(x)))

        return x


class TinyTransformerMCQ(nn.Module):
    """
    Tiny Transformer for multiple-choice classification (4 options).

    We implement embeddings ourselves:
    - token embedding
    - learned positional embedding

    Then:
    - N encoder layers
    - mean pooling (masked) -> one vector per candidate
    - linear head -> one logit per candidate
    """

    def __init__(
        self,
        vocab_size: int,
        max_len: int = 128,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.max_len = max_len
        self.d_model = d_model

        # Token embeddings (semantic priors)
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)

        # Positional embeddings (order info)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.emb_drop = nn.Dropout(dropout)

        # Stack of encoder blocks
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Classification head: produces one score for each candidate sequence
        self.cls = nn.Linear(d_model, 1)

        # Lightweight init (keeps training stable)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        input_ids:      [B, 4, L]
        attention_mask: [B, 4, L] (1 for real tokens, 0 for padding)

        returns logits: [B, 4]
        """
        B, C, L = input_ids.shape
        assert C == 4, "Expected exactly 4 answer options per example"

        # Flatten the choices so we can run them through the encoder in one batch.
        # [B, 4, L] -> [B*4, L]
        input_ids = input_ids.view(B * C, L)
        attention_mask = attention_mask.view(B * C, L)

        # Build position indices [0..L-1] for each sequence
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B * C, L)
        # If L > max_len, clamp positions so we don't index beyond pos_emb
        pos = pos.clamp(max=self.max_len - 1)

        # Embedding lookup and sum: token + position
        x = self.tok_emb(input_ids) + self.pos_emb(pos)  # [B*4, L, D]
        x = self.emb_drop(x)

        # Encoder stack (attention happens here!)
        for layer in self.layers:
            x = layer(x, attention_mask)  # still [B*4, L, D]

        # Convert token-level output into one vector per sequence:
        # Masked mean pooling over non-pad tokens.
        mask = attention_mask.unsqueeze(-1)  # [B*4, L, 1]
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)  # [B*4, D]

        # Score each candidate sequence -> 1 logit
        logits = self.cls(pooled).view(B, C)  # [B, 4]
        return logits
