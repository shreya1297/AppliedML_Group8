"""
LSTM-based Multiple Choice Model trained from scratch.
Uses custom embeddings from src/embeddings.py.
"""

import torch
import torch.nn as nn
from src.embeddings import TokenEmbedding, PositionalEncoding
from src.config import EMBED_D_MODEL, EMBED_MAX_LEN, EMBED_DROPOUT


class LSTMMultipleChoice(nn.Module):
    """
    LSTM-based model for multiple-choice question answering.
    
    Architecture:
    1. Token Embeddings (trainable from scratch)
    2. Positional Encoding
    3. Bidirectional LSTM layers
    4. Attention mechanism over LSTM outputs
    5. Classification head for 4-way choice
    
    Input shape: [batch_size, num_choices=4, seq_len]
    Output shape: [batch_size, num_choices=4]
    """
    
    def __init__(
        self,
        vocab_size,
        d_model=EMBED_D_MODEL,
        hidden_size=256,
        num_layers=2,
        dropout=EMBED_DROPOUT,
        max_len=EMBED_MAX_LEN,
        pad_token_id=1,
        bidirectional=True,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # 1. Token Embeddings (trained from scratch)
        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            pad_token_id=pad_token_id,
        )
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len)
        
        # 3. Input Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 4. LSTM Layers
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        
        # 5. Attention mechanism
        lstm_output_size = hidden_size * self.num_directions
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        
        # 6. Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),  # Single score per choice
        )
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass for multiple-choice classification.
        
        Args:
            input_ids: [batch_size, num_choices=4, seq_len]
            attention_mask: [batch_size, num_choices=4, seq_len]
            labels: [batch_size] - correct choice index (0-3)
            
        Returns:
            dict with 'logits' and optionally 'loss'
        """
        batch_size, num_choices, seq_len = input_ids.shape
        
        # 1. Embed tokens: [B, 4, S] → [B*4, S, d_model]
        x = self.token_embedding(input_ids)
        
        # 2. Add positional encodings
        x = self.pos_encoder(x)
        
        # 3. Apply dropout
        x = self.dropout(x)
        
        # 4. Pass through LSTM
        # x: [B*4, S, d_model] → lstm_out: [B*4, S, hidden_size * num_directions]
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 5. Apply attention mechanism
        if attention_mask is not None:
            # Flatten attention mask: [B, 4, S] → [B*4, S]
            flat_mask = attention_mask.view(batch_size * num_choices, seq_len)
            
            # Compute attention scores: [B*4, S, 1]
            attn_scores = self.attention(lstm_out)
            
            # Mask out padding tokens (set to -inf before softmax)
            attn_scores = attn_scores.masked_fill(
                flat_mask.unsqueeze(-1) == 0, float('-inf')
            )
            
            # Apply softmax: [B*4, S, 1]
            attn_weights = torch.softmax(attn_scores, dim=1)
            
            # Weighted sum: [B*4, hidden_size * num_directions]
            context = (lstm_out * attn_weights).sum(dim=1)
        else:
            # No masking: use mean pooling over sequence
            context = lstm_out.mean(dim=1)
        
        # 6. Classification: [B*4, hidden_size * num_directions] → [B*4, 1]
        logits_flat = self.classifier(context)
        
        # 7. Reshape to [B, 4]
        logits = logits_flat.view(batch_size, num_choices)
        
        # 8. Compute loss if labels provided
        output = {"logits": logits}
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            output["loss"] = loss
        
        return output
