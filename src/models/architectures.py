# ============================================================
# src/models/architectures.py  —  All Model Architectures
# ============================================================

import math
import torch
import torch.nn as nn


# ── Shared MLP Regression Head ────────────────────────────────────────────────

def mlp_head(in_dim: int, dropout: float = 0.15) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, 64), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(64, 32),     nn.GELU(),
        nn.Linear(32, 1),
    )


# ── 1. Simple RNN (Baseline) ──────────────────────────────────────────────────

class SimpleRNN(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64,
                 layers: int = 1, dropout: float = 0.10):
        super().__init__()
        self.rnn  = nn.RNN(input_dim, hidden, layers,
                           batch_first=True, dropout=0.0, nonlinearity="tanh")
        self.norm = nn.LayerNorm(hidden)
        self.head = mlp_head(hidden, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o, _ = self.rnn(x)
        return self.head(self.norm(o[:, -1, :])).squeeze(-1)


# ── 2. LSTM (Baseline) ────────────────────────────────────────────────────────

class LSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 96,
                 layers: int = 2, dropout: float = 0.15):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers,
                            batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(hidden)
        self.head = mlp_head(hidden, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o, _ = self.lstm(x)
        return self.head(self.norm(o[:, -1, :])).squeeze(-1)


# ── 3. Positional Encoding (shared) ──────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x + self.pe[:, : x.size(1)])


# ── 4. Transformer Encoder-Only (Baseline) ────────────────────────────────────

class TransformerOnly(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 96, nhead: int = 4,
                 n_enc: int = 2, dropout: float = 0.15):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pe   = PositionalEncoding(d_model, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.enc  = nn.TransformerEncoder(enc_layer, num_layers=n_enc)
        self.norm = nn.LayerNorm(d_model)
        self.head = mlp_head(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pe(self.proj(x))
        x = self.enc(x)
        return self.head(self.norm(x[:, -1, :])).squeeze(-1)


# ── 5. Transformer-LSTM Hybrid (Proposed) ────────────────────────────────────

class TransformerLSTM(nn.Module):
    """
    Proposed model:
      Input → Linear projection → Positional Encoding
            → Transformer Encoder (global attention)
            → LSTM (sequential memory)
            → MLP head → scalar prediction
    """
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 8,
                 n_enc: int = 4, lstm_h: int = 128, lstm_l: int = 2,
                 dropout: float = 0.15):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pe   = PositionalEncoding(d_model, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_enc)
        self.lstm = nn.LSTM(d_model, lstm_h, lstm_l,
                            batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(lstm_h)
        self.head = mlp_head(lstm_h, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pe(self.proj(x))
        x = self.transformer(x)
        o, _ = self.lstm(x)
        return self.head(self.norm(o[:, -1, :])).squeeze(-1)


# ── Model Registry ────────────────────────────────────────────────────────────

def build_model_registry(input_dim: int) -> dict:
    """Return dict of model_name → model_factory callable."""
    return {
        "Simple RNN"       : lambda: SimpleRNN(input_dim),
        "LSTM"             : lambda: LSTMModel(input_dim),
        "Transformer"      : lambda: TransformerOnly(input_dim),
        "Transformer-LSTM" : lambda: TransformerLSTM(input_dim),
    }
