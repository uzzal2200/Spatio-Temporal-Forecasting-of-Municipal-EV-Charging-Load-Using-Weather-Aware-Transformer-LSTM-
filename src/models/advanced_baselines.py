import math
from dataclasses import dataclass

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


def _regression_head(in_dim: int, dropout: float = 0.1) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, in_dim // 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(in_dim // 2, 1),
    )


class InformerBaseline(nn.Module):
    """Lightweight Informer-style baseline with sparse-like distillation."""

    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 8, n_enc: int = 2, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pe = PositionalEncoding(d_model, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_enc)
        self.distill = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.head = _regression_head(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pe(self.proj(x))
        x = self.encoder(x)
        x = self.distill(x.transpose(1, 2)).transpose(1, 2)
        return self.head(x[:, -1, :]).squeeze(-1)


class PatchTSTBaseline(nn.Module):
    """PatchTST-like baseline with channel-last patch tokenization."""

    def __init__(
        self,
        input_dim: int,
        seq_len: int = 21,
        patch_len: int = 3,
        stride: int = 3,
        d_model: int = 128,
        nhead: int = 8,
        n_enc: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.seq_len = seq_len
        self.patch_proj = nn.Linear(input_dim * patch_len, d_model)
        self.pe = PositionalEncoding(d_model, max_len=256, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_enc)
        self.head = _regression_head(d_model, dropout)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, feat = x.shape
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.contiguous().view(bsz, patches.size(1), self.patch_len * feat)
        return patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self._patchify(x)
        p = self.pe(self.patch_proj(p))
        p = self.encoder(p)
        return self.head(p[:, -1, :]).squeeze(-1)


class TFTBaseline(nn.Module):
    """Compact TFT-style baseline with variable gating + recurrent encoder."""

    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 8, lstm_hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.var_gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid(),
        )
        self.input_proj = nn.Linear(input_dim, d_model)
        self.lstm = nn.LSTM(d_model, lstm_hidden, num_layers=2, dropout=dropout, batch_first=True)
        self.attn = nn.MultiheadAttention(embed_dim=lstm_hidden, num_heads=nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, lstm_hidden),
        )
        self.norm = nn.LayerNorm(lstm_hidden)
        self.head = _regression_head(lstm_hidden, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.var_gate(x)
        x = self.input_proj(x * g)
        x, _ = self.lstm(x)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + self.ff(attn_out))
        return self.head(x[:, -1, :]).squeeze(-1)


@dataclass(frozen=True)
class AdvancedBaselineSpec:
    name: str
    params_k: float


ADVANCED_BASELINE_SPECS = [
    AdvancedBaselineSpec("Informer", 285.00),
    AdvancedBaselineSpec("PatchTST", 178.00),
    AdvancedBaselineSpec("TFT", 412.00),
]


def build_advanced_baseline_registry(input_dim: int, seq_len: int = 21) -> dict:
    return {
        "Informer": lambda: InformerBaseline(input_dim=input_dim),
        "PatchTST": lambda: PatchTSTBaseline(input_dim=input_dim, seq_len=seq_len),
        "TFT": lambda: TFTBaseline(input_dim=input_dim),
    }
