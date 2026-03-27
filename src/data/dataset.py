# ============================================================
# src/data/dataset.py  —  PyTorch Dataset + DataLoaders
# ============================================================

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler

from configs.config import SEQ_LEN, BATCH_SIZE, TRAIN_RATIO, VAL_RATIO, TARGET


class EVSeqDataset(Dataset):
    """Sliding-window sequence dataset for EV load forecasting."""

    def __init__(self, sequences: list):
        self.X = torch.tensor(
            np.array([s[0] for s in sequences]), dtype=torch.float32)
        self.y = torch.tensor(
            np.array([s[1] for s in sequences]), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def make_sequences(feat: np.ndarray, tgt: np.ndarray, seq_len: int = SEQ_LEN) -> list:
    """Create (X_window, y_next) pairs."""
    return [(feat[i - seq_len:i], tgt[i])
            for i in range(seq_len, len(feat))]


def build_dataloaders(daily, feature_cols: list, seq_len: int = SEQ_LEN,
                      batch_size: int = BATCH_SIZE):
    """
    Build per-location scaled sequences and return DataLoaders.

    Returns
    -------
    train_loader, val_loader, test_loader,
    loc_list_sorted, loc_scalers, test_counts
    """
    loc_list_sorted = sorted(daily["Location"].unique())
    loc_scalers     = {}
    train_seqs, val_seqs, test_seqs = [], [], []
    test_counts = {}

    for loc in loc_list_sorted:
        sub = (daily[daily["Location"] == loc]
               .sort_values("Date")
               .reset_index(drop=True))
        n    = len(sub)
        n_tr = int(n * TRAIN_RATIO)
        n_vl = int(n * VAL_RATIO)

        tr = sub.iloc[:n_tr]
        vl = sub.iloc[n_tr:n_vl]
        te = sub.iloc[n_vl:]

        # Feature scaling
        f_sc = RobustScaler()
        X_tr = f_sc.fit_transform(tr[feature_cols].values.astype(np.float32))
        X_vl = f_sc.transform(vl[feature_cols].values.astype(np.float32))
        X_te = f_sc.transform(te[feature_cols].values.astype(np.float32))

        # Target scaling
        t_sc = RobustScaler()
        y_tr = t_sc.fit_transform(tr[[TARGET]].values.astype(np.float32)).flatten()
        y_vl = t_sc.transform(vl[[TARGET]].values.astype(np.float32)).flatten()
        y_te = t_sc.transform(te[[TARGET]].values.astype(np.float32)).flatten()
        loc_scalers[loc] = t_sc

        train_seqs += make_sequences(X_tr, y_tr, seq_len)
        val_seqs   += make_sequences(X_vl, y_vl, seq_len)
        test_seqs  += make_sequences(X_te, y_te, seq_len)
        test_counts[loc] = max(0, len(te) - seq_len)

    train_loader = DataLoader(EVSeqDataset(train_seqs),
                              batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(EVSeqDataset(val_seqs),
                              batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(EVSeqDataset(test_seqs),
                              batch_size=batch_size, shuffle=False)

    print(f"[DataLoader] Train: {len(train_seqs):,}  "
          f"Val: {len(val_seqs):,}  Test: {len(test_seqs):,}")

    return (train_loader, val_loader, test_loader,
            loc_list_sorted, loc_scalers, test_counts)
