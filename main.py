"""
============================================================
Spatio-Temporal Forecasting of Municipal EV Charging Load
MODEL COMPARISON  v3
============================================================
Models compared:
  1. Simple RNN
  2. LSTM
  3. Transformer (encoder-only)
  4. CNN-LSTM
  5. Transformer-LSTM  ← proposed
============================================================
"""

# ── 0. IMPORTS ───────────────────────────────────────────────────────────────
import warnings, math, random
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import norm as sp_norm

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*65}")
print(f"  EV Load Forecasting — Multi-Model Comparison  (v3)")
print(f"{'='*65}")
print(f"  Device : {DEVICE}")
print(f"{'='*65}\n")

# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA  (same pipeline as v2)
# ═══════════════════════════════════════════════════════════════════════════
print("[ 1/4 ]  Loading & Preprocessing …")

df = pd.read_csv("Data/EV.csv")
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])

df["month"]       = df["Date"].dt.month
df["weekday"]     = df["Date"].dt.weekday
df["is_weekend"]  = (df["weekday"] >= 5).astype(int)
df["quarter"]     = df["Date"].dt.quarter
df["month_sin"]   = np.sin(2*np.pi*df["month"]/12)
df["month_cos"]   = np.cos(2*np.pi*df["month"]/12)
df["weekday_sin"] = np.sin(2*np.pi*df["weekday"]/7)
df["weekday_cos"] = np.cos(2*np.pi*df["weekday"]/7)

LOC_MAP = {
    "CSQ - Court Square Municipal Parking Garage"          : "Court Square",
    "Court Square Municipal Parking Garage"                : "Court Square",
    "QBO - Queens Borough Hall Municipal Parking Garage"   : "Queens Borough Hall",
    "Queens Borough Hall Municipal Parking Garage"         : "Queens Borough Hall",
    "Queensboro Hall"                                      : "Queens Borough Hall",
    "JON - Jerome 190th Street Municipal Parking Garage"   : "Jerome 190th",
    "JON - Jerome 190th Street Municipal Parking"          : "Jerome 190th",
    "Jerome 190th Street Municipal Parking"                : "Jerome 190th",
    "DES - Delancey and Essex Municipal Parking Garage"    : "Delancey Essex",
    "Delancey and Essex Municipal Parking Garage"          : "Delancey Essex",
    "BRI - Bay Ridge Municipal Parking Garage"             : "Bay Ridge",
    "Bay Ridge Municipal Parking Garage"                   : "Bay Ridge",
    "JGU - Jerome Gun Hill Road Municipal Parking Garage"  : "Jerome Gun Hill",
    "Jerome-Gun Hill Road Municipal Parking Garage"        : "Jerome Gun Hill",
    "SGE - St. George Courthouse Municipal Parking Garage" : "St George",
    "SGE - St. George Courthouse Garage"                   : "St George",
    "St. George Courthouse"                                : "St George",
    "QFA - Queens Family Court Municipal Garage"           : "Queens Family Court",
    "Queens Family Court Municipal Garage"                 : "Queens Family Court",
}
df["Location"] = df["Location Name"].map(LOC_MAP).fillna(df["Location Name"])

WEATHER = ["tmpf","relh","feel","sped","p01m","snowdepth"]
df.sort_values(["weather_station","Date"], inplace=True)
df[WEATHER] = (df.groupby("weather_station")[WEATHER]
                 .transform(lambda x: x.ffill().bfill()))
df[WEATHER] = df[WEATHER].fillna(df[WEATHER].median())

AGG = {
    "Energy Provided (kWh)"   : "sum",
    "Charge Duration (min)"   : ["mean","count"],
    "tmpf":"mean","relh":"mean","feel":"mean",
    "sped":"mean","p01m":"sum","snowdepth":"max",
    "is_weekend":"first","month_sin":"first","month_cos":"first",
    "weekday_sin":"first","weekday_cos":"first","quarter":"first",
}
daily = df.groupby(["Date","Location"]).agg(AGG)
daily.columns = ["load_kwh","charge_dur_mean","session_count",
                 "tmpf","relh","feel","sped","p01m","snowdepth",
                 "is_weekend","month_sin","month_cos",
                 "weekday_sin","weekday_cos","quarter"]
daily = daily.reset_index().sort_values(["Location","Date"]).reset_index(drop=True)

q_lo = daily["load_kwh"].quantile(0.005)
q_hi = daily["load_kwh"].quantile(0.995)
daily = daily[(daily["load_kwh"] >= q_lo) & (daily["load_kwh"] <= q_hi)].copy()

counts = daily.groupby("Location").size()
valid_locs = counts[counts >= 100].index.tolist()
daily = daily[daily["Location"].isin(valid_locs)].copy()

BASE_FEATS = ["tmpf","relh","feel","sped","p01m","snowdepth",
              "charge_dur_mean","session_count",
              "is_weekend","month_sin","month_cos",
              "weekday_sin","weekday_cos","quarter"]

for lag in [1,2,3,7,14]:
    col = f"load_lag{lag}"
    daily[col] = daily.groupby("Location")["load_kwh"].shift(lag)
    BASE_FEATS.append(col)

for win in [7,14]:
    col  = f"load_roll{win}_mean"
    col2 = f"load_roll{win}_std"
    daily[col]  = daily.groupby("Location")["load_kwh"].transform(
        lambda x: x.shift(1).rolling(win, min_periods=1).mean())
    daily[col2] = daily.groupby("Location")["load_kwh"].transform(
        lambda x: x.shift(1).rolling(win, min_periods=1).std().fillna(0))
    BASE_FEATS += [col, col2]

daily.dropna(inplace=True)
daily.reset_index(drop=True, inplace=True)

FEATURE_COLS = BASE_FEATS
TARGET       = "load_kwh"
INPUT_DIM    = len(FEATURE_COLS)
SEQ_LEN      = 21
BATCH_SIZE   = 32

print(f"   Records : {len(daily):,}  |  Locations : {daily['Location'].nunique()}")
print(f"   Features: {INPUT_DIM}")

# ── Dataset helper ────────────────────────────────────────────────────────────
class EVSeqDataset(Dataset):
    def __init__(self, seqs):
        self.X = torch.tensor(np.array([s[0] for s in seqs]), dtype=torch.float32)
        self.y = torch.tensor(np.array([s[1] for s in seqs]), dtype=torch.float32)
    def __len__(self):  return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

def make_sequences(feat, tgt, seq_len):
    seqs = []
    for i in range(seq_len, len(feat)):
        seqs.append((feat[i-seq_len:i], tgt[i]))
    return seqs

# ── Build per-location splits (reuse for all models) ─────────────────────────
loc_list_sorted = sorted(daily["Location"].unique())
loc_scalers     = {}
train_seqs_all, val_seqs_all, test_seqs_all = [], [], []
test_counts = {}

for loc in loc_list_sorted:
    sub  = daily[daily["Location"]==loc].sort_values("Date").reset_index(drop=True)
    n    = len(sub)
    n_tr = int(n*0.70); n_vl = int(n*0.85)
    tr, vl, te = sub.iloc[:n_tr], sub.iloc[n_tr:n_vl], sub.iloc[n_vl:]

    f_sc = StandardScaler()
    X_tr = f_sc.fit_transform(tr[FEATURE_COLS].values.astype(np.float32))
    X_vl = f_sc.transform(vl[FEATURE_COLS].values.astype(np.float32))
    X_te = f_sc.transform(te[FEATURE_COLS].values.astype(np.float32))

    t_sc = StandardScaler()
    y_tr = t_sc.fit_transform(tr[[TARGET]].values.astype(np.float32)).flatten()
    y_vl = t_sc.transform(vl[[TARGET]].values.astype(np.float32)).flatten()
    y_te = t_sc.transform(te[[TARGET]].values.astype(np.float32)).flatten()
    loc_scalers[loc] = t_sc

    train_seqs_all += make_sequences(X_tr, y_tr, SEQ_LEN)
    val_seqs_all   += make_sequences(X_vl, y_vl, SEQ_LEN)
    test_seqs_all  += make_sequences(X_te, y_te, SEQ_LEN)
    test_counts[loc] = max(0, len(te) - SEQ_LEN)

train_ds = EVSeqDataset(train_seqs_all)
val_ds   = EVSeqDataset(val_seqs_all)
test_ds  = EVSeqDataset(test_seqs_all)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

print(f"   Train:{len(train_ds):,}  Val:{len(val_ds):,}  Test:{len(test_ds):,}")

# ═══════════════════════════════════════════════════════════════════════════
# 2. MODEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════
print("\n[ 2/4 ]  Defining models …")

# ── Helper: regression head ───────────────────────────────────────────────────
def mlp_head(in_dim, dropout=0.15):
    return nn.Sequential(
        nn.Linear(in_dim, 64), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(64, 32),     nn.GELU(),
        nn.Linear(32, 1))

# ── 1. Simple RNN ─────────────────────────────────────────────────────────────
class SimpleRNN(nn.Module):
    def __init__(self, input_dim, hidden=128, layers=2, dropout=0.15):
        super().__init__()
        self.rnn  = nn.RNN(input_dim, hidden, layers,
                           batch_first=True, dropout=dropout, nonlinearity="tanh")
        self.norm = nn.LayerNorm(hidden)
        self.head = mlp_head(hidden, dropout)
    def forward(self, x):
        o, _ = self.rnn(x)
        return self.head(self.norm(o[:,-1,:])).squeeze(-1)

# ── 2. LSTM ───────────────────────────────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden=128, layers=2, dropout=0.15):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers,
                            batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(hidden)
        self.head = mlp_head(hidden, dropout)
    def forward(self, x):
        o, _ = self.lstm(x)
        return self.head(self.norm(o[:,-1,:])).squeeze(-1)

# ── 3. Transformer (encoder-only) ─────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(10000.)/d_model))
        pe[:,0::2] = torch.sin(pos*div)
        pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return self.drop(x + self.pe[:,:x.size(1)])

class TransformerOnly(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, n_enc=4, dropout=0.15):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pe   = PositionalEncoding(d_model, dropout=dropout)
        enc = nn.TransformerEncoderLayer(d_model, nhead,
              dim_feedforward=d_model*4, dropout=dropout,
              batch_first=True, norm_first=True)
        self.enc  = nn.TransformerEncoder(enc, num_layers=n_enc)
        self.norm = nn.LayerNorm(d_model)
        self.head = mlp_head(d_model, dropout)
    def forward(self, x):
        x = self.pe(self.proj(x))
        x = self.enc(x)
        return self.head(self.norm(x[:,-1,:])).squeeze(-1)

# ── 4. CNN-LSTM ───────────────────────────────────────────────────────────────
class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim, cnn_ch=64, kernel=3,
                 lstm_h=128, lstm_l=2, dropout=0.15):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_ch, kernel, padding=kernel//2),
            nn.BatchNorm1d(cnn_ch), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(cnn_ch, cnn_ch*2, kernel, padding=kernel//2),
            nn.BatchNorm1d(cnn_ch*2), nn.GELU(), nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(cnn_ch*2, lstm_h, lstm_l,
                            batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(lstm_h)
        self.head = mlp_head(lstm_h, dropout)
    def forward(self, x):                          # x:(B,T,F)
        x = self.cnn(x.permute(0,2,1))             # (B, cnn_ch*2, T)
        x = x.permute(0,2,1)                       # (B, T, cnn_ch*2)
        o, _ = self.lstm(x)
        return self.head(self.norm(o[:,-1,:])).squeeze(-1)

# ── 5. Transformer-LSTM (proposed) ────────────────────────────────────────────
class TransformerLSTM(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8,
                 n_enc=4, lstm_h=128, lstm_l=2, dropout=0.15):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pe   = PositionalEncoding(d_model, dropout=dropout)
        enc = nn.TransformerEncoderLayer(d_model, nhead,
              dim_feedforward=d_model*4, dropout=dropout,
              batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=n_enc)
        self.lstm = nn.LSTM(d_model, lstm_h, lstm_l,
                            batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(lstm_h)
        self.head = mlp_head(lstm_h, dropout)
    def forward(self, x):
        x = self.pe(self.proj(x))
        x = self.transformer(x)
        o, _ = self.lstm(x)
        return self.head(self.norm(o[:,-1,:])).squeeze(-1)

# ── Model registry ────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "Simple RNN"       : lambda: SimpleRNN(INPUT_DIM),
    "LSTM"             : lambda: LSTMModel(INPUT_DIM),
    "Transformer"      : lambda: TransformerOnly(INPUT_DIM),
    "CNN-LSTM"         : lambda: CNNLSTMModel(INPUT_DIM),
    "Transformer-LSTM" : lambda: TransformerLSTM(INPUT_DIM),
}

# ═══════════════════════════════════════════════════════════════════════════
# 3. TRAINING LOOP (shared)
# ═══════════════════════════════════════════════════════════════════════════
print("\n[ 3/4 ]  Training all models …\n")

EPOCHS   = 80
LR       = 3e-4
PATIENCE = 12

def compute_metrics(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    mae    = mean_absolute_error(y_true, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_true, y_pred))
    r2     = r2_score(y_true, y_pred)
    mask   = y_true > 1.0
    mape   = np.mean(np.abs((y_true[mask]-y_pred[mask]) / y_true[mask])) * 100
    return {"R2": r2, "MAE": mae, "RMSE": rmse, "MAPE": mape}

def inverse_test(preds_sc, trues_sc):
    all_p, all_t = [], []
    idx = 0
    for loc in loc_list_sorted:
        n_seq = test_counts[loc]
        if n_seq == 0: continue
        sc = loc_scalers[loc]
        p  = sc.inverse_transform(np.array(preds_sc[idx:idx+n_seq]).reshape(-1,1)).flatten()
        t  = sc.inverse_transform(np.array(trues_sc[idx:idx+n_seq]).reshape(-1,1)).flatten()
        all_p.extend(p); all_t.extend(t)
        idx += n_seq
    return np.array(all_t), np.array(all_p)

def run_epoch(model, loader, criterion, optimizer=None):
    if optimizer: model.train()
    else: model.eval()
    tot_loss = tot_mae = n = 0
    ctx = torch.enable_grad() if optimizer else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            if optimizer: optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            if optimizer:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            tot_loss += loss.item()*len(yb)
            tot_mae  += (pred-yb).abs().sum().item()
            n        += len(yb)
    return tot_loss/n, tot_mae/n

all_results  = {}   # model_name → metrics dict
all_history  = {}   # model_name → {loss,val_loss,...}
all_preds_inv = {}  # model_name → (y_true, y_pred) in kWh

for model_name, model_fn in MODEL_REGISTRY.items():
    print(f"\n{'━'*65}")
    print(f"  ▶  Training : {model_name}")
    print(f"{'━'*65}")

    torch.manual_seed(SEED)
    mdl = model_fn().to(DEVICE)
    n_p = sum(p.numel() for p in mdl.parameters() if p.requires_grad)
    print(f"     Parameters : {n_p:,}")

    criterion = nn.HuberLoss(delta=1.0)
    opt = torch.optim.AdamW(mdl.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)

    hist = {"loss":[],"mae":[],"val_loss":[],"val_mae":[]}
    best_vl, p_cnt, best_w = float("inf"), 0, None

    for ep in range(1, EPOCHS+1):
        tl, tm = run_epoch(mdl, train_loader, criterion, opt)
        vl, vm = run_epoch(mdl, val_loader,   criterion)
        sched.step()
        hist["loss"].append(tl); hist["mae"].append(tm)
        hist["val_loss"].append(vl); hist["val_mae"].append(vm)

        print(f"  {model_name} | Epoch {ep:>3} | "
              f"loss: {tl:.6f} | mae: {tm:.6f} | "
              f"val_loss: {vl:.6f} | val_mae: {vm:.6f}")

        if vl < best_vl:
            best_vl = vl
            best_w  = {k: v.cpu().clone() for k, v in mdl.state_dict().items()}
            p_cnt   = 0
        else:
            p_cnt += 1
            if p_cnt >= PATIENCE:
                print(f"  ⚡ Early stopping at epoch {ep}")
                break

    mdl.load_state_dict(best_w)
    all_history[model_name] = hist

    # ── Test evaluation ──────────────────────────────────────────────────────
    mdl.eval()
    preds_sc, trues_sc = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds_sc.extend(mdl(xb.to(DEVICE)).cpu().numpy())
            trues_sc.extend(yb.numpy())

    y_true, y_pred = inverse_test(preds_sc, trues_sc)
    metrics = compute_metrics(y_true, y_pred)
    all_results[model_name]   = metrics
    all_preds_inv[model_name] = (y_true, y_pred)

    print(f"\n  ✅ {model_name} → R²:{metrics['R2']:.4f}  "
          f"MAE:{metrics['MAE']:.2f}  RMSE:{metrics['RMSE']:.2f}  MAPE:{metrics['MAPE']:.2f}%")

# ═══════════════════════════════════════════════════════════════════════════
# 4. RESULTS TABLE + PLOTS
# ═══════════════════════════════════════════════════════════════════════════
print("\n[ 4/4 ]  Generating comparison table & plots …")

# ── Console table ─────────────────────────────────────────────────────────────
print(f"\n{'═'*70}")
print(f"  {'Model':<20} {'R²':>8} {'MAE':>10} {'RMSE':>10} {'MAPE (%)':>10}")
print(f"{'─'*70}")
for name, m in all_results.items():
    marker = "  ◀ PROPOSED" if name == "Transformer-LSTM" else ""
    print(f"  {name:<20} {m['R2']:>8.4f} {m['MAE']:>10.4f} {m['RMSE']:>10.4f} {m['MAPE']:>10.4f}{marker}")
print(f"{'═'*70}")

# ── Colors & model order ──────────────────────────────────────────────────────
MODEL_COLORS = {
    "Simple RNN"       : "#94A3B8",
    "LSTM"             : "#60A5FA",
    "Transformer"      : "#F59E0B",
    "CNN-LSTM"         : "#34D399",
    "Transformer-LSTM" : "#EF4444",
}
model_names = list(MODEL_REGISTRY.keys())

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(26, 20))
fig.suptitle(
    "EV Charging Load Forecasting — Multi-Model Comparison\n"
    "NYC Municipal Garages  |  Weather-Aware Spatio-Temporal Models",
    fontsize=16, fontweight="bold", y=1.005)

gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

# ── Row 0 : Loss curves for each model (3 × subplot) ─────────────────────────
ax_loss = [fig.add_subplot(gs[0, i]) for i in range(3)]
shown_models = ["LSTM", "Transformer", "Transformer-LSTM"]
for ax, mname in zip(ax_loss, shown_models):
    h = all_history[mname]
    ax.plot(h["loss"],     color="#2563EB", lw=2, label="Train")
    ax.plot(h["val_loss"], color="#DC2626", lw=2, ls="--", label="Val")
    ax.set_title(f"{mname} — Loss", fontweight="bold", fontsize=10)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Huber Loss")
    ax.legend(fontsize=8); ax.grid(alpha=0.4)

# ── Row 1 col 0 : R² bar ─────────────────────────────────────────────────────
ax_r2 = fig.add_subplot(gs[1, 0])
vals   = [all_results[m]["R2"] for m in model_names]
colors = [MODEL_COLORS[m] for m in model_names]
bars   = ax_r2.bar(model_names, vals, color=colors, edgecolor="white", lw=1.5)
for b, v in zip(bars, vals):
    ax_r2.text(b.get_x()+b.get_width()/2, b.get_height()+0.003,
               f"{v:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax_r2.set_title("R² Score  (higher ↑ better)", fontweight="bold")
ax_r2.set_ylim(0, 1.12); ax_r2.set_ylabel("R²")
ax_r2.set_xticklabels(model_names, rotation=20, ha="right", fontsize=9)
ax_r2.grid(alpha=0.3, axis="y")

# ── Row 1 col 1 : MAPE bar ────────────────────────────────────────────────────
ax_mape = fig.add_subplot(gs[1, 1])
vals_m  = [all_results[m]["MAPE"] for m in model_names]
bars2   = ax_mape.bar(model_names, vals_m, color=colors, edgecolor="white", lw=1.5)
for b, v in zip(bars2, vals_m):
    ax_mape.text(b.get_x()+b.get_width()/2, b.get_height()+0.2,
                 f"{v:.2f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax_mape.set_title("MAPE  (lower ↓ better)", fontweight="bold")
ax_mape.set_ylabel("MAPE (%)"); ax_mape.set_xticklabels(model_names, rotation=20, ha="right", fontsize=9)
ax_mape.grid(alpha=0.3, axis="y")

# ── Row 1 col 2 : MAE & RMSE grouped bar ─────────────────────────────────────
ax_err = fig.add_subplot(gs[1, 2])
x      = np.arange(len(model_names))
w      = 0.35
mae_v  = [all_results[m]["MAE"]  for m in model_names]
rmse_v = [all_results[m]["RMSE"] for m in model_names]
ax_err.bar(x - w/2, mae_v,  w, label="MAE",  color="#2563EB", alpha=0.85)
ax_err.bar(x + w/2, rmse_v, w, label="RMSE", color="#DC2626", alpha=0.85)
ax_err.set_xticks(x); ax_err.set_xticklabels(model_names, rotation=20, ha="right", fontsize=9)
ax_err.set_title("MAE & RMSE  (lower ↓ better)", fontweight="bold")
ax_err.set_ylabel("kWh"); ax_err.legend(); ax_err.grid(alpha=0.3, axis="y")

# ── Row 2 col 0 : Actual vs Predicted — Transformer-LSTM ─────────────────────
ax_avp = fig.add_subplot(gs[2, 0])
yt, yp = all_preds_inv["Transformer-LSTM"]
n_show = min(300, len(yt))
ax_avp.plot(yt[:n_show], color="#7C3AED", lw=1.5, label="Actual",    alpha=0.85)
ax_avp.plot(yp[:n_show], color="#059669", lw=1.5, ls="--", label="Predicted", alpha=0.85)
ax_avp.set_title("Transformer-LSTM: Actual vs Predicted", fontweight="bold")
ax_avp.set_xlabel("Test Sample"); ax_avp.set_ylabel("Load (kWh)")
ax_avp.legend(); ax_avp.grid(alpha=0.4)

# ── Row 2 col 1 : Scatter — Transformer-LSTM ─────────────────────────────────
ax_sc = fig.add_subplot(gs[2, 1])
ax_sc.scatter(yt, yp, alpha=0.25, s=8, color="#2563EB", edgecolors="none")
mn, mx = min(yt.min(), yp.min()), max(yt.max(), yp.max())
ax_sc.plot([mn,mx],[mn,mx],"r--",lw=2)
r2_p = all_results["Transformer-LSTM"]["R2"]
ax_sc.set_title(f"Transformer-LSTM Scatter  R²={r2_p:.4f}", fontweight="bold")
ax_sc.set_xlabel("Actual (kWh)"); ax_sc.set_ylabel("Predicted (kWh)")
ax_sc.grid(alpha=0.4)

# ── Row 2 col 2 : Radar / spider chart ───────────────────────────────────────
ax_rad = fig.add_subplot(gs[2, 2], polar=True)
metrics_radar = ["R²", "1-MAPE/100", "1-MAE_n", "1-RMSE_n"]

# Normalise (higher = better for all after transform)
mae_max  = max(all_results[m]["MAE"]  for m in model_names)
rmse_max = max(all_results[m]["RMSE"] for m in model_names)

def radar_vals(mname):
    r  = all_results[mname]
    return [
        r["R2"],
        max(0, 1 - r["MAPE"]/100),
        1 - r["MAE"] / mae_max,
        1 - r["RMSE"] / rmse_max,
    ]

N   = len(metrics_radar)
angles = [n / float(N) * 2 * math.pi for n in range(N)]
angles += angles[:1]

for mname in model_names:
    vals_r = radar_vals(mname)
    vals_r += vals_r[:1]
    ax_rad.plot(angles, vals_r, lw=2, color=MODEL_COLORS[mname], label=mname)
    ax_rad.fill(angles, vals_r, alpha=0.08, color=MODEL_COLORS[mname])

ax_rad.set_xticks(angles[:-1])
ax_rad.set_xticklabels(["R²", "1-MAPE", "1-MAE_n", "1-RMSE_n"], fontsize=9)
ax_rad.set_title("Model Comparison Radar", fontweight="bold", pad=15)
ax_rad.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)

plt.savefig("EV_model_comparison_v3.png", dpi=150, bbox_inches="tight")
plt.show()
print("   Plot saved → EV_model_comparison_v3.png")

# ── Paper-ready table (DataFrame) ────────────────────────────────────────────
df_res = pd.DataFrame(all_results).T.round(4)
df_res.index.name = "Model"
df_res.columns    = ["R² Score", "MAE (kWh)", "RMSE (kWh)", "MAPE (%)"]
df_res["Rank"]    = df_res["R² Score"].rank(ascending=False).astype(int)
df_res.sort_values("Rank", inplace=True)
print(f"\n{'═'*60}")
print("  PAPER-READY COMPARISON TABLE")
print(f"{'═'*60}")
print(df_res.to_string())
print(f"{'═'*60}")
df_res.to_csv("model_comparison_results.csv")
print("  Table saved → model_comparison_results.csv")
print("\n  ✅  All models trained and compared!")