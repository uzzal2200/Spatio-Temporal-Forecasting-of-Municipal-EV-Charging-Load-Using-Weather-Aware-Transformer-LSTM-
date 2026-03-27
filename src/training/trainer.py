# ============================================================
# src/training/trainer.py  —  Training & Evaluation Engine
# ============================================================

import numpy as np
import torch
import torch.nn as nn

from configs.config import EPOCHS, LR, SEED, DEVICE
from src.evaluation.metrics import compute_metrics


# ── Single Epoch ─────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer=None):
    """Run one train or eval epoch. Returns (loss, mae)."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, total_mae, n = 0.0, 0.0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()

    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred   = model(xb)
            loss   = criterion(pred, yb)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            bs = len(yb)
            total_loss += loss.item() * bs
            total_mae  += torch.abs(pred.detach() - yb).sum().item()
            n          += bs

    return total_loss / n, total_mae / n


def inverse_test(preds_sc, trues_sc, loc_list_sorted, loc_scalers, test_counts):
    """Inverse-transform scaled predictions back to kWh."""
    all_p, all_t = [], []
    idx = 0
    for loc in loc_list_sorted:
        n_seq = test_counts[loc]
        if n_seq == 0:
            continue
        sc = loc_scalers[loc]
        p  = sc.inverse_transform(
            np.array(preds_sc[idx: idx + n_seq]).reshape(-1, 1)).flatten()
        t  = sc.inverse_transform(
            np.array(trues_sc[idx: idx + n_seq]).reshape(-1, 1)).flatten()
        all_p.extend(p)
        all_t.extend(t)
        idx += n_seq
    return np.array(all_t), np.array(all_p)


# ── Full Training Loop ────────────────────────────────────────────────────────

def train_model(model_name: str, model_fn, train_loader, val_loader,
                test_loader, loc_list_sorted, loc_scalers, test_counts,
                epochs: int = EPOCHS, lr: float = LR):
    """
    Train one model and return (metrics_dict, history_dict, (y_true, y_pred)).
    """
    torch.manual_seed(SEED)
    model = model_fn().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'━'*60}")
    print(f"  ▶  {model_name}   |   Params: {n_params:,}")
    print(f"{'━'*60}")

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)

    history = {"loss": [], "mae": [], "val_loss": [], "val_mae": []}
    best_val, best_weights = float("inf"), None

    for ep in range(1, epochs + 1):
        tl, tm = run_epoch(model, train_loader, criterion, optimizer)
        vl, vm = run_epoch(model, val_loader,   criterion)
        scheduler.step()

        history["loss"].append(tl);     history["mae"].append(tm)
        history["val_loss"].append(vl); history["val_mae"].append(vm)

        print(f"  Epoch {ep:>3} | loss: {tl:.6f} | mae: {tm:.6f} "
              f"| val_loss: {vl:.6f} | val_mae: {vm:.6f}")

        if vl < best_val:
            best_val     = vl
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_weights)
    model.eval()

    preds_sc, trues_sc = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds_sc.extend(model(xb.to(DEVICE)).cpu().numpy())
            trues_sc.extend(yb.numpy())

    y_true, y_pred = inverse_test(
        preds_sc, trues_sc, loc_list_sorted, loc_scalers, test_counts)
    metrics = compute_metrics(y_true, y_pred)

    print(f"  → R²:{metrics['R2']:.4f}  MAE:{metrics['MAE']:.4f}  "
          f"RMSE:{metrics['RMSE']:.4f}  MAPE:{metrics['MAPE']:.2f}%  "
          f"sMAPE:{metrics['sMAPE']:.2f}%  PSNR:{metrics['PSNR']:.2f}dB")

    return metrics, history, (y_true, y_pred), model


def train_all_models(model_registry: dict, train_loader, val_loader,
                     test_loader, loc_list_sorted, loc_scalers, test_counts):
    """Train all models in registry and return results."""
    all_results, all_history, all_preds_inv, trained_models = {}, {}, {}, {}

    for name, fn in model_registry.items():
        metrics, history, preds, model = train_model(
            name, fn, train_loader, val_loader, test_loader,
            loc_list_sorted, loc_scalers, test_counts,
        )
        all_results[name]    = metrics
        all_history[name]    = history
        all_preds_inv[name]  = preds
        trained_models[name] = model

    # Print summary table
    print(f"\n{'═'*90}")
    print(f"  {'Model':<22} {'R²':>7} {'MAE':>9} {'RMSE':>9} {'MAPE%':>8} {'sMAPE%':>9} {'PSNR(dB)':>10}")
    print(f"{'─'*90}")
    for name, m in all_results.items():
        tag = "  ◀ PROPOSED" if name == "Transformer-LSTM" else ""
        print(f"  {name:<22} {m['R2']:>7.4f} {m['MAE']:>9.4f} "
              f"{m['RMSE']:>9.4f} {m['MAPE']:>7.2f} {m['sMAPE']:>9.4f} {m['PSNR']:>10.4f}{tag}")
    print(f"{'═'*90}")

    return all_results, all_history, all_preds_inv, trained_models
