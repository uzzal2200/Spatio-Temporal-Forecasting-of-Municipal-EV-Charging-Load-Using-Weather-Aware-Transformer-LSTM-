import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


BASELINE_COLOR = "#5b8dd9"
PROPOSED_COLOR = "#e84c3d"
PROPOSED_MODEL = "Transformer-LSTM"


def _save_show(save_path: str | None = None):
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_loss_curves(all_history: dict, save_path: str | None = None):
    models = list(all_history.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), dpi=150)
    if len(models) == 1:
        axes = [axes]

    for ax, name in zip(axes, models):
        h = all_history[name]
        ax.plot(h["loss"], label="Train")
        ax.plot(h["val_loss"], label="Val")
        ax.set_title(f"{name} - Loss")
        ax.legend()
        ax.grid(alpha=0.4)

    _save_show(save_path)


def plot_r2_comparison(all_results: dict, save_path: str | None = None):
    names = list(all_results.keys())
    vals = [all_results[m]["R2"] for m in names]
    colors = [PROPOSED_COLOR if m == PROPOSED_MODEL else BASELINE_COLOR for m in names]

    plt.figure(figsize=(10, 6), dpi=150)
    bars = plt.bar(names, vals, color=colors)
    plt.title("R2 Score Comparison", fontsize=13)
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0, 1.1)
    plt.grid(axis="y", alpha=0.3)

    for bar, v in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.4f}", ha="center", fontsize=9)

    legend = [
        mpatches.Patch(color=BASELINE_COLOR, label="Baseline"),
        mpatches.Patch(color=PROPOSED_COLOR, label=f"Proposed ({PROPOSED_MODEL})"),
    ]
    plt.legend(handles=legend)
    _save_show(save_path)


def plot_mape_comparison(all_results: dict, save_path: str | None = None):
    names = list(all_results.keys())
    vals = [all_results[m]["MAPE"] for m in names]
    colors = [PROPOSED_COLOR if m == PROPOSED_MODEL else BASELINE_COLOR for m in names]

    plt.figure(figsize=(10, 6), dpi=150)
    bars = plt.bar(names, vals, color=colors)
    plt.title("MAPE Comparison", fontsize=13)
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.3)

    for bar, v in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 0.2, f"{v:.2f}%", ha="center", fontsize=9)

    legend = [
        mpatches.Patch(color=BASELINE_COLOR, label="Baseline"),
        mpatches.Patch(color=PROPOSED_COLOR, label=f"Proposed ({PROPOSED_MODEL})"),
    ]
    plt.legend(handles=legend)
    _save_show(save_path)


def plot_mae_rmse(all_results: dict, save_path: str | None = None):
    model_names = list(all_results.keys())
    mae_v = [all_results[m]["MAE"] for m in model_names]
    rmse_v = [all_results[m]["RMSE"] for m in model_names]

    c_mae = "#2166AC"
    c_rmse = "#1B7837"
    c_prop = "#C0392B"
    c_prop_rmse = "#E07B3A"

    x = np.arange(len(model_names))
    w = 0.28
    gap = 0.06

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=150)

    for i, (mv, rv) in enumerate(zip(mae_v, rmse_v)):
        is_p = model_names[i] == PROPOSED_MODEL
        mc = c_prop if is_p else c_mae
        rc = c_prop_rmse if is_p else c_rmse

        ax.bar(x[i] - w / 2 - gap / 2, mv, w, color=mc, edgecolor="black", linewidth=0.6, alpha=0.92, zorder=3)
        ax.bar(x[i] + w / 2 + gap / 2, rv, w, color=rc, edgecolor="black", linewidth=0.6, hatch="////", alpha=0.88, zorder=3)

    ax.plot(x - w / 2 - gap / 2, mae_v, color=c_mae, lw=1.4, linestyle=(0, (5, 3)), marker="o", zorder=7)
    ax.plot(x + w / 2 + gap / 2, rmse_v, color=c_rmse, lw=1.4, linestyle=(0, (5, 3)), marker="s", zorder=7)

    p_idx = model_names.index(PROPOSED_MODEL)
    ax.axvspan(p_idx - 0.46, p_idx + 0.46, color="#FDECEA", alpha=1.0, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=20, ha="right")
    ax.set_ylabel("Error (kWh)")
    ax.grid(axis="y", alpha=0.3)

    legend_handles = [
        mpatches.Patch(facecolor=c_mae, edgecolor="black", label="MAE"),
        mpatches.Patch(facecolor=c_rmse, edgecolor="black", hatch="////", label="RMSE"),
        mpatches.Patch(facecolor=c_prop, edgecolor="black", label="Proposed MAE"),
        mpatches.Patch(facecolor=c_prop_rmse, edgecolor="black", hatch="////", label="Proposed RMSE"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9)
    _save_show(save_path)


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = PROPOSED_MODEL,
    n_show: int = 400,
    save_path: str | None = None,
):
    n = min(n_show, len(y_true))
    plt.figure(figsize=(12, 4), dpi=150)
    plt.plot(y_true[:n], label="Actual", linewidth=2)
    plt.plot(y_pred[:n], linestyle="--", label="Predicted", linewidth=2)
    plt.title(f"{model_name}: Actual vs Predicted")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Load (kWh)")
    plt.legend()
    plt.grid(alpha=0.3)
    _save_show(save_path)


def plot_all_models_overlay(all_preds_inv: dict, n_show: int = 200, save_path: str | None = None):
    plt.figure(figsize=(14, 8), dpi=150)
    model_styles = {
        "Simple RNN": {"color": "#1f77b4", "lw": 1.2, "ls": "--", "alpha": 0.85},
        "LSTM": {"color": "#f1c40f", "lw": 1.2, "ls": "--", "alpha": 0.85},
        "Transformer": {"color": "#e8000d", "lw": 1.2, "ls": "--", "alpha": 0.85},
        "Transformer-LSTM": {"color": "#000000", "lw": 2.0, "ls": "-", "alpha": 1.0},
    }

    actual_ref = None
    for name, (yt, yp) in all_preds_inv.items():
        style = model_styles.get(name, {"color": "#666666", "lw": 1.2, "ls": "--", "alpha": 0.8})
        plt.plot(yp[:n_show], label=f"{name} Pred", color=style["color"], linewidth=style["lw"], linestyle=style["ls"], alpha=style["alpha"])
        if actual_ref is None:
            actual_ref = yt

    if actual_ref is not None:
        plt.plot(actual_ref[:n_show], label="Actual", color="#27ae60", linewidth=2.2)

    plt.xlabel("Time Step")
    plt.ylabel("Load (kWh)")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    _save_show(save_path)


def plot_scatter_all(all_preds_inv: dict, save_path: str | None = None):
    names = list(all_preds_inv.keys())
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=150)
    axes = axes.flatten()

    for ax, name in zip(axes, names):
        yt, yp = all_preds_inv[name]
        ax.scatter(yt, yp, alpha=0.25)
        mn, mx = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax.plot([mn, mx], [mn, mx], "r--")
        ax.set_title(name)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.grid(alpha=0.3)

    _save_show(save_path)


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = PROPOSED_MODEL, save_path: str | None = None):
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    axes[0].scatter(y_pred, residuals, alpha=0.3)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_title(f"Residual Plot - {model_name}")
    axes[0].set_xlabel("Predicted Load (kWh)")
    axes[0].set_ylabel("Residual (Actual - Predicted)")
    axes[0].grid(alpha=0.3)

    count, bins, _ = axes[1].hist(residuals, bins=40, density=True, alpha=0.6)
    mean, std = np.mean(residuals), np.std(residuals)
    x = np.linspace(bins.min(), bins.max(), 200)
    pdf = (1.0 / (std * np.sqrt(2.0 * np.pi))) * np.exp(-(x - mean) ** 2 / (2.0 * std ** 2))
    axes[1].plot(x, pdf)
    axes[1].set_title("Residual Error Distribution")
    axes[1].set_xlabel("Prediction Error (kWh)")
    axes[1].set_ylabel("Density")
    axes[1].grid(alpha=0.3)

    _save_show(save_path)


def plot_temporal_scales(y_true: np.ndarray, y_pred: np.ndarray, daily_dates: pd.Series, save_path: str | None = None):
    df = pd.DataFrame({
        "Date": pd.to_datetime(daily_dates[-len(y_true):].values),
        "Actual": y_true,
        "Predicted": y_pred,
    }).set_index("Date")

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), dpi=150)

    scales = [
        ("Daily", df),
        ("Weekly", df.resample("W").sum()),
        ("Monthly", df.resample("M").sum()),
        ("Yearly", df.resample("Y").sum()),
    ]

    for ax, (label, data) in zip(axes, scales):
        ax.plot(data["Actual"], label="Actual")
        ax.plot(data["Predicted"], label="Predicted")
        ax.set_title(f"{label} Load Forecast")
        ax.legend()
        ax.grid(alpha=0.3)

    _save_show(save_path)


def plot_uncertainty_intervals(y_true: np.ndarray, y_pred: np.ndarray, n_show: int = 150, save_path: str | None = None):
    residuals = y_true - y_pred
    n_show = min(n_show, len(y_true))

    lo80 = y_pred[:n_show] + np.percentile(residuals, 10)
    hi80 = y_pred[:n_show] + np.percentile(residuals, 90)
    lo95 = y_pred[:n_show] + np.percentile(residuals, 2.5)
    hi95 = y_pred[:n_show] + np.percentile(residuals, 97.5)

    lo80 = np.clip(lo80, 0, None)
    lo95 = np.clip(lo95, 0, None)

    plt.figure(figsize=(14, 5), dpi=150)
    plt.fill_between(range(n_show), lo95, hi95, alpha=0.15, color=PROPOSED_COLOR, label="95% PI")
    plt.fill_between(range(n_show), lo80, hi80, alpha=0.30, color=PROPOSED_COLOR, label="80% PI")
    plt.plot(y_true[:n_show], color="black", lw=1.5, label="Actual")
    plt.plot(y_pred[:n_show], color=PROPOSED_COLOR, lw=1.5, linestyle="--", label="Predicted")
    plt.ylim(bottom=0)
    plt.xlabel("Time Step")
    plt.ylabel("Energy Provided (kWh)")
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)

    _save_show(save_path)


def plot_peak_load_comparison(y_true: np.ndarray, y_pred: np.ndarray, save_path: str | None = None):
    actual_peak = float(np.max(y_true))
    pred_peak = float(np.max(y_pred))

    plt.figure(figsize=(6, 4), dpi=150)
    plt.bar(["Actual Peak", "Predicted Peak"], [actual_peak, pred_peak], color=["#2166AC", "#E07B3A"])
    plt.ylabel("Load (kWh)")
    plt.grid(axis="y", alpha=0.3)

    _save_show(save_path)


def plot_model_radar(all_results: dict, metrics_cols: list[str] | None = None, save_path: str | None = None):
    if metrics_cols is None:
        metrics_cols = ["RMSE", "MAE", "MAPE"]

    df = pd.DataFrame(all_results).T.reset_index().rename(columns={"index": "Model"})
    normalized = df.copy()

    for col in metrics_cols:
        col_min = normalized[col].min()
        col_max = normalized[col].max()
        if col_max == col_min:
            normalized[col] = 0.5
        else:
            normalized[col] = (normalized[col] - col_min) / (col_max - col_min)

    epsilon = 0.05
    normalized[metrics_cols] = normalized[metrics_cols] + epsilon

    labels = metrics_cols
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = plt.subplot(111, polar=True)

    palette = ["#1f77b4", "#f1c40f", "#e8000d", "#000000"]
    for i, (_, row) in enumerate(normalized.iterrows()):
        values = row[metrics_cols].tolist()
        values += values[:1]
        color = palette[i % len(palette)]
        ax.plot(angles, values, linewidth=2, label=row["Model"], color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    _save_show(save_path)


def plot_best_prediction_region(y_true: np.ndarray, y_pred: np.ndarray, window: int = 50, save_path: str | None = None):
    error = np.abs(y_true - y_pred)
    if len(error) < window + 1:
        window = max(5, len(error) // 2)

    rolling_err = np.convolve(error, np.ones(window) / window, mode="valid")
    best_start = int(np.argmin(rolling_err))
    best_end = best_start + window

    plt.figure(figsize=(12, 4), dpi=150)
    plt.plot(y_true, label="Actual", linewidth=2)
    plt.plot(y_pred, label="Predicted", linewidth=2)
    plt.axvspan(best_start, best_end, color="green", alpha=0.1, label="Best Prediction Region")
    plt.xlabel("Time Step")
    plt.ylabel("Load (kWh)")
    plt.legend()
    plt.grid(alpha=0.3)

    _save_show(save_path)


def plot_station_level_average(station_df: pd.DataFrame, save_path: str | None = None):
    if station_df.empty:
        print("Station table is empty. No station-level plot generated.")
        return

    locs = station_df["Location"].tolist()
    actual_vals = station_df["Actual"].values
    predicted_vals = station_df["Predicted"].values

    x = np.arange(len(locs))
    w = 0.28
    gap = 0.06

    fig, ax = plt.subplots(figsize=(13, 7), dpi=150)
    for i, (av, pv) in enumerate(zip(actual_vals, predicted_vals)):
        ax.bar(x[i] - w / 2 - gap / 2, av, w, color="#2166AC", edgecolor="black", linewidth=0.6, alpha=0.92, zorder=3)
        ax.bar(x[i] + w / 2 + gap / 2, pv, w, color="#1B7837", edgecolor="black", linewidth=0.6, hatch="////", alpha=0.88, zorder=3)

    ax.plot(x - w / 2 - gap / 2, actual_vals, color="#2166AC", lw=1.4, linestyle=(0, (5, 3)), marker="o", zorder=7)
    ax.plot(x + w / 2 + gap / 2, predicted_vals, color="#1B7837", lw=1.4, linestyle=(0, (5, 3)), marker="s", zorder=7)

    ax.set_xticks(x)
    ax.set_xticklabels(locs, rotation=20, ha="right")
    ax.set_ylabel("Average Load (kWh)")
    ax.grid(axis="y", alpha=0.3)

    legend_handles = [
        mpatches.Patch(color="#2166AC", label="Actual"),
        mpatches.Patch(color="#1B7837", hatch="////", label="Predicted"),
        Line2D([0], [0], color="#2166AC", linestyle=(0, (5, 3)), marker="o", label="Actual trend"),
        Line2D([0], [0], color="#1B7837", linestyle=(0, (5, 3)), marker="s", label="Predicted trend"),
    ]
    ax.legend(handles=legend_handles, fontsize=9)

    _save_show(save_path)
