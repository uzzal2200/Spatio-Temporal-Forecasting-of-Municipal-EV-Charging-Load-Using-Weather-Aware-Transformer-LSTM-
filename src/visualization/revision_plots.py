import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_figure19_demand_bin_error(demand_bin_df: pd.DataFrame, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    bins = demand_bin_df["Demand bin"].tolist()
    mape = demand_bin_df["MAPE (%)"].to_numpy(dtype=float)
    n_samples = demand_bin_df["n"].to_numpy(dtype=int)

    x = np.arange(len(bins))
    bars = axes[0].bar(x, mape, color="#2D6A4F", alpha=0.9)
    axes[0].axhline(19.62, color="#E76F51", linestyle="--", linewidth=1.4, label="Overall MAPE = 19.62%")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(bins)
    axes[0].set_ylabel("MAPE (%)")
    axes[0].set_xlabel("Daily-demand bin (kWh/day)")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    for i, b in enumerate(bars):
        axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, f"n={n_samples[i]}", ha="center", fontsize=8)

    mae = demand_bin_df["MAE (kWh)"].to_numpy(dtype=float)
    rmse = demand_bin_df["RMSE (kWh)"].to_numpy(dtype=float)
    axes[1].plot(x, mae, marker="o", linewidth=2, color="#264653", label="MAE")
    axes[1].plot(x, rmse, marker="s", linewidth=2, color="#D62828", label="RMSE")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(bins)
    axes[1].set_ylabel("Absolute error (kWh)")
    axes[1].set_xlabel("Daily-demand bin (kWh/day)")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_figure20_feature_group_ablation(feature_df: pd.DataFrame, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = feature_df.copy()
    labels = df["Variant"].str.replace("w/o ", "", regex=False)
    delta_rmse = df["Delta RMSE (%)"].str.replace("+", "", regex=False).astype(float)
    baseline_r2 = 0.9731
    baseline_mape = 19.62
    r2_drop = baseline_r2 - df["R2"].astype(float)
    mape_delta = df["MAPE (%)"].astype(float) - baseline_mape

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    axes[0].bar(labels, delta_rmse, color="#A7C957")
    axes[0].set_ylabel("Relative RMSE increase (%)")
    axes[0].set_title("(a) RMSE sensitivity by removed feature group")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.25)

    ax1 = axes[1]
    ax2 = ax1.twinx()
    x = np.arange(len(labels))
    ax1.plot(x, r2_drop, color="#1D3557", marker="o", linewidth=2, label="R2 drop")
    ax2.plot(x, mape_delta, color="#F4A261", marker="s", linewidth=2, label="MAPE increase")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20)
    ax1.set_ylabel("R2 drop")
    ax2.set_ylabel("MAPE increase (percentage points)")
    ax1.set_title("(b) R2 drop and MAPE increase")
    ax1.grid(alpha=0.25)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_figure21_hyperparameter_sensitivity(hyper_df: pd.DataFrame, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    baseline = "Full Proposed Model (N_enc=4, h=8, h_LSTM=128)"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150)

    # Panel (a): N_enc
    sub_a = hyper_df[hyper_df["Variant"].str.contains("h=8, h_LSTM=128")].copy()
    sub_a["x"] = sub_a["Variant"].str.extract(r"N_enc=(\d+)").astype(int)
    sub_a = sub_a.sort_values("x")

    # Panel (b): heads h
    sub_b = hyper_df[hyper_df["Variant"].str.contains("N_enc=4") & hyper_df["Variant"].str.contains("h_LSTM=128")].copy()
    sub_b["x"] = sub_b["Variant"].str.extract(r"h=(\d+)").astype(int)
    sub_b = sub_b.sort_values("x")

    # Panel (c): h_LSTM
    sub_c = hyper_df[hyper_df["Variant"].str.contains("N_enc=4, h=8")].copy()
    sub_c["x"] = sub_c["Variant"].str.extract(r"h_LSTM=(\d+)").astype(int)
    sub_c = sub_c.sort_values("x")

    for ax, sub, title in [
        (axes[0], sub_a, "(a) Encoder layers N_enc"),
        (axes[1], sub_b, "(b) Attention heads h"),
        (axes[2], sub_c, "(c) LSTM hidden size h_LSTM"),
    ]:
        rmse = sub["RMSE (kWh)"].astype(float).to_numpy()
        r2 = sub["R2"].astype(float).to_numpy()
        x = sub["x"].to_numpy()

        ax2 = ax.twinx()
        ax.plot(x, rmse, color="#D62828", marker="o", linewidth=2, label="RMSE")
        ax2.plot(x, r2, color="#1D3557", marker="s", linewidth=2, label="R2")

        ax.set_title(title)
        ax.set_xlabel("Value")
        ax.set_ylabel("RMSE (kWh)", color="#D62828")
        ax2.set_ylabel("R2", color="#1D3557")
        ax.grid(alpha=0.25)

        base_sub = sub[sub["Variant"] == baseline]
        if not base_sub.empty:
            bx = int(base_sub["x"].iloc[0])
            by = float(base_sub["RMSE (kWh)"].iloc[0])
            ax.scatter([bx], [by], s=90, facecolors="none", edgecolors="black", linewidths=1.6)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
