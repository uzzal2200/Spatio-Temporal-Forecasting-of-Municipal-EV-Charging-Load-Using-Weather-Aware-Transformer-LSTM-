# ============================================================
# scripts/run_pipeline.py  —  Single entry point to run everything
# ============================================================
"""
Usage (from project root):
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --skip-data      # skip data prep
    python scripts/run_pipeline.py --skip-train     # skip training
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from configs.config import DEVICE, FIGURE_OUTPUT_DIR, RESULTS_OUTPUT_CSV

from src.data.ev_cleaner        import run_ev_cleaning
from src.data.asos_cleaner      import run_asos_cleaning
from src.data.feature_engineering import build_features
from src.data.dataset           import build_dataloaders
from src.models.architectures   import build_model_registry
from src.training.trainer       import train_all_models
from src.evaluation.reporting   import build_results_table, build_station_average_table
from src.visualization.plots    import (
    plot_loss_curves, plot_r2_comparison, plot_mape_comparison,
    plot_mae_rmse, plot_actual_vs_predicted, plot_all_models_overlay,
    plot_scatter_all, plot_residuals, plot_temporal_scales,
    plot_uncertainty_intervals, plot_peak_load_comparison,
    plot_model_radar, plot_best_prediction_region, plot_station_level_average,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-data",  action="store_true",
                        help="Skip EV + ASOS cleaning (use existing processed files)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training (requires saved results)")
    return parser.parse_args()


def main(args=None):
    if args is None:
        args = parse_args()
    print(f"\n{'='*60}")
    print(f"  EV Load Forecasting Pipeline")
    print(f"  Device : {DEVICE}")
    print(f"{'='*60}\n")

    # ── Step 1: Data Preparation ─────────────────────────────
    if not args.skip_data:
        print("[ 1/4 ]  Data Cleaning …")
        run_ev_cleaning()
        run_asos_cleaning()
    else:
        print("[ 1/4 ]  Skipping data cleaning (--skip-data)")

    # ── Step 2: Feature Engineering ──────────────────────────
    print("\n[ 2/4 ]  Feature Engineering …")
    daily, feat_cols = build_features()
    print(f"         Features: {len(feat_cols)}  |  Records: {len(daily):,}")

    # ── Step 3: Build DataLoaders ─────────────────────────────
    print("\n[ 3/4 ]  Building DataLoaders …")
    (train_loader, val_loader, test_loader,
     loc_list_sorted, loc_scalers, test_counts) = build_dataloaders(daily, feat_cols)

    input_dim      = len(feat_cols)
    model_registry = build_model_registry(input_dim)

    # ── Step 4: Train ─────────────────────────────────────────
    if not args.skip_train:
        print("\n[ 4/4 ]  Training Models …")
        all_results, all_history, all_preds_inv, _trained_models = train_all_models(
            model_registry, train_loader, val_loader, test_loader,
            loc_list_sorted, loc_scalers, test_counts,
        )
    else:
        print("[ 4/4 ]  Skipping training (--skip-train)")
        return

    # ── Visualization ─────────────────────────────────────────
    print("\n[ Plots ]  Generating figures …")
    os.makedirs(FIGURE_OUTPUT_DIR, exist_ok=True)

    plot_loss_curves(all_history,
                     save_path=os.path.join(FIGURE_OUTPUT_DIR, "loss_curves.png"))
    plot_r2_comparison(all_results,
                       save_path=os.path.join(FIGURE_OUTPUT_DIR, "r2_comparison.png"))
    plot_mape_comparison(all_results,
                         save_path=os.path.join(FIGURE_OUTPUT_DIR, "mape_comparison.png"))
    plot_mae_rmse(all_results,
                  save_path=os.path.join(FIGURE_OUTPUT_DIR, "mae_rmse.png"))

    yt, yp = all_preds_inv["Transformer-LSTM"]
    plot_actual_vs_predicted(yt, yp,
                             save_path=os.path.join(FIGURE_OUTPUT_DIR, "actual_vs_predicted.png"))
    plot_all_models_overlay(all_preds_inv,
                            save_path=os.path.join(FIGURE_OUTPUT_DIR, "all_models_overlay.png"))
    plot_scatter_all(all_preds_inv,
                     save_path=os.path.join(FIGURE_OUTPUT_DIR, "scatter_all.png"))
    plot_residuals(yt, yp,
                   save_path=os.path.join(FIGURE_OUTPUT_DIR, "residuals.png"))
    plot_uncertainty_intervals(yt, yp,
                               save_path=os.path.join(FIGURE_OUTPUT_DIR, "uncertainty_intervals.png"))
    plot_temporal_scales(yt, yp, daily["Date"],
                         save_path=os.path.join(FIGURE_OUTPUT_DIR, "temporal_scales.png"))
    plot_peak_load_comparison(yt, yp,
                              save_path=os.path.join(FIGURE_OUTPUT_DIR, "peak_load_comparison.png"))
    plot_model_radar(all_results,
                     save_path=os.path.join(FIGURE_OUTPUT_DIR, "radar_metrics.png"))
    plot_best_prediction_region(yt, yp,
                                save_path=os.path.join(FIGURE_OUTPUT_DIR, "best_prediction_region.png"))

    station_df = build_station_average_table(yt, yp, loc_list_sorted, test_counts)
    plot_station_level_average(
        station_df,
        save_path=os.path.join(FIGURE_OUTPUT_DIR, "station_level_average.png"),
    )

    summary_df = build_results_table(all_results)
    os.makedirs(os.path.dirname(RESULTS_OUTPUT_CSV), exist_ok=True)
    summary_df.to_csv(RESULTS_OUTPUT_CSV, index=False)
    print("\n[ Results ]  Model comparison table")
    print(summary_df.to_string(index=False))

    print(f"\nPipeline complete. Figures saved to: {FIGURE_OUTPUT_DIR}")
    print(f"Metrics table saved to: {RESULTS_OUTPUT_CSV}")


if __name__ == "__main__":
    main()
