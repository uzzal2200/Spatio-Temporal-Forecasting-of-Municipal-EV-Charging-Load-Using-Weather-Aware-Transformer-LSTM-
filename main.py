"""Project entrypoint.

Run:
    python main.py
"""

import os

from configs.config import DEVICE, FIGURE_OUTPUT_DIR, RESULTS_OUTPUT_CSV
from src.data.asos_cleaner import run_asos_cleaning
from src.data.dataset import build_dataloaders
from src.data.ev_cleaner import run_ev_cleaning
from src.data.feature_engineering import build_features
from src.evaluation.reporting import build_results_table, build_station_average_table
from src.models.architectures import build_model_registry
from src.training.trainer import train_all_models
from src.visualization.plots import (
    plot_actual_vs_predicted,
    plot_all_models_overlay,
    plot_best_prediction_region,
    plot_loss_curves,
    plot_mae_rmse,
    plot_mape_comparison,
    plot_model_radar,
    plot_peak_load_comparison,
    plot_r2_comparison,
    plot_residuals,
    plot_scatter_all,
    plot_station_level_average,
    plot_temporal_scales,
    plot_uncertainty_intervals,
)


class ProjectEVPipeline:
    """Step-by-step orchestration for training, evaluation, and visualization."""

    def __init__(self):
        self.daily = None
        self.feat_cols = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.loc_list_sorted = None
        self.loc_scalers = None
        self.test_counts = None
        self.all_results = None
        self.all_history = None
        self.all_preds_inv = None

    def run_data_preprocessing(self):
        print("[STEP 1] Data preprocessing start...")
        run_ev_cleaning()
        run_asos_cleaning()
        self.daily, self.feat_cols = build_features()
        print("[STEP 1] Data preprocessing done.")

    def run_training(self):
        print("[STEP 2] Training start...")
        (
            self.train_loader,
            self.val_loader,
            self.test_loader,
            self.loc_list_sorted,
            self.loc_scalers,
            self.test_counts,
        ) = build_dataloaders(self.daily, self.feat_cols)

        model_registry = build_model_registry(len(self.feat_cols))
        self.all_results, self.all_history, self.all_preds_inv, _trained_models = train_all_models(
            model_registry,
            self.train_loader,
            self.val_loader,
            self.test_loader,
            self.loc_list_sorted,
            self.loc_scalers,
            self.test_counts,
        )
        print("[STEP 2] Training done.")

    def run_evaluation(self):
        print("[STEP 3] Evaluation start...")
        summary_df = build_results_table(self.all_results)
        os.makedirs(os.path.dirname(RESULTS_OUTPUT_CSV), exist_ok=True)
        summary_df.to_csv(RESULTS_OUTPUT_CSV, index=False)
        print(summary_df.to_string(index=False))
        print("[STEP 3] Evaluation done.")

    def run_visualization(self):
        print("[STEP 4] Visualization start...")
        os.makedirs(FIGURE_OUTPUT_DIR, exist_ok=True)

        plot_loss_curves(self.all_history, save_path=os.path.join(FIGURE_OUTPUT_DIR, "loss_curves.png"))
        plot_r2_comparison(self.all_results, save_path=os.path.join(FIGURE_OUTPUT_DIR, "r2_comparison.png"))
        plot_mape_comparison(self.all_results, save_path=os.path.join(FIGURE_OUTPUT_DIR, "mape_comparison.png"))
        plot_mae_rmse(self.all_results, save_path=os.path.join(FIGURE_OUTPUT_DIR, "mae_rmse.png"))

        yt, yp = self.all_preds_inv["Transformer-LSTM"]
        plot_actual_vs_predicted(yt, yp, save_path=os.path.join(FIGURE_OUTPUT_DIR, "actual_vs_predicted.png"))
        plot_all_models_overlay(self.all_preds_inv, save_path=os.path.join(FIGURE_OUTPUT_DIR, "all_models_overlay.png"))
        plot_scatter_all(self.all_preds_inv, save_path=os.path.join(FIGURE_OUTPUT_DIR, "scatter_all.png"))
        plot_residuals(yt, yp, save_path=os.path.join(FIGURE_OUTPUT_DIR, "residuals.png"))
        plot_uncertainty_intervals(yt, yp, save_path=os.path.join(FIGURE_OUTPUT_DIR, "uncertainty_intervals.png"))
        plot_temporal_scales(yt, yp, self.daily["Date"], save_path=os.path.join(FIGURE_OUTPUT_DIR, "temporal_scales.png"))
        plot_peak_load_comparison(yt, yp, save_path=os.path.join(FIGURE_OUTPUT_DIR, "peak_load_comparison.png"))
        plot_model_radar(self.all_results, save_path=os.path.join(FIGURE_OUTPUT_DIR, "radar_metrics.png"))
        plot_best_prediction_region(yt, yp, save_path=os.path.join(FIGURE_OUTPUT_DIR, "best_prediction_region.png"))

        station_df = build_station_average_table(yt, yp, self.loc_list_sorted, self.test_counts)
        plot_station_level_average(
            station_df,
            save_path=os.path.join(FIGURE_OUTPUT_DIR, "station_level_average.png"),
        )

        print("[STEP 4] Visualization done.")

    def run(self):
        print("=" * 60)
        print("Project-EV Full Pipeline Execution")
        print(f"Device: {DEVICE}")
        print("=" * 60)

        self.run_data_preprocessing()
        self.run_training()
        self.run_evaluation()
        self.run_visualization()

        print("=" * 60)
        print("Pipeline completed successfully.")
        print(f"Results CSV: {RESULTS_OUTPUT_CSV}")
        print(f"Figure output dir: {FIGURE_OUTPUT_DIR}")
        print("=" * 60)


if __name__ == "__main__":
    ProjectEVPipeline().run()
