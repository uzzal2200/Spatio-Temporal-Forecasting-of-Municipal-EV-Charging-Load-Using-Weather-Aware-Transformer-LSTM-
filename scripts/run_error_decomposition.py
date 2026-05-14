"""Generate Table 8 + Figure 19 data and figure."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.error_decomposition import build_demand_bin_results_table, build_station_results_table
from src.visualization.revision_plots import plot_figure19_demand_bin_error


def main():
    root = Path(__file__).resolve().parents[1]
    outputs = root / "outputs"
    figure_dir = outputs / "revision_figures"
    os.makedirs(outputs, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    per_station = build_station_results_table()
    demand_bin = build_demand_bin_results_table()

    station_csv = outputs / "per_station_results.csv"
    bin_csv = outputs / "demand_bin_results.csv"
    fig19 = figure_dir / "figure_19_demand_bin_error.png"

    per_station.to_csv(station_csv, index=False)
    demand_bin.to_csv(bin_csv, index=False)
    plot_figure19_demand_bin_error(demand_bin, str(fig19))

    print(f"Saved Table 8 data: {station_csv}")
    print(f"Saved Figure 19 data: {bin_csv}")
    print(f"Saved Figure 19 plot: {fig19}")


if __name__ == "__main__":
    main()
