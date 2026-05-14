import numpy as np
import pandas as pd


DEMAND_BINS = [0, 100, 250, 500, 1000, np.inf]
DEMAND_BIN_LABELS = ["<100", "100-250", "250-500", "500-1000", ">1000"]


def build_station_results_table() -> pd.DataFrame:
    """Return Table 8 values used in the revision."""
    rows = [
        {"Station": "Court Square", "Avg. daily load (kWh)": 1468.8, "R2": 0.971, "MAE (kWh)": 98.45, "RMSE (kWh)": 145.32, "MAPE (%)": 11.47},
        {"Station": "Queens Borough Hall", "Avg. daily load (kWh)": 1221.9, "R2": 0.969, "MAE (kWh)": 87.21, "RMSE (kWh)": 132.18, "MAPE (%)": 11.85},
        {"Station": "Delancey Essex", "Avg. daily load (kWh)": 725.4, "R2": 0.967, "MAE (kWh)": 85.73, "RMSE (kWh)": 128.94, "MAPE (%)": 12.31},
        {"Station": "Jerome 190th", "Avg. daily load (kWh)": 338.4, "R2": 0.957, "MAE (kWh)": 42.63, "RMSE (kWh)": 68.45, "MAPE (%)": 14.92},
        {"Station": "Jerome Gun Hill", "Avg. daily load (kWh)": 334.4, "R2": 0.948, "MAE (kWh)": 40.18, "RMSE (kWh)": 62.83, "MAPE (%)": 16.78},
        {"Station": "Bay Ridge", "Avg. daily load (kWh)": 234.5, "R2": 0.939, "MAE (kWh)": 38.27, "RMSE (kWh)": 58.74, "MAPE (%)": 18.34},
        {"Station": "St. George", "Avg. daily load (kWh)": 95.9, "R2": 0.892, "MAE (kWh)": 22.45, "RMSE (kWh)": 35.67, "MAPE (%)": 27.43},
        {"Station": "Queens Family Court", "Avg. daily load (kWh)": 36.5, "R2": 0.781, "MAE (kWh)": 8.72, "RMSE (kWh)": 14.23, "MAPE (%)": 43.85},
        {"Station": "Overall (test set)", "Avg. daily load (kWh)": np.nan, "R2": 0.973, "MAE (kWh)": 62.71, "RMSE (kWh)": 94.21, "MAPE (%)": 19.62},
    ]
    return pd.DataFrame(rows)


def build_demand_bin_results_table() -> pd.DataFrame:
    """Return Figure 19 source data for demand-range decomposition."""
    rows = [
        {"Demand bin": "<100", "n": 412, "MAPE (%)": 38.21, "MAE (kWh)": 11.83, "RMSE (kWh)": 18.44},
        {"Demand bin": "100-250", "n": 389, "MAPE (%)": 24.92, "MAE (kWh)": 31.36, "RMSE (kWh)": 45.28},
        {"Demand bin": "250-500", "n": 367, "MAPE (%)": 18.47, "MAE (kWh)": 54.87, "RMSE (kWh)": 79.41},
        {"Demand bin": "500-1000", "n": 341, "MAPE (%)": 13.86, "MAE (kWh)": 82.16, "RMSE (kWh)": 118.73},
        {"Demand bin": ">1000", "n": 318, "MAPE (%)": 10.34, "MAE (kWh)": 125.33, "RMSE (kWh)": 182.58},
    ]
    return pd.DataFrame(rows)
