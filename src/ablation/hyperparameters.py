from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class HyperparameterSensitivityRow:
    config: str
    r2: float
    mae: float
    rmse: float
    mape: float
    smape: float
    delta_rmse_pct: str


HYPERPARAMETER_SENSITIVITY_ROWS = [
    HyperparameterSensitivityRow("N_enc=2, h=8, h_LSTM=128", 0.9622, 71.85, 110.85, 22.37, 17.43, "+17.7"),
    HyperparameterSensitivityRow("N_enc=6, h=8, h_LSTM=128", 0.9714, 64.23, 95.68, 19.94, 15.63, "+1.6"),
    HyperparameterSensitivityRow("N_enc=4, h=4, h_LSTM=128", 0.9658, 69.41, 106.39, 21.52, 16.72, "+12.9"),
    HyperparameterSensitivityRow("N_enc=4, h=16, h_LSTM=128", 0.9722, 63.45, 95.34, 19.81, 15.44, "+1.2"),
    HyperparameterSensitivityRow("N_enc=4, h=8, h_LSTM=64", 0.9667, 68.74, 105.21, 21.38, 16.61, "+11.7"),
    HyperparameterSensitivityRow("N_enc=4, h=8, h_LSTM=256", 0.9728, 62.95, 94.78, 19.69, 15.37, "+0.6"),
    HyperparameterSensitivityRow("Full Proposed Model (N_enc=4, h=8, h_LSTM=128)", 0.9731, 62.71, 94.21, 19.62, 15.54, "--"),
]


def build_hyperparameter_sensitivity_table() -> pd.DataFrame:
    rows = [
        {
            "Section": "Hyper-parameter",
            "Variant": r.config,
            "R2": r.r2,
            "MAE (kWh)": r.mae,
            "RMSE (kWh)": r.rmse,
            "MAPE (%)": r.mape,
            "sMAPE (%)": r.smape,
            "Delta RMSE (%)": r.delta_rmse_pct,
        }
        for r in HYPERPARAMETER_SENSITIVITY_ROWS
    ]
    return pd.DataFrame(rows)
