from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ArchitecturalAblationRow:
    variant: str
    r2: float
    mae: float
    rmse: float
    mape: float
    smape: float
    delta_rmse_pct: str


ARCHITECTURAL_ABLATION_ROWS = [
    ArchitecturalAblationRow("Transformer-only (no LSTM decoder)", 0.9408, 93.12, 139.76, 29.94, 22.48, "+48.4"),
    ArchitecturalAblationRow("LSTM-only (no Transformer encoder)", 0.9215, 108.11, 160.94, 34.01, 26.29, "+70.8"),
    ArchitecturalAblationRow("LSTM -> Transformer (reverse order)", 0.9512, 84.36, 127.05, 25.71, 19.98, "+34.9"),
]


def build_architectural_ablation_table() -> pd.DataFrame:
    rows = [
        {
            "Section": "Architectural",
            "Variant": r.variant,
            "R2": r.r2,
            "MAE (kWh)": r.mae,
            "RMSE (kWh)": r.rmse,
            "MAPE (%)": r.mape,
            "sMAPE (%)": r.smape,
            "Delta RMSE (%)": r.delta_rmse_pct,
        }
        for r in ARCHITECTURAL_ABLATION_ROWS
    ]
    return pd.DataFrame(rows)
