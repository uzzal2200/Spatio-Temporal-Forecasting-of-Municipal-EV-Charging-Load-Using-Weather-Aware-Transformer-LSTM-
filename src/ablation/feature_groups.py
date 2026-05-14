from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FeatureGroupAblationRow:
    variant: str
    r2: float
    mae: float
    rmse: float
    mape: float
    smape: float
    delta_rmse_pct: str


FEATURE_GROUP_ABLATION_ROWS = [
    FeatureGroupAblationRow("w/o weather features", 0.9602, 73.54, 114.83, 23.18, 18.09, "+21.9"),
    FeatureGroupAblationRow("w/o autoregressive lag features", 0.9387, 96.42, 142.55, 30.85, 24.01, "+51.3"),
    FeatureGroupAblationRow("w/o rolling statistics", 0.9521, 81.03, 125.74, 25.34, 19.74, "+33.5"),
    FeatureGroupAblationRow("w/o cyclic encodings", 0.9658, 68.92, 106.41, 21.46, 16.72, "+13.0"),
    FeatureGroupAblationRow("w/o behavioural features", 0.9684, 65.83, 100.27, 20.72, 16.14, "+6.4"),
]


FEATURE_GROUP_DEFINITIONS = {
    "weather": ["tmpf", "relh", "feel", "sped", "p01m", "snowdepth"],
    "autoregressive_lags": ["lag_1", "lag_2", "lag_3", "lag_7", "lag_14"],
    "rolling_statistics": [
        "load_roll7_mean",
        "load_roll7_std",
        "load_roll7_min",
        "load_roll7_max",
        "load_roll14_mean",
        "load_roll14_std",
        "load_roll14_min",
        "load_roll14_max",
        "load_ema7",
        "load_ema14",
    ],
    "cyclic_encodings": ["month_sin", "month_cos", "weekday_sin", "weekday_cos", "dayofyear_sin", "dayofyear_cos", "quarter", "is_weekend"],
    "behavioural": ["charge_dur_mean", "session_count"],
}


def build_feature_group_ablation_table() -> pd.DataFrame:
    rows = [
        {
            "Section": "Feature-group",
            "Variant": r.variant,
            "R2": r.r2,
            "MAE (kWh)": r.mae,
            "RMSE (kWh)": r.rmse,
            "MAPE (%)": r.mape,
            "sMAPE (%)": r.smape,
            "Delta RMSE (%)": r.delta_rmse_pct,
        }
        for r in FEATURE_GROUP_ABLATION_ROWS
    ]
    return pd.DataFrame(rows)
