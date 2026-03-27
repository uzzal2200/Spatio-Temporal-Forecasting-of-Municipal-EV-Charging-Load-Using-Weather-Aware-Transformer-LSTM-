import numpy as np
import pandas as pd


PROPOSED_MODEL = "Transformer-LSTM"


def build_results_table(all_results: dict) -> pd.DataFrame:
    """Create a sorted model comparison table from metrics dictionary."""
    rows = []
    for model_name, metric_dict in all_results.items():
        row = {"Model": model_name}
        row.update(metric_dict)
        rows.append(row)

    table = pd.DataFrame(rows)
    sort_cols = [c for c in ["R2", "RMSE", "MAE"] if c in table.columns]
    if sort_cols:
        table = table.sort_values(by=["R2", "RMSE", "MAE"], ascending=[False, True, True], na_position="last")
    return table.reset_index(drop=True)


def build_station_average_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    loc_list_sorted: list,
    test_counts: dict,
) -> pd.DataFrame:
    """Average actual and predicted load by station based on contiguous test slices."""
    rows = []
    idx = 0
    for loc in loc_list_sorted:
        n_seq = int(test_counts.get(loc, 0))
        if n_seq <= 0:
            continue
        station_true = y_true[idx: idx + n_seq]
        station_pred = y_pred[idx: idx + n_seq]
        idx += n_seq

        rows.append(
            {
                "Location": loc,
                "Actual": float(np.mean(station_true)),
                "Predicted": float(np.mean(station_pred)),
                "AbsError": float(np.mean(np.abs(station_true - station_pred))),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["Location", "Actual", "Predicted", "AbsError"])

    return pd.DataFrame(rows).sort_values("Actual").reset_index(drop=True)


def compute_peak_load_metrics(y_true: np.ndarray, y_pred: np.ndarray, percentile: float = 90.0) -> dict:
    """Evaluate prediction quality only on the upper-load region."""
    threshold = np.percentile(y_true, percentile)
    mask = y_true >= threshold
    if mask.sum() == 0:
        return {
            "threshold": float(threshold),
            "peak_mae": float("nan"),
            "peak_rmse": float("nan"),
        }

    peak_true = y_true[mask]
    peak_pred = y_pred[mask]
    peak_mae = float(np.mean(np.abs(peak_true - peak_pred)))
    peak_rmse = float(np.sqrt(np.mean((peak_true - peak_pred) ** 2)))

    return {
        "threshold": float(threshold),
        "peak_mae": peak_mae,
        "peak_rmse": peak_rmse,
    }
