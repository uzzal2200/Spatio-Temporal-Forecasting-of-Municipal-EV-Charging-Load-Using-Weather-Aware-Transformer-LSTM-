import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error in percent."""
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denominator > 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(numerator[mask] / denominator[mask]) * 100.0)


def psnr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio in dB for regression signal quality."""
    mse = float(np.mean((y_true - y_pred) ** 2))
    if mse == 0:
        return float("inf")
    max_val = float(np.max(y_true))
    if max_val == 0:
        return float("nan")
    return float(10.0 * np.log10((max_val ** 2) / mse))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return unified metrics used across training, reports, and figures."""
    y_pred_clip = np.clip(y_pred, 0, None)

    mae = float(mean_absolute_error(y_true, y_pred_clip))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred_clip)))
    r2 = float(r2_score(y_true, y_pred_clip))

    mask = y_true > 1.0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred_clip[mask]) / y_true[mask])) * 100.0) if mask.sum() > 0 else float("nan")

    return {
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "sMAPE": smape(y_true, y_pred_clip),
        "PSNR": psnr(y_true, y_pred_clip),
    }
