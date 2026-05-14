import pandas as pd


def build_efficiency_table() -> pd.DataFrame:
    """Return Table 10 computational benchmark data."""
    rows = [
        {"Model": "Simple RNN", "Params (K)": 12.10, "Train. time (min)": 8.4, "Inference (ms/sample)": 0.42, "Peak GPU mem (MB)": 28, "RMSE (kWh)": 181.01},
        {"Model": "LSTM", "Params (K)": 129.47, "Train. time (min)": 12.7, "Inference (ms/sample)": 0.68, "Peak GPU mem (MB)": 65, "RMSE (kWh)": 160.94},
        {"Model": "Transformer (encoder-only)", "Params (K)": 160.38, "Train. time (min)": 14.2, "Inference (ms/sample)": 0.81, "Peak GPU mem (MB)": 78, "RMSE (kWh)": 139.76},
        {"Model": "Informer", "Params (K)": 285.00, "Train. time (min)": 21.5, "Inference (ms/sample)": 1.34, "Peak GPU mem (MB)": 132, "RMSE (kWh)": 124.18},
        {"Model": "PatchTST", "Params (K)": 178.00, "Train. time (min)": 15.8, "Inference (ms/sample)": 0.89, "Peak GPU mem (MB)": 96, "RMSE (kWh)": 113.74},
        {"Model": "TFT", "Params (K)": 412.00, "Train. time (min)": 28.4, "Inference (ms/sample)": 1.72, "Peak GPU mem (MB)": 184, "RMSE (kWh)": 109.42},
        {"Model": "Trans.-LSTM (Proposed)", "Params (K)": 1070.00, "Train. time (min)": 32.7, "Inference (ms/sample)": 1.96, "Peak GPU mem (MB)": 248, "RMSE (kWh)": 94.21},
    ]
    return pd.DataFrame(rows)
