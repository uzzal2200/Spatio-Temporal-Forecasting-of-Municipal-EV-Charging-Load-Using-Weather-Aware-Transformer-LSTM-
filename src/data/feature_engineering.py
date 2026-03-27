# ============================================================
# src/data/feature_engineering.py  —  Merge + Feature Pipeline
# ============================================================

import numpy as np
import pandas as pd
from configs.config import (
    CLEANED_EV_PATH, CLEANED_ASOS_PATH, MERGED_EV_PATH,
    LOCATION_WEATHER_MAP, WEATHER_COLS,
    BASE_FEATS, ROLLING_WINDOWS, EMA_SPANS, TARGET,
    LAG_FEATURES, MIN_LOCATION_SAMPLES, LOWER_OUTLIER_Q, UPPER_OUTLIER_Q,
)


# ── Step 1: Merge EV + Weather ────────────────────────────────────────────────

def merge_ev_weather(ev_path: str = CLEANED_EV_PATH,
                     asos_path: str = CLEANED_ASOS_PATH,
                     out_path: str  = MERGED_EV_PATH) -> pd.DataFrame:
    df_ev      = pd.read_csv(ev_path)
    df_weather = pd.read_csv(asos_path)

    df_ev["Date"]      = pd.to_datetime(df_ev["Date"]).dt.date
    df_weather["Date"] = pd.to_datetime(df_weather["Date"]).dt.date

    # Map location → weather station
    df_ev["weather_station"] = df_ev["Location Name"].map(LOCATION_WEATHER_MAP)

    df_final = df_ev.merge(
        df_weather,
        left_on=["Date", "weather_station"],
        right_on=["Date", "station"],
        how="left",
    )
    df_final.drop(columns=["station"], inplace=True, errors="ignore")

    import os; os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_final.to_csv(out_path, index=False)
    print(f"[Merge] Saved → {out_path}  ({len(df_final):,} rows)")
    return df_final


# ── Step 2: Temporal Feature Engineering ─────────────────────────────────────

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Date"]        = pd.to_datetime(df["Date"], errors="coerce")
    df["month"]       = df["Date"].dt.month
    df["weekday"]     = df["Date"].dt.weekday
    df["is_weekend"]  = (df["weekday"] >= 5).astype(int)
    df["quarter"]     = df["Date"].dt.quarter
    df["dayofyear"]   = df["Date"].dt.dayofyear

    df["month_sin"]      = np.sin(2 * np.pi * df["month"]     / 12)
    df["month_cos"]      = np.cos(2 * np.pi * df["month"]     / 12)
    df["weekday_sin"]    = np.sin(2 * np.pi * df["weekday"]   / 7)
    df["weekday_cos"]    = np.cos(2 * np.pi * df["weekday"]   / 7)
    df["dayofyear_sin"]  = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["dayofyear_cos"]  = np.cos(2 * np.pi * df["dayofyear"] / 365)
    return df


# ── Step 3: Interpolate Missing Weather ──────────────────────────────────────

def fill_weather(df: pd.DataFrame) -> pd.DataFrame:
    df.sort_values(["weather_station", "Date"], inplace=True)
    df[WEATHER_COLS] = (
        df.groupby("weather_station")[WEATHER_COLS]
          .transform(lambda x: x.interpolate())
    )
    df[WEATHER_COLS] = df[WEATHER_COLS].bfill().ffill()
    df[WEATHER_COLS] = df[WEATHER_COLS].fillna(df[WEATHER_COLS].median())
    return df


# ── Step 4: Daily Aggregation ─────────────────────────────────────────────────

def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    AGG = {
        "Energy Provided (kWh)": "sum",
        "Charge Duration (min)": ["mean", "count"],
        **{c: "mean" for c in ["tmpf", "relh", "feel", "sped"]},
        "p01m": "sum", "snowdepth": "max",
        "is_weekend": "first",
        "month_sin": "first", "month_cos": "first",
        "weekday_sin": "first", "weekday_cos": "first",
        "quarter": "first",
        "dayofyear_sin": "first", "dayofyear_cos": "first",
    }

    daily = (df.groupby(["Location Name", "Date"])
               .agg(AGG)
               .reset_index())

    # Flatten MultiIndex columns
    daily.columns = [
        "_".join(c).strip("_") if isinstance(c, tuple) else c
        for c in daily.columns
    ]

    rename_map = {
        "Location Name": "Location",
        "Energy Provided (kWh)_sum": "load_kwh",
        "Charge Duration (min)_mean": "charge_dur_mean",
        "Charge Duration (min)_count": "session_count",
    }
    daily.rename(columns=rename_map, inplace=True)
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily.sort_values(["Location", "Date"], inplace=True)
    daily.reset_index(drop=True, inplace=True)

    q_lo = daily[TARGET].quantile(LOWER_OUTLIER_Q)
    q_hi = daily[TARGET].quantile(UPPER_OUTLIER_Q)
    daily = daily[(daily[TARGET] >= q_lo) & (daily[TARGET] <= q_hi)].copy()

    counts = daily.groupby("Location").size()
    valid_locs = counts[counts >= MIN_LOCATION_SAMPLES].index.tolist()
    daily = daily[daily["Location"].isin(valid_locs)].copy()

    print(f"[Daily] Aggregated: {daily.shape}  |  Locations: {daily['Location'].nunique()}")
    return daily


# ── Step 5: Lag + Rolling + EMA Features ─────────────────────────────────────

def add_lag_rolling_features(daily: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    feat_cols = list(BASE_FEATS)

    # Lag features
    for lag in LAG_FEATURES:
        col = f"lag_{lag}"
        daily[col] = daily.groupby("Location")["load_kwh"].transform(
            lambda x: x.shift(lag))
        feat_cols.append(col)

    # Rolling features
    for win in ROLLING_WINDOWS:
        for stat, fn in [("mean",  lambda x: x.shift(1).rolling(win, min_periods=1).mean()),
                         ("std",   lambda x: x.shift(1).rolling(win, min_periods=1).std().fillna(0)),
                         ("min",   lambda x: x.shift(1).rolling(win, min_periods=1).min()),
                         ("max",   lambda x: x.shift(1).rolling(win, min_periods=1).max())]:
            col = f"load_roll{win}_{stat}"
            daily[col] = daily.groupby("Location")["load_kwh"].transform(fn)
            feat_cols.append(col)

    # Exponential moving averages
    for span in EMA_SPANS:
        col = f"load_ema{span}"
        daily[col] = daily.groupby("Location")["load_kwh"].transform(
            lambda x: x.shift(1).ewm(span=span, adjust=False).mean())
        feat_cols.append(col)

    daily.dropna(inplace=True)
    daily.reset_index(drop=True, inplace=True)
    print(f"[Features] Total: {len(feat_cols)}  |  Records: {len(daily):,}")
    return daily, feat_cols


# ── Full Pipeline ─────────────────────────────────────────────────────────────

def build_features(ev_path: str  = CLEANED_EV_PATH,
                   asos_path: str = CLEANED_ASOS_PATH,
                   out_path: str  = MERGED_EV_PATH) -> tuple[pd.DataFrame, list]:
    """Run full merge + feature engineering pipeline."""
    df    = merge_ev_weather(ev_path, asos_path, out_path)
    df    = add_temporal_features(df)
    df    = fill_weather(df)
    daily = aggregate_daily(df)
    daily, feat_cols = add_lag_rolling_features(daily)
    return daily, feat_cols


if __name__ == "__main__":
    daily, feat_cols = build_features()
    print(f"\nFinal shape: {daily.shape}")
    print(f"Feature columns ({len(feat_cols)}):\n{feat_cols}")
