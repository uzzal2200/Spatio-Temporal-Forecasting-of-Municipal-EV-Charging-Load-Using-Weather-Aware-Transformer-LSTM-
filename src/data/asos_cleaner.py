# ============================================================
# src/data/asos_cleaner.py  —  ASOS weather data cleaning
# ============================================================

import pandas as pd
from configs.config import (
    RAW_ASOS_PATH, CLEANED_ASOS_PATH,
    ASOS_REQUIRED_COLS, DATE_START, DATE_END,
)


def load_raw_asos(path: str = RAW_ASOS_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[ASOS] Loaded : {df.shape[0]:,} rows, {df.shape[1]} cols")
    return df


def select_and_rename(df: pd.DataFrame) -> pd.DataFrame:
    existing = [c for c in ASOS_REQUIRED_COLS if c in df.columns]
    df = df[existing].copy()
    if "valid" in df.columns:
        df.rename(columns={"valid": "Date"}, inplace=True)
    print(f"[ASOS] Columns selected: {list(df.columns)}")
    return df


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["Date"].notna()].copy()
    df["Date"] = df["Date"].dt.date

    weather_cols = ["tmpf", "relh", "feel", "sped", "p01m", "snowdepth"]
    existing_w   = [c for c in weather_cols if c in df.columns]

    agg_funcs = {c: "mean" for c in existing_w}
    if "p01m" in agg_funcs:
        agg_funcs["p01m"] = "sum"
    if "snowdepth" in agg_funcs:
        agg_funcs["snowdepth"] = "max"

    df_daily = df.groupby(["station", "Date"], as_index=False).agg(agg_funcs)

    float_cols = df_daily.select_dtypes("float64").columns
    df_daily[float_cols] = df_daily[float_cols].round(2)
    print(f"[ASOS] Daily aggregated: {df_daily.shape}")
    return df_daily


def filter_dates(df: pd.DataFrame) -> pd.DataFrame:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[(df["Date"] >= DATE_START) & (df["Date"] <= DATE_END)].copy()
    df = df.sort_values(["Date", "station"]).reset_index(drop=True)
    print(f"[ASOS] Date filtered: {len(df):,} rows")
    return df


def save_cleaned_asos(df: pd.DataFrame, path: str = CLEANED_ASOS_PATH) -> None:
    import os; os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[ASOS] Saved → {path}  ({len(df):,} rows)")


def run_asos_cleaning(raw_path: str = RAW_ASOS_PATH,
                      out_path: str = CLEANED_ASOS_PATH) -> pd.DataFrame:
    """Full ASOS cleaning pipeline."""
    df = load_raw_asos(raw_path)
    df = select_and_rename(df)
    df = aggregate_daily(df)
    df = filter_dates(df)
    save_cleaned_asos(df, out_path)
    return df


if __name__ == "__main__":
    run_asos_cleaning()
