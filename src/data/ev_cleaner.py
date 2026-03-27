# ============================================================
# src/data/ev_cleaner.py  —  EV charging data cleaning
# ============================================================

import pandas as pd
from configs.config import (
    RAW_EV_PATH, CLEANED_EV_PATH,
    EV_REQUIRED_COLS, DATE_START, DATE_END,
)


def load_raw_ev(path: str = RAW_EV_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[EV] Loaded  : {df.shape[0]:,} rows, {df.shape[1]} cols")
    return df


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in EV_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df[EV_REQUIRED_COLS].copy()
    print(f"[EV] Columns selected: {df.shape}")
    return df


def remove_missing(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna().copy()
    print(f"[EV] Dropped NaNs: {before - len(df):,} rows removed → {len(df):,} remain")
    return df


def filter_dates(df: pd.DataFrame) -> pd.DataFrame:
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
    df = df[df["Date"].notna()].copy()
    df = df[(df["Date"] >= DATE_START) & (df["Date"] <= DATE_END)].copy()
    df = df.sort_values("Date").reset_index(drop=True)
    print(f"[EV] Date filtered ({DATE_START} → {DATE_END}): {len(df):,} rows")
    return df


def save_cleaned_ev(df: pd.DataFrame, path: str = CLEANED_EV_PATH) -> None:
    import os; os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[EV] Saved → {path}  ({len(df):,} rows)")


def run_ev_cleaning(raw_path: str = RAW_EV_PATH,
                    out_path: str = CLEANED_EV_PATH) -> pd.DataFrame:
    """Full EV cleaning pipeline."""
    df = load_raw_ev(raw_path)
    df = select_columns(df)
    df = remove_missing(df)
    df = filter_dates(df)
    save_cleaned_ev(df, out_path)
    return df


if __name__ == "__main__":
    run_ev_cleaning()
