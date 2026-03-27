# ============================================================
# configs/config.py  —  All hyperparameters & paths
# ============================================================

from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── Paths ────────────────────────────────────────────────────
RAW_EV_PATH       = str(PROJECT_ROOT / "Data" / "Electric_Vehicle__EV__Charging_Data-_Municipal_Lots_and_Garages.csv")
RAW_ASOS_PATH     = str(PROJECT_ROOT / "Data" / "asos.csv")
CLEANED_EV_PATH   = str(PROJECT_ROOT / "Data" / "EV_Charging_Cleaned.csv")
CLEANED_ASOS_PATH = str(PROJECT_ROOT / "Data" / "ASOS_Cleaned.csv")
MERGED_EV_PATH    = str(PROJECT_ROOT / "Data" / "EV.csv")

FIGURE_OUTPUT_DIR = str(PROJECT_ROOT / "outputs" / "figures")
RESULTS_OUTPUT_CSV = str(PROJECT_ROOT / "outputs" / "model_comparison_results.csv")

# ── Training Config ──────────────────────────────────────────
SEQ_LEN    = 21
BATCH_SIZE = 32
EPOCHS     = 100
LR         = 3e-4
SEED       = 42
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# Train / Val / Test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.85   # cumulative (0.70 to 0.85 = 15% val)

# ── Feature Config ───────────────────────────────────────────
WEATHER_COLS = ["tmpf", "relh", "feel", "sped", "p01m", "snowdepth"]

BASE_FEATS = [
    "tmpf", "relh", "feel", "sped", "p01m", "snowdepth",
    "charge_dur_mean", "session_count",
    "is_weekend", "month_sin", "month_cos",
    "weekday_sin", "weekday_cos", "quarter",
    "dayofyear_sin", "dayofyear_cos",
]

ROLLING_WINDOWS = [7, 14]
EMA_SPANS       = [7, 14]
LAG_FEATURES    = [1, 2, 3, 7, 14]

MIN_LOCATION_SAMPLES = 100
LOWER_OUTLIER_Q = 0.005
UPPER_OUTLIER_Q = 0.995

TARGET = "load_kwh"

# ── Location → Weather Station Mapping ──────────────────────
LOCATION_WEATHER_MAP = {
    # Queens → LGA
    "CSQ - Court Square Municipal Parking Garage": "LGA",
    "QBO - Queens Borough Hall Municipal Parking Garage": "LGA",
    "QFA - Queens Family Court Municipal Garage": "LGA",
    "Queens Borough Hall Municipal Parking Garage": "LGA",
    "Queens Family Court Municipal Garage": "LGA",
    "Queensboro Hall": "LGA",
    "Court Square Municipal Parking Garage": "LGA",
    # Bronx → JRB
    "JON - Jerome 190th Street Municipal Parking Garage": "JRB",
}

# ── Date Filter ──────────────────────────────────────────────
DATE_START = "2021-01-01"
DATE_END   = "2025-12-15"

# ── Required Columns ─────────────────────────────────────────
EV_REQUIRED_COLS = [
    "Date", "Station ID", "Location Name",
    "Connected Time", "Disconnected Time",
    "Charge Duration (min)", "Connected Duration (min)",
    "Energy Provided (kWh)",
]

ASOS_REQUIRED_COLS = ["station", "valid", "tmpf", "relh", "feel", "sped", "p01m", "snowdepth"]
