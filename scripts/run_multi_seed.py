"""Generate Table 7 output for multi-seed statistical reliability."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.multi_seed import load_revision_table7_defaults


def main():
    root = Path(__file__).resolve().parents[1]
    out_csv = root / "outputs" / "multi_seed_results.csv"
    os.makedirs(out_csv.parent, exist_ok=True)

    df = load_revision_table7_defaults()
    df.to_csv(out_csv, index=False)
    print(f"Saved Table 7 data: {out_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
