"""Generate Table 10 computational-efficiency outputs."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.efficiency import build_efficiency_table


def main():
    root = Path(__file__).resolve().parents[1]
    out_csv = root / "outputs" / "efficiency_results.csv"
    os.makedirs(out_csv.parent, exist_ok=True)

    df = build_efficiency_table()
    df.to_csv(out_csv, index=False)
    print(f"Saved Table 10 data: {out_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
