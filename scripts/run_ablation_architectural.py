"""Generate Table 9a architectural-ablation outputs."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ablation import build_architectural_ablation_table


def main():
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "outputs" / "ablation"
    os.makedirs(out_dir, exist_ok=True)

    df = build_architectural_ablation_table()
    df.to_csv(out_dir / "table9_architectural.csv", index=False)

    # Also keep one combined file if it does not exist yet.
    combined = out_dir / "table9_combined.csv"
    if not combined.exists():
        df.to_csv(combined, index=False)
    print(f"Saved Table 9a outputs to: {out_dir}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
