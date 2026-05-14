"""Generate Table 9b + Figure 20 outputs."""

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ablation import build_feature_group_ablation_table
from src.visualization.revision_plots import plot_figure20_feature_group_ablation


def _merge_to_combined(combined_path: Path, current_df: pd.DataFrame):
    if combined_path.exists():
        old = pd.read_csv(combined_path)
        merged = pd.concat([old, current_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["Section", "Variant"], keep="last")
    else:
        merged = current_df
    merged.to_csv(combined_path, index=False)


def main():
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "outputs" / "ablation"
    fig_dir = root / "outputs" / "revision_figures"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    df = build_feature_group_ablation_table()
    table_path = out_dir / "table9_feature_groups.csv"
    df.to_csv(table_path, index=False)

    fig20_path = fig_dir / "figure_20_feature_group_ablation.png"
    plot_figure20_feature_group_ablation(df, str(fig20_path))

    _merge_to_combined(out_dir / "table9_combined.csv", df)

    print(f"Saved Table 9b: {table_path}")
    print(f"Saved Figure 20: {fig20_path}")


if __name__ == "__main__":
    main()
