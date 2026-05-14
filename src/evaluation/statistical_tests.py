from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


def paired_significance_tests(baseline_abs_errors: np.ndarray, proposed_abs_errors: np.ndarray) -> dict:
    """Run paired two-sided t-test and Wilcoxon signed-rank test."""
    if baseline_abs_errors.shape != proposed_abs_errors.shape:
        raise ValueError("Both arrays must have identical shape for paired tests.")

    t_stat, t_p = ttest_rel(baseline_abs_errors, proposed_abs_errors, alternative="two-sided")
    try:
        w_stat, w_p = wilcoxon(baseline_abs_errors, proposed_abs_errors, alternative="two-sided")
    except ValueError:
        w_stat, w_p = np.nan, np.nan

    return {
        "t_stat": float(t_stat),
        "t_p": float(t_p),
        "w_stat": float(w_stat) if not np.isnan(w_stat) else np.nan,
        "w_p": float(w_p) if not np.isnan(w_p) else np.nan,
    }


def summarise_multi_seed_results(rows: Iterable[dict]) -> pd.DataFrame:
    """Convert row dicts into the manuscript-ready multi-seed table format."""
    df = pd.DataFrame(list(rows))
    if df.empty:
        return df
    return df[[
        "Model",
        "R2",
        "MAE (kWh)",
        "RMSE (kWh)",
        "MAPE (%)",
        "p vs. Proposed",
        "p_W",
    ]]
