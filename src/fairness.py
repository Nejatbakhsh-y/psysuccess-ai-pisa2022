"""Fairness and subgroup audit utilities."""

from __future__ import annotations

import pandas as pd

from .evaluation import subgroup_metrics


def prediction_audit_table(
    data: pd.DataFrame,
    y_col: str,
    prob_col: str,
    subgroup_cols: list[str],
) -> dict[str, pd.DataFrame]:
    """Return subgroup metrics for each subgroup column."""
    return {col: subgroup_metrics(data, y_col, prob_col, col) for col in subgroup_cols if col in data.columns}
