"""Explainability helpers."""

from __future__ import annotations

import pandas as pd
from sklearn.inspection import permutation_importance


def permutation_importance_table(model, x: pd.DataFrame, y: pd.Series, n_repeats: int = 10, random_state: int = 42) -> pd.DataFrame:
    """Model-agnostic permutation importance table."""
    result = permutation_importance(model, x, y, n_repeats=n_repeats, random_state=random_state, scoring="roc_auc")
    return pd.DataFrame({
        "feature": x.columns,
        "importance_mean": result.importances_mean,
        "importance_sd": result.importances_std,
    }).sort_values("importance_mean", ascending=False)
