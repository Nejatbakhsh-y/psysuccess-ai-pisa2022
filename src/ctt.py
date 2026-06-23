"""Classical Test Theory functions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def kr20(response_matrix: pd.DataFrame | np.ndarray) -> float:
    """Compute KR-20/Cronbach alpha for dichotomous item data.

    Missing item values are mean-imputed itemwise for reliability estimation.
    For final publication, sensitivity checks should report missing-data handling.
    """
    x = _as_numeric_matrix(response_matrix)
    if x.shape[1] < 2:
        return float("nan")
    x = _mean_impute_columns(x)
    k = x.shape[1]
    item_vars = x.var(axis=0, ddof=1)
    total_var = x.sum(axis=1).var(ddof=1)
    if total_var <= 0:
        return float("nan")
    return float((k / (k - 1)) * (1 - item_vars.sum() / total_var))


def item_difficulty(response_matrix: pd.DataFrame | np.ndarray) -> pd.Series:
    """Item difficulty as proportion correct."""
    df = _as_dataframe(response_matrix)
    return df.apply(pd.to_numeric, errors="coerce").mean(axis=0)


def item_discrimination(response_matrix: pd.DataFrame | np.ndarray) -> pd.Series:
    """Corrected item-total correlation for each item."""
    df = _as_dataframe(response_matrix).apply(pd.to_numeric, errors="coerce")
    out = {}
    for c in df.columns:
        item = df[c]
        total_minus_item = df.drop(columns=[c]).sum(axis=1, skipna=True)
        valid = item.notna() & total_minus_item.notna()
        if valid.sum() < 3 or item[valid].nunique() < 2 or total_minus_item[valid].nunique() < 2:
            out[c] = np.nan
        else:
            out[c] = item[valid].corr(total_minus_item[valid])
    return pd.Series(out, name="item_total_correlation")


def student_raw_scores(response_matrix: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """Compute raw score, valid item count, and percent correct."""
    df = _as_dataframe(response_matrix).apply(pd.to_numeric, errors="coerce")
    valid_count = df.notna().sum(axis=1)
    raw_score = df.sum(axis=1, skipna=True)
    percent_correct = raw_score / valid_count.replace(0, np.nan)
    return pd.DataFrame({
        "raw_score": raw_score,
        "valid_item_count": valid_count,
        "percent_correct": percent_correct,
    })


def ctt_item_table(response_matrix: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """Return item-level CTT statistics."""
    df = _as_dataframe(response_matrix)
    return pd.DataFrame({
        "item": df.columns,
        "difficulty_p_value": item_difficulty(df).values,
        "discrimination_item_total_corr": item_discrimination(df).values,
        "missing_rate": df.isna().mean(axis=0).values,
    })


def _as_dataframe(x: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.copy()
    return pd.DataFrame(x)


def _as_numeric_matrix(x: pd.DataFrame | np.ndarray) -> np.ndarray:
    return _as_dataframe(x).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def _mean_impute_columns(x: np.ndarray) -> np.ndarray:
    out = x.copy().astype(float)
    means = np.nanmean(out, axis=0)
    inds = np.where(np.isnan(out))
    out[inds] = np.take(means, inds[1])
    return out
