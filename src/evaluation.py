"""Evaluation metrics for student-success prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Compute standard classification and probability metrics."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "brier": brier_score_loss(y_true, y_prob),
        "average_precision": average_precision_score(y_true, y_prob),
    }
    if len(np.unique(y_true)) == 2:
        out["auc"] = roc_auc_score(y_true, y_prob)
    else:
        out["auc"] = np.nan
    return out


def subgroup_metrics(df: pd.DataFrame, y_col: str, prob_col: str, group_col: str) -> pd.DataFrame:
    """Compute metrics by subgroup."""
    rows = []
    for group_value, g in df.dropna(subset=[group_col]).groupby(group_col):
        if len(g) < 20 or g[y_col].nunique() < 2:
            continue
        metrics = classification_metrics(g[y_col], g[prob_col])
        metrics[group_col] = group_value
        metrics["n"] = len(g)
        rows.append(metrics)
    return pd.DataFrame(rows)
