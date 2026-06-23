"""Validity-gating rules for model predictions."""

from __future__ import annotations

import pandas as pd


def apply_validity_gate(
    df: pd.DataFrame,
    valid_item_count_col: str = "valid_item_count",
    irt_se_col: str = "rasch_theta_se",
    probability_col: str = "predicted_success_probability",
    min_valid_item_count: int = 8,
    max_irt_se: float = 1.25,
    probability_margin: float = 0.05,
) -> pd.DataFrame:
    """Flag predictions as valid or review-needed using measurement rules."""
    out = df.copy()
    reasons = []
    valid_flags = []
    for _, row in out.iterrows():
        row_reasons = []
        if valid_item_count_col in out.columns and pd.notna(row.get(valid_item_count_col)):
            if row[valid_item_count_col] < min_valid_item_count:
                row_reasons.append("too_few_valid_item_responses")
        if irt_se_col in out.columns and pd.notna(row.get(irt_se_col)):
            if row[irt_se_col] > max_irt_se:
                row_reasons.append("high_irt_standard_error")
        if probability_col in out.columns and pd.notna(row.get(probability_col)):
            p = row[probability_col]
            if abs(p - 0.5) < probability_margin:
                row_reasons.append("prediction_near_decision_boundary")
        valid_flags.append(len(row_reasons) == 0)
        reasons.append(";".join(row_reasons) if row_reasons else "valid")
    out["validity_gate_pass"] = valid_flags
    out["validity_gate_reason"] = reasons
    return out
