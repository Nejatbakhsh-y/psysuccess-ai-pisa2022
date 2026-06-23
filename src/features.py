"""Feature engineering utilities."""

from __future__ import annotations

import pandas as pd


def assemble_student_features(
    student_df: pd.DataFrame,
    id_column: str,
    candidate_columns: list[str],
    ctt_scores: pd.DataFrame | None = None,
    irt_scores: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create a student-level feature table from selected columns and psychometric features."""
    keep = [id_column] + [c for c in candidate_columns if c in student_df.columns]
    features = student_df[keep].copy()

    if ctt_scores is not None:
        features = features.merge(ctt_scores, left_on=id_column, right_index=True, how="left")
    if irt_scores is not None:
        irt = irt_scores.copy()
        if "student_id" in irt.columns:
            features = features.merge(irt, left_on=id_column, right_on="student_id", how="left")
            features = features.drop(columns=["student_id"], errors="ignore")
        else:
            features = features.merge(irt, left_index=True, right_index=True, how="left")
    return features


def encode_features(df: pd.DataFrame, id_column: str) -> pd.DataFrame:
    """One-hot encode categorical variables and retain numeric features."""
    x = df.drop(columns=[id_column], errors="ignore").copy()
    for c in x.columns:
        x[c] = pd.to_numeric(x[c], errors="ignore")
    return pd.get_dummies(x, drop_first=True, dummy_na=True)
