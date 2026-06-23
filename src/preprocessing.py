"""Preprocessing utilities for PISA-style student, school, and item files."""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with stripped string column names."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def find_pv_columns(df: pd.DataFrame, domain: str = "MATH") -> list[str]:
    """Find PISA plausible-value columns for a domain.

    Example: PV1MATH, PV2MATH, ..., PV10MATH.
    """
    pattern = re.compile(rf"^PV\d+{re.escape(domain)}$", re.IGNORECASE)
    return sorted([c for c in df.columns if pattern.match(c)], key=lambda x: int(re.findall(r"\d+", x)[0]))


def find_item_columns(
    df: pd.DataFrame,
    id_columns: Iterable[str] = (),
    prefixes: Iterable[str] = ("CM", "CR", "DM", "M"),
    min_nonmissing_rate: float = 0.20,
    max_missing_rate: float = 0.80,
) -> list[str]:
    """Heuristically detect scored dichotomous item columns.

    This is intentionally conservative. Users should verify detected items
    against the PISA codebook before final analysis.
    """
    exclude = set(id_columns)
    item_cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if not any(str(c).upper().startswith(p.upper()) for p in prefixes):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        valid_rate = s.notna().mean()
        missing_rate = 1.0 - valid_rate
        values = set(s.dropna().unique().tolist())
        if valid_rate >= min_nonmissing_rate and missing_rate <= max_missing_rate and values.issubset({0, 1}):
            item_cols.append(c)
    return item_cols


def make_success_labels(
    student_df: pd.DataFrame,
    pv_columns: list[str],
    threshold: float,
    id_column: str,
) -> pd.DataFrame:
    """Construct binary and probability success labels from plausible values."""
    if not pv_columns:
        raise ValueError("No plausible-value columns were supplied.")
    work = student_df[[id_column] + pv_columns].copy()
    for c in pv_columns:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    pv = work[pv_columns]
    out = pd.DataFrame({id_column: work[id_column]})
    out["pv_mean"] = pv.mean(axis=1)
    out["pv_sd"] = pv.std(axis=1)
    out["success_probability"] = (pv >= threshold).mean(axis=1)
    out["success_label"] = (out["success_probability"] >= 0.50).astype(int)
    return out


def binary_item_matrix(df: pd.DataFrame, item_columns: list[str], id_column: str | None = None) -> pd.DataFrame:
    """Return a numeric binary item matrix, preserving ID column if requested."""
    out = df[[id_column] + item_columns].copy() if id_column else df[item_columns].copy()
    for c in item_columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out.loc[~out[c].isin([0, 1]), c] = np.nan
    return out
