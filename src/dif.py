"""Differential item functioning utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def logistic_dif(
    response_matrix: pd.DataFrame,
    total_score: pd.Series,
    group: pd.Series,
    min_group_n: int = 30,
) -> pd.DataFrame:
    """Run logistic-regression DIF screening item by item.

    For each item, fit: item ~ total_score + group + total_score:group.
    The group coefficient screens uniform DIF; the interaction screens nonuniform DIF.
    """
    group_num = _binary_group(group)
    rows = []
    score = pd.to_numeric(total_score, errors="coerce")
    score_z = (score - score.mean()) / score.std(ddof=0)

    for item in response_matrix.columns:
        y = pd.to_numeric(response_matrix[item], errors="coerce")
        data = pd.DataFrame({
            "y": y,
            "score": score_z,
            "group": group_num,
        }).dropna()
        if data["y"].nunique() < 2 or data["group"].nunique() < 2:
            rows.append(_empty_item_result(item, "insufficient variation"))
            continue
        counts = data["group"].value_counts()
        if counts.min() < min_group_n:
            rows.append(_empty_item_result(item, "insufficient group size"))
            continue
        data["interaction"] = data["score"] * data["group"]
        x = sm.add_constant(data[["score", "group", "interaction"]])
        try:
            model = sm.Logit(data["y"], x).fit(disp=False, maxiter=100)
            rows.append({
                "item": item,
                "n": int(len(data)),
                "group_coef_uniform_dif": model.params.get("group", np.nan),
                "group_pvalue_uniform_dif": model.pvalues.get("group", np.nan),
                "interaction_coef_nonuniform_dif": model.params.get("interaction", np.nan),
                "interaction_pvalue_nonuniform_dif": model.pvalues.get("interaction", np.nan),
                "status": "ok",
            })
        except Exception as exc:  # noqa: BLE001
            rows.append(_empty_item_result(item, f"model failed: {exc}"))
    out = pd.DataFrame(rows)
    out["uniform_dif_flag_0_01"] = out["group_pvalue_uniform_dif"] < 0.01
    out["nonuniform_dif_flag_0_01"] = out["interaction_pvalue_nonuniform_dif"] < 0.01
    return out


def mantel_haenszel_dif(
    response_matrix: pd.DataFrame,
    total_score: pd.Series,
    group: pd.Series,
) -> pd.DataFrame:
    """Simple Mantel-Haenszel common odds-ratio DIF screen for binary groups."""
    group_num = _binary_group(group)
    score = pd.to_numeric(total_score, errors="coerce").round().astype("Int64")
    rows = []
    for item in response_matrix.columns:
        y = pd.to_numeric(response_matrix[item], errors="coerce")
        data = pd.DataFrame({"y": y, "score": score, "group": group_num}).dropna()
        if data["y"].nunique() < 2 or data["group"].nunique() < 2:
            rows.append({"item": item, "mh_common_odds_ratio": np.nan, "mh_delta": np.nan, "status": "insufficient variation"})
            continue
        num = 0.0
        den = 0.0
        for _, stratum in data.groupby("score"):
            a = ((stratum["group"] == 1) & (stratum["y"] == 1)).sum()
            b = ((stratum["group"] == 1) & (stratum["y"] == 0)).sum()
            c = ((stratum["group"] == 0) & (stratum["y"] == 1)).sum()
            d = ((stratum["group"] == 0) & (stratum["y"] == 0)).sum()
            n = a + b + c + d
            if n > 0:
                num += (a * d) / n
                den += (b * c) / n
        odds = num / den if den > 0 else np.nan
        mh_delta = -2.35 * np.log(odds) if odds and odds > 0 else np.nan
        rows.append({"item": item, "mh_common_odds_ratio": odds, "mh_delta": mh_delta, "status": "ok"})
    return pd.DataFrame(rows)


def _binary_group(group: pd.Series) -> pd.Series:
    g = group.copy()
    if pd.api.types.is_numeric_dtype(g):
        values = sorted(g.dropna().unique().tolist())
        if len(values) > 2:
            # Use top two categories by frequency for a conservative starter analysis.
            top = g.value_counts().head(2).index.tolist()
            g = g.where(g.isin(top), np.nan)
        values = sorted(g.dropna().unique().tolist())
        mapping = {values[0]: 0, values[1]: 1} if len(values) >= 2 else {}
        return g.map(mapping)
    top = g.astype("object").value_counts().head(2).index.tolist()
    mapping = {top[0]: 0, top[1]: 1} if len(top) >= 2 else {}
    return g.map(mapping)


def _empty_item_result(item: str, status: str) -> dict:
    return {
        "item": item,
        "n": 0,
        "group_coef_uniform_dif": np.nan,
        "group_pvalue_uniform_dif": np.nan,
        "interaction_coef_nonuniform_dif": np.nan,
        "interaction_pvalue_nonuniform_dif": np.nan,
        "status": status,
    }
