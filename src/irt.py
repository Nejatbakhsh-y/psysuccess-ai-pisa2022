"""Lightweight Rasch/1PL estimation utilities.

This module implements a simple joint maximum-likelihood style Rasch estimator
for project prototyping. For a journal paper, confirm results with a dedicated
IRT package such as TAM/mirt in R or another validated psychometric library.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit


@dataclass
class RaschResult:
    theta: np.ndarray
    theta_se: np.ndarray
    item_difficulty: np.ndarray
    item_information: np.ndarray
    n_iter: int

    def student_table(self, id_values=None) -> pd.DataFrame:
        out = pd.DataFrame({
            "rasch_theta": self.theta,
            "rasch_theta_se": self.theta_se,
        })
        if id_values is not None:
            out.insert(0, "student_id", id_values)
        return out

    def item_table(self, item_names=None) -> pd.DataFrame:
        if item_names is None:
            item_names = [f"item_{j}" for j in range(len(self.item_difficulty))]
        return pd.DataFrame({
            "item": item_names,
            "rasch_b": self.item_difficulty,
            "rasch_item_information": self.item_information,
        })


def fit_rasch_jml(
    response_matrix: pd.DataFrame | np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-4,
    clip: float = 4.0,
) -> RaschResult:
    """Fit a simple Rasch model with missing responses ignored.

    Model: P(X_ij = 1 | theta_i, b_j) = logistic(theta_i - b_j)
    """
    x = _as_numeric_array(response_matrix)
    mask = ~np.isnan(x)
    n, m = x.shape
    if n == 0 or m == 0:
        raise ValueError("Response matrix must be non-empty.")

    # Initialize theta from row percent correct and b from item proportion correct.
    row_p = _safe_mean(x, axis=1)
    col_p = _safe_mean(x, axis=0)
    theta = _logit(np.clip(row_p, 0.01, 0.99))
    b = -_logit(np.clip(col_p, 0.01, 0.99))
    theta = _center(theta)
    b = _center(b)

    last_delta = np.inf
    for it in range(1, max_iter + 1):
        theta_old = theta.copy()
        b_old = b.copy()

        eta = theta[:, None] - b[None, :]
        p = expit(eta)
        pq = p * (1 - p)

        for i in range(n):
            obs = mask[i, :]
            if obs.sum() == 0:
                continue
            grad = np.nansum(x[i, obs] - p[i, obs])
            info = np.nansum(pq[i, obs])
            if info > 1e-8:
                theta[i] += grad / info

        theta = np.clip(theta, -clip, clip)
        theta = _center(theta)

        eta = theta[:, None] - b[None, :]
        p = expit(eta)
        pq = p * (1 - p)

        for j in range(m):
            obs = mask[:, j]
            if obs.sum() == 0:
                continue
            grad = np.nansum(p[obs, j] - x[obs, j])
            info = np.nansum(pq[obs, j])
            if info > 1e-8:
                b[j] += grad / info

        b = np.clip(b, -clip, clip)
        b = _center(b)

        last_delta = max(np.nanmax(np.abs(theta - theta_old)), np.nanmax(np.abs(b - b_old)))
        if last_delta < tol:
            break

    eta = theta[:, None] - b[None, :]
    p = expit(eta)
    pq = p * (1 - p)
    theta_info = np.where(mask, pq, np.nan)
    student_info = np.nansum(theta_info, axis=1)
    theta_se = np.where(student_info > 0, 1 / np.sqrt(student_info), np.nan)
    item_info = np.nansum(np.where(mask, pq, np.nan), axis=0)

    return RaschResult(theta=theta, theta_se=theta_se, item_difficulty=b, item_information=item_info, n_iter=it)


def _as_numeric_array(x: pd.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(x, pd.DataFrame):
        return x.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return np.asarray(x, dtype=float)


def _safe_mean(x: np.ndarray, axis: int) -> np.ndarray:
    out = np.nanmean(x, axis=axis)
    return np.where(np.isnan(out), 0.5, out)


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1 - p))


def _center(x: np.ndarray) -> np.ndarray:
    return x - np.nanmean(x)
