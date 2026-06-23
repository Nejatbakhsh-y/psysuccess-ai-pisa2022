"""Machine-learning model training utilities."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .evaluation import classification_metrics


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = x.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in x.columns if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols),
    ])


def candidate_models(random_state: int = 42) -> dict:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=random_state, class_weight="balanced_subsample", n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=random_state),
    }


def train_model_suite(
    features: pd.DataFrame,
    labels: pd.Series,
    id_column: str | None = None,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Train baseline candidate models and return metrics plus fitted models."""
    x = features.drop(columns=[id_column], errors="ignore") if id_column else features.copy()
    y = labels.astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )
    rows = []
    fitted = {}
    for name, clf in candidate_models(random_state).items():
        pipe = Pipeline([
            ("preprocessor", build_preprocessor(x_train)),
            ("classifier", clf),
        ])
        pipe.fit(x_train, y_train)
        prob = pipe.predict_proba(x_test)[:, 1]
        metrics = classification_metrics(y_test, prob)
        metrics["model"] = name
        metrics["n_train"] = len(x_train)
        metrics["n_test"] = len(x_test)
        rows.append(metrics)
        fitted[name] = pipe
    return pd.DataFrame(rows).sort_values("auc", ascending=False), fitted
