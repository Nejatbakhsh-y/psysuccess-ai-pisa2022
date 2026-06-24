# scripts/09_model_diagnostics_fairness.py
"""
Step H: Model diagnostics, interpretation, and fairness/measurement checks

This script:
1. Loads the clean PISA student analysis dataset.
2. Builds diagnostic student-success prediction models.
3. Produces model diagnostics:
   - ROC curve
   - precision-recall curve
   - confusion matrix
   - calibration curve
   - probability-decile summary
4. Produces interpretation outputs:
   - permutation feature importance
5. Produces fairness / measurement checks:
   - subgroup metrics by gender, country, and ESCS quartile when available
   - score-prediction alignment checks when math_score_mean_pv is available
6. Saves all outputs to reports/ and figures/.
"""

from pathlib import Path
import warnings
import re

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RANDOM_STATE = 42
MAX_MODEL_ROWS = 120_000
MIN_GROUP_N = 50

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "model_diagnostics_fairness"
FIGURE_DIR = ROOT / "figures" / "model_diagnostics_fairness"
MODEL_DIR = ROOT / "models"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def clean_name(name: str) -> str:
    """Create a safe filename component."""
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def find_dataset() -> Path:
    """
    Find the clean student dataset.

    The preferred file is:
    data/processed/pisa_student_analysis_dataset.csv

    If that exact file does not exist, the script searches for any CSV
    in data/processed/ containing the column student_success.
    """
    candidates = [
        DATA_DIR / "pisa_student_analysis_dataset.csv",
        DATA_DIR / "pisa_student_clean.csv",
        DATA_DIR / "pisa2022_student_clean.csv",
        DATA_DIR / "student_analysis_dataset.csv",
        DATA_DIR / "clean_pisa_student_dataset.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    if DATA_DIR.exists():
        for path in DATA_DIR.glob("*.csv"):
            try:
                sample = pd.read_csv(path, nrows=10)
                cols_lower = [c.lower() for c in sample.columns]
                if "student_success" in cols_lower:
                    return path
            except Exception:
                continue

    raise FileNotFoundError(
        "Could not find the clean student dataset. Expected a CSV in data/processed/ "
        "with a student_success column."
    )


def find_column(df: pd.DataFrame, candidates) -> str | None:
    """Find a column using case-insensitive matching."""
    lower_map = {c.lower(): c for c in df.columns}

    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    return None


def make_binary_target(series: pd.Series) -> pd.Series:
    """Convert the target column to 0/1."""
    s = series.copy()

    if pd.api.types.is_numeric_dtype(s):
        unique_values = sorted(pd.Series(s.dropna().unique()).tolist())
        if set(unique_values).issubset({0, 1}):
            return s.astype("Int64")
        if set(unique_values).issubset({1, 2}):
            return s.map({1: 0, 2: 1}).astype("Int64")
        return (s > s.median()).astype("Int64")

    s_lower = s.astype(str).str.strip().str.lower()

    positive_values = {
        "1", "yes", "y", "true", "success", "successful",
        "pass", "passed", "high", "proficient", "above"
    }

    negative_values = {
        "0", "no", "n", "false", "failure", "fail",
        "failed", "low", "not proficient", "below"
    }

    mapped = pd.Series(index=s.index, dtype="float")

    mapped[s_lower.isin(positive_values)] = 1
    mapped[s_lower.isin(negative_values)] = 0

    return mapped.astype("Int64")


def choose_onehot_encoder():
    """Handle sklearn version differences."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def safe_auc(y_true, y_prob):
    """Compute ROC AUC only when both classes exist."""
    if pd.Series(y_true).nunique() < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob)


def safe_log_loss(y_true, y_prob):
    """Compute log loss safely."""
    try:
        return log_loss(y_true, y_prob, labels=[0, 1])
    except Exception:
        return np.nan


def false_positive_rate(y_true, y_pred):
    """Compute false positive rate."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    denominator = fp + tn
    if denominator == 0:
        return np.nan
    return fp / denominator


def select_threshold_from_training(y_train, p_train):
    """
    Select a classification threshold using training predictions.

    The threshold is chosen to maximize F1 on the training set.
    The test set is not used for threshold selection.
    """
    precision, recall, thresholds = precision_recall_curve(y_train, p_train)

    if len(thresholds) == 0:
        return 0.50

    f1_values = []
    for p, r in zip(precision[:-1], recall[:-1]):
        if p + r == 0:
            f1_values.append(0)
        else:
            f1_values.append(2 * p * r / (p + r))

    best_idx = int(np.nanargmax(f1_values))
    return float(thresholds[best_idx])


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """Train and evaluate one model."""
    model.fit(X_train, y_train)

    p_train = model.predict_proba(X_train)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]

    threshold = select_threshold_from_training(y_train, p_train)
    y_pred = (p_test >= threshold).astype(int)

    metrics = {
        "model": name,
        "threshold": threshold,
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred, zero_division=0),
        "test_recall": recall_score(y_test, y_pred, zero_division=0),
        "test_f1": f1_score(y_test, y_pred, zero_division=0),
        "test_roc_auc": safe_auc(y_test, p_test),
        "test_average_precision": average_precision_score(y_test, p_test),
        "test_brier_score": brier_score_loss(y_test, p_test),
        "test_log_loss": safe_log_loss(y_test, p_test),
        "train_roc_auc": safe_auc(y_train, p_train),
        "train_average_precision": average_precision_score(y_train, p_train),
    }

    metrics["roc_auc_gap_train_minus_test"] = (
        metrics["train_roc_auc"] - metrics["test_roc_auc"]
        if pd.notna(metrics["train_roc_auc"]) and pd.notna(metrics["test_roc_auc"])
        else np.nan
    )

    return model, p_test, y_pred, metrics


def save_roc_curve(y_test, p_test):
    fpr, tpr, _ = roc_curve(y_test, p_test)
    auc_value = safe_auc(y_test, p_test)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc_value:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC Curve: Student-Success Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "09_roc_curve.png", dpi=300)
    plt.close()


def save_precision_recall_curve(y_test, p_test):
    precision, recall, _ = precision_recall_curve(y_test, p_test)
    ap_value = average_precision_score(y_test, p_test)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"Average precision = {ap_value:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve: Student-Success Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "09_precision_recall_curve.png", dpi=300)
    plt.close()


def save_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "09_confusion_matrix.png", dpi=300)
    plt.close()


def save_calibration_curve(y_test, p_test):
    df_cal = pd.DataFrame({"y": y_test, "p": p_test}).copy()
    df_cal["probability_bin"] = pd.qcut(
        df_cal["p"],
        q=min(10, df_cal["p"].nunique()),
        duplicates="drop"
    )

    calibration = (
        df_cal.groupby("probability_bin", observed=True)
        .agg(
            n=("y", "size"),
            mean_predicted_probability=("p", "mean"),
            observed_success_rate=("y", "mean"),
        )
        .reset_index()
    )

    calibration.to_csv(REPORT_DIR / "09_calibration_by_decile.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.plot(
        calibration["mean_predicted_probability"],
        calibration["observed_success_rate"],
        marker="o",
        label="Model"
    )
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed success rate")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "09_calibration_curve.png", dpi=300)
    plt.close()

    return calibration


def make_decile_summary(y_test, p_test):
    df_decile = pd.DataFrame({"student_success": y_test, "predicted_probability": p_test})

    df_decile["risk_decile"] = pd.qcut(
        df_decile["predicted_probability"],
        q=min(10, df_decile["predicted_probability"].nunique()),
        labels=False,
        duplicates="drop"
    )

    df_decile["risk_decile"] = df_decile["risk_decile"] + 1

    decile_summary = (
        df_decile.groupby("risk_decile", observed=True)
        .agg(
            n=("student_success", "size"),
            mean_predicted_probability=("predicted_probability", "mean"),
            observed_success_rate=("student_success", "mean"),
            min_predicted_probability=("predicted_probability", "min"),
            max_predicted_probability=("predicted_probability", "max"),
        )
        .reset_index()
        .sort_values("risk_decile")
    )

    decile_summary.to_csv(REPORT_DIR / "09_success_rate_by_probability_decile.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(
        decile_summary["risk_decile"],
        decile_summary["observed_success_rate"],
        marker="o",
        label="Observed success rate"
    )
    plt.plot(
        decile_summary["risk_decile"],
        decile_summary["mean_predicted_probability"],
        marker="o",
        label="Mean predicted probability"
    )
    plt.xlabel("Predicted-probability decile")
    plt.ylabel("Rate / probability")
    plt.title("Success Rate by Predicted-Probability Decile")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "09_success_rate_by_probability_decile.png", dpi=300)
    plt.close()

    return decile_summary


def compute_group_metrics(meta_df, y_test, p_test, y_pred, group_col):
    """Compute fairness diagnostics for one subgroup variable."""
    temp = meta_df.copy()
    temp["actual"] = np.asarray(y_test)
    temp["pred_prob"] = np.asarray(p_test)
    temp["pred_label"] = np.asarray(y_pred)

    temp[group_col] = temp[group_col].astype("object")
    temp[group_col] = temp[group_col].where(temp[group_col].notna(), "Missing")

    overall_pred_positive_rate = temp["pred_label"].mean()
    overall_recall = recall_score(temp["actual"], temp["pred_label"], zero_division=0)
    overall_fpr = false_positive_rate(temp["actual"], temp["pred_label"])
    overall_auc = safe_auc(temp["actual"], temp["pred_prob"])
    overall_avg_prob = temp["pred_prob"].mean()

    rows = []

    for group_value, g in temp.groupby(group_col, observed=True):
        if len(g) < MIN_GROUP_N:
            continue

        group_auc = safe_auc(g["actual"], g["pred_prob"])
        group_fpr = false_positive_rate(g["actual"], g["pred_label"])

        row = {
            "group_variable": group_col,
            "group_value": str(group_value),
            "n": len(g),
            "actual_success_rate": g["actual"].mean(),
            "average_predicted_probability": g["pred_prob"].mean(),
            "predicted_positive_rate": g["pred_label"].mean(),
            "accuracy": accuracy_score(g["actual"], g["pred_label"]),
            "precision": precision_score(g["actual"], g["pred_label"], zero_division=0),
            "recall_equal_opportunity": recall_score(g["actual"], g["pred_label"], zero_division=0),
            "false_positive_rate": group_fpr,
            "f1": f1_score(g["actual"], g["pred_label"], zero_division=0),
            "roc_auc": group_auc,
        }

        row["predicted_positive_rate_gap"] = (
            row["predicted_positive_rate"] - overall_pred_positive_rate
        )
        row["recall_gap_equal_opportunity"] = (
            row["recall_equal_opportunity"] - overall_recall
        )
        row["false_positive_rate_gap"] = (
            row["false_positive_rate"] - overall_fpr
            if pd.notna(row["false_positive_rate"]) and pd.notna(overall_fpr)
            else np.nan
        )
        row["roc_auc_gap"] = (
            row["roc_auc"] - overall_auc
            if pd.notna(row["roc_auc"]) and pd.notna(overall_auc)
            else np.nan
        )
        row["average_probability_gap"] = (
            row["average_predicted_probability"] - overall_avg_prob
        )

        rows.append(row)

    return pd.DataFrame(rows)


def save_group_fairness_plot(fairness_df, group_col):
    subset = fairness_df[fairness_df["group_variable"] == group_col].copy()

    if subset.empty:
        return

    subset = subset.sort_values("n", ascending=False).head(20)
    subset = subset.sort_values("recall_gap_equal_opportunity")

    plt.figure(figsize=(9, 6))
    plt.barh(subset["group_value"], subset["recall_gap_equal_opportunity"])
    plt.axvline(0, linestyle="--")
    plt.xlabel("Recall gap versus overall recall")
    plt.ylabel(group_col)
    plt.title(f"Equal Opportunity Check by {group_col}")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"09_fairness_equal_opportunity_{clean_name(group_col)}.png", dpi=300)
    plt.close()


def save_probability_distribution_plot(y_test, p_test):
    temp = pd.DataFrame({"actual": y_test, "pred_prob": p_test})

    plt.figure(figsize=(8, 5))
    plt.hist(
        temp.loc[temp["actual"] == 0, "pred_prob"],
        bins=30,
        alpha=0.6,
        label="Actual 0"
    )
    plt.hist(
        temp.loc[temp["actual"] == 1, "pred_prob"],
        bins=30,
        alpha=0.6,
        label="Actual 1"
    )
    plt.xlabel("Predicted probability of student success")
    plt.ylabel("Number of students")
    plt.title("Predicted Probability Distribution by Actual Outcome")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "09_probability_distribution_by_actual_outcome.png", dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

dataset_path = find_dataset()
df = pd.read_csv(dataset_path, low_memory=False)

print(f"Loaded dataset: {dataset_path}")
print(f"Original shape: {df.shape}")

target_col = find_column(df, ["student_success"])

if target_col is None:
    raise ValueError("The dataset must contain a student_success column.")

df[target_col] = make_binary_target(df[target_col])
df = df[df[target_col].notna()].copy()
df[target_col] = df[target_col].astype(int)

if df[target_col].nunique() != 2:
    raise ValueError("student_success must contain two classes after cleaning.")

# Important columns for subgroup and measurement checks
gender_col = find_column(df, ["gender", "student_gender", "sex", "st004d01t", "ST004D01T"])
country_col = find_column(df, ["country", "cnt", "CNT"])
escs_col = find_column(df, ["escs", "ESCS"])
school_col = find_column(df, ["school_id", "cntschid", "CNTSCHID"])
score_col = find_column(
    df,
    [
        "math_score_mean_pv",
        "math_score",
        "mean_math_score",
        "pisa_math_score",
        "math_achievement",
    ],
)

# Create ESCS quartile group if ESCS exists
if escs_col is not None:
    df["escs_quartile"] = pd.qcut(
        df[escs_col],
        q=4,
        labels=["Q1_lowest_ESCS", "Q2", "Q3", "Q4_highest_ESCS"],
        duplicates="drop",
    )

# Stratified sample for manageable runtime
if len(df) > MAX_MODEL_ROWS:
    df_sampled, _ = train_test_split(
        df,
        train_size=MAX_MODEL_ROWS,
        random_state=RANDOM_STATE,
        stratify=df[target_col],
    )
    df_model = df_sampled.copy()
else:
    df_model = df.copy()

print(f"Modeling shape after optional sampling: {df_model.shape}")


# ---------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------

exclude_exact = {target_col}

if gender_col is not None:
    exclude_exact.add(gender_col)

if country_col is not None:
    exclude_exact.add(country_col)

if school_col is not None:
    exclude_exact.add(school_col)

exclude_exact.add("escs_quartile")

# Exclude obvious leakage, IDs, weights, plausible-value scores, and outcome scores.
exclude_keywords = [
    "student_success",
    "math_score",
    "reading_score",
    "science_score",
    "score_mean",
    "plausible",
    "pv1",
    "pv2",
    "pv3",
    "pv4",
    "pv5",
    "pv6",
    "pv7",
    "pv8",
    "pv9",
    "pv10",
    "cntstuid",
    "cntschid",
    "student_id",
    "school_id",
    "id",
    "weight",
    "w_fstuwt",
]

feature_cols = []

for col in df_model.columns:
    col_lower = col.lower()

    if col in exclude_exact:
        continue

    if any(key in col_lower for key in exclude_keywords):
        continue

    if df_model[col].isna().mean() > 0.90:
        continue

    if df_model[col].nunique(dropna=True) <= 1:
        continue

    # Avoid extremely high-cardinality categorical variables.
    if (
        not pd.api.types.is_numeric_dtype(df_model[col])
        and df_model[col].nunique(dropna=True) > 100
    ):
        continue

    feature_cols.append(col)

if not feature_cols:
    raise ValueError(
        "No usable feature columns found. Check the clean dataset columns."
    )

print(f"Number of model features: {len(feature_cols)}")
print("First 20 model features:")
print(feature_cols[:20])

X = df_model[feature_cols].copy()
y = df_model[target_col].copy()

numeric_features = [
    c for c in feature_cols if pd.api.types.is_numeric_dtype(X[c])
]

categorical_features = [
    c for c in feature_cols if c not in numeric_features
]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", choose_onehot_encoder()),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop",
)


# ---------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------

train_idx, test_idx = train_test_split(
    df_model.index,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=df_model[target_col],
)

X_train = X.loc[train_idx]
X_test = X.loc[test_idx]
y_train = y.loc[train_idx]
y_test = y.loc[test_idx]

print(f"Training rows: {len(X_train)}")
print(f"Test rows: {len(X_test)}")


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

models = {
    "logistic_regression_balanced": Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    ),
    "random_forest_balanced": Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=RANDOM_STATE,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    min_samples_leaf=5,
                ),
            ),
        ]
    ),
}

all_metrics = []
trained_models = {}
test_probabilities = {}
test_predictions = {}

for model_name, model in models.items():
    print(f"\nTraining model: {model_name}")

    fitted_model, p_test, y_pred, metrics = evaluate_model(
        model_name,
        model,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    trained_models[model_name] = fitted_model
    test_probabilities[model_name] = p_test
    test_predictions[model_name] = y_pred
    all_metrics.append(metrics)

metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv(REPORT_DIR / "09_model_diagnostics_metrics.csv", index=False)

# Select best model by ROC AUC, then average precision if needed.
metrics_for_selection = metrics_df.copy()
metrics_for_selection["selection_score"] = metrics_for_selection["test_roc_auc"]

if metrics_for_selection["selection_score"].isna().all():
    metrics_for_selection["selection_score"] = metrics_for_selection["test_average_precision"]

best_model_name = (
    metrics_for_selection.sort_values("selection_score", ascending=False)
    .iloc[0]["model"]
)

best_model = trained_models[best_model_name]
p_test_best = test_probabilities[best_model_name]
y_pred_best = test_predictions[best_model_name]
best_metrics = metrics_df[metrics_df["model"] == best_model_name].iloc[0].to_dict()

print(f"\nBest diagnostic model: {best_model_name}")

joblib.dump(best_model, MODEL_DIR / "09_best_diagnostic_student_success_model.joblib")


# ---------------------------------------------------------------------
# Diagnostic figures and tables
# ---------------------------------------------------------------------

save_roc_curve(y_test, p_test_best)
save_precision_recall_curve(y_test, p_test_best)
save_confusion_matrix(y_test, y_pred_best)
save_probability_distribution_plot(y_test, p_test_best)

calibration_df = save_calibration_curve(y_test, p_test_best)
decile_summary = make_decile_summary(y_test, p_test_best)


# ---------------------------------------------------------------------
# Permutation feature importance
# ---------------------------------------------------------------------

print("\nComputing permutation importance...")

importance_sample_size = min(3000, len(X_test))
importance_idx = X_test.sample(
    n=importance_sample_size,
    random_state=RANDOM_STATE,
).index

try:
    perm = permutation_importance(
        best_model,
        X_test.loc[importance_idx],
        y_test.loc[importance_idx],
        n_repeats=3,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
        n_jobs=-1,
    )

    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    importance_df.to_csv(
        REPORT_DIR / "09_permutation_feature_importance.csv",
        index=False,
    )

    top_importance = importance_df.head(20).sort_values("importance_mean")

    plt.figure(figsize=(9, 7))
    plt.barh(top_importance["feature"], top_importance["importance_mean"])
    plt.xlabel("Permutation importance: decrease in ROC AUC")
    plt.ylabel("Feature")
    plt.title("Top Permutation Feature Importance")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "09_permutation_feature_importance.png", dpi=300)
    plt.close()

except Exception as exc:
    print(f"Permutation importance could not be computed: {exc}")
    importance_df = pd.DataFrame()


# ---------------------------------------------------------------------
# Fairness and measurement checks
# ---------------------------------------------------------------------

meta_cols = []

for col in [gender_col, country_col, escs_col, "escs_quartile", score_col]:
    if col is not None and col in df_model.columns and col not in meta_cols:
        meta_cols.append(col)

meta_test = df_model.loc[test_idx, meta_cols].copy()

fairness_frames = []

for group_col in [gender_col, country_col, "escs_quartile"]:
    if group_col is not None and group_col in meta_test.columns:
        group_metrics = compute_group_metrics(
            meta_test,
            y_test,
            p_test_best,
            y_pred_best,
            group_col,
        )

        if not group_metrics.empty:
            fairness_frames.append(group_metrics)
            save_group_fairness_plot(group_metrics, group_col)

if fairness_frames:
    fairness_df = pd.concat(fairness_frames, ignore_index=True)
else:
    fairness_df = pd.DataFrame()

fairness_df.to_csv(REPORT_DIR / "09_subgroup_fairness_metrics.csv", index=False)

if not fairness_df.empty:
    fairness_flags = fairness_df[
        (fairness_df["n"] >= MIN_GROUP_N)
        & (
            (fairness_df["recall_gap_equal_opportunity"].abs() >= 0.10)
            | (fairness_df["predicted_positive_rate_gap"].abs() >= 0.10)
            | (fairness_df["false_positive_rate_gap"].abs() >= 0.10)
        )
    ].copy()

    fairness_flags = fairness_flags.sort_values(
        ["group_variable", "n"],
        ascending=[True, False],
    )

else:
    fairness_flags = pd.DataFrame()

fairness_flags.to_csv(REPORT_DIR / "09_fairness_flags.csv", index=False)


# ---------------------------------------------------------------------
# Measurement check: alignment between predicted probability and math score
# ---------------------------------------------------------------------

measurement_rows = []

if score_col is not None and score_col in meta_test.columns:
    temp_measure = meta_test.copy()
    temp_measure["student_success"] = y_test.values
    temp_measure["predicted_probability"] = p_test_best

    valid = temp_measure[[score_col, "predicted_probability", "student_success"]].dropna()

    if not valid.empty:
        corr_prob_score = valid[score_col].corr(valid["predicted_probability"])
        corr_success_score = valid[score_col].corr(valid["student_success"])

        measurement_rows.append(
            {
                "check": "correlation_predicted_probability_with_math_score",
                "value": corr_prob_score,
            }
        )

        measurement_rows.append(
            {
                "check": "correlation_student_success_with_math_score",
                "value": corr_success_score,
            }
        )

        valid["probability_decile"] = pd.qcut(
            valid["predicted_probability"],
            q=min(10, valid["predicted_probability"].nunique()),
            labels=False,
            duplicates="drop",
        )

        valid["probability_decile"] = valid["probability_decile"] + 1

        score_alignment = (
            valid.groupby("probability_decile", observed=True)
            .agg(
                n=("student_success", "size"),
                mean_predicted_probability=("predicted_probability", "mean"),
                observed_success_rate=("student_success", "mean"),
                mean_math_score=(score_col, "mean"),
                min_math_score=(score_col, "min"),
                max_math_score=(score_col, "max"),
            )
            .reset_index()
            .sort_values("probability_decile")
        )

        score_alignment.to_csv(
            REPORT_DIR / "09_prediction_math_score_alignment_by_decile.csv",
            index=False,
        )

        plt.figure(figsize=(8, 5))
        plt.plot(
            score_alignment["probability_decile"],
            score_alignment["mean_math_score"],
            marker="o",
        )
        plt.xlabel("Predicted-probability decile")
        plt.ylabel("Mean math score")
        plt.title("Mean Math Score by Predicted-Probability Decile")
        plt.tight_layout()
        plt.savefig(
            FIGURE_DIR / "09_mean_math_score_by_probability_decile.png",
            dpi=300,
        )
        plt.close()

measurement_df = pd.DataFrame(measurement_rows)
measurement_df.to_csv(REPORT_DIR / "09_measurement_checks.csv", index=False)


# ---------------------------------------------------------------------
# Plain-language diagnostic summary
# ---------------------------------------------------------------------

summary_lines = []

summary_lines.append("STEP H: MODEL DIAGNOSTICS, INTERPRETATION, AND FAIRNESS CHECKS")
summary_lines.append("=" * 75)
summary_lines.append("")
summary_lines.append(f"Dataset used: {dataset_path}")
summary_lines.append(f"Original dataset rows: {len(df):,}")
summary_lines.append(f"Rows used for modeling: {len(df_model):,}")
summary_lines.append(f"Target column: {target_col}")
summary_lines.append(f"Best diagnostic model: {best_model_name}")
summary_lines.append("")
summary_lines.append("Best-model test metrics:")
summary_lines.append(f"- Accuracy: {best_metrics['test_accuracy']:.4f}")
summary_lines.append(f"- Precision: {best_metrics['test_precision']:.4f}")
summary_lines.append(f"- Recall: {best_metrics['test_recall']:.4f}")
summary_lines.append(f"- F1: {best_metrics['test_f1']:.4f}")
summary_lines.append(f"- ROC AUC: {best_metrics['test_roc_auc']:.4f}")
summary_lines.append(f"- Average precision: {best_metrics['test_average_precision']:.4f}")
summary_lines.append(f"- Brier score: {best_metrics['test_brier_score']:.4f}")
summary_lines.append(f"- Threshold selected from training data: {best_metrics['threshold']:.4f}")
summary_lines.append("")
summary_lines.append("Feature policy:")
summary_lines.append("- student_success was used only as the prediction target.")
summary_lines.append("- Plausible-value scores and direct achievement-score columns were excluded from model features.")
summary_lines.append("- Gender and country were excluded as direct model features when available.")
summary_lines.append("- ESCS was retained as a substantive background predictor when available.")
summary_lines.append("")
summary_lines.append("Fairness / subgroup checks:")
if fairness_df.empty:
    summary_lines.append("- No subgroup fairness table was produced because no recognized subgroup columns were available.")
else:
    summary_lines.append(f"- Subgroup rows produced: {len(fairness_df):,}")
    summary_lines.append(f"- Flagged subgroup rows: {len(fairness_flags):,}")
    summary_lines.append(
        "- A subgroup is flagged when recall gap, predicted-positive-rate gap, "
        "or false-positive-rate gap is at least 0.10 in absolute value."
    )

summary_lines.append("")
summary_lines.append("Measurement checks:")
if measurement_df.empty:
    summary_lines.append("- Math-score alignment checks were not produced because no math score column was found.")
else:
    for _, row in measurement_df.iterrows():
        summary_lines.append(f"- {row['check']}: {row['value']:.4f}")

summary_lines.append("")
summary_lines.append("Main output files:")
summary_lines.append(f"- {REPORT_DIR / '09_model_diagnostics_metrics.csv'}")
summary_lines.append(f"- {REPORT_DIR / '09_calibration_by_decile.csv'}")
summary_lines.append(f"- {REPORT_DIR / '09_success_rate_by_probability_decile.csv'}")
summary_lines.append(f"- {REPORT_DIR / '09_permutation_feature_importance.csv'}")
summary_lines.append(f"- {REPORT_DIR / '09_subgroup_fairness_metrics.csv'}")
summary_lines.append(f"- {REPORT_DIR / '09_fairness_flags.csv'}")
summary_lines.append(f"- {REPORT_DIR / '09_measurement_checks.csv'}")
summary_lines.append(f"- {FIGURE_DIR}")
summary_lines.append(f"- {MODEL_DIR / '09_best_diagnostic_student_success_model.joblib'}")

summary_text = "\n".join(summary_lines)

with open(REPORT_DIR / "09_model_diagnostics_fairness_summary.txt", "w", encoding="utf-8") as f:
    f.write(summary_text)

print("\n" + summary_text)
print("\nStep H complete.")