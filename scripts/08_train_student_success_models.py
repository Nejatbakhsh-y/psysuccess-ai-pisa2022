"""
Step G: Train first student-success prediction models

Input:
    data/processed/pisa_student_analysis_dataset.csv

Target:
    student_success

Important leakage rule:
    If student_success was created from math_score_mean_pv, then math_score_mean_pv
    and all plausible-value test score columns must NOT be used as predictors.

Outputs:
    reports/tables/08_model_metrics.csv
    reports/tables/08_confusion_matrices.csv
    reports/tables/08_test_predictions.csv
    reports/tables/08_feature_importance.csv
    reports/figures/08_roc_curves.png
    reports/figures/08_best_confusion_matrix.png
    models/08_best_student_success_model.joblib
    models/08_model_manifest.json
"""

from pathlib import Path
import json
import re
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
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
# 1. Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
REPORTS_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

REPORTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# 2. Load clean student dataset
# ---------------------------------------------------------------------

candidate_files = [
    DATA_DIR / "pisa_student_analysis_dataset.csv",
    DATA_DIR / "pisa_student_clean.csv",
    DATA_DIR / "student_analysis_dataset.csv",
]

input_path = None
for file_path in candidate_files:
    if file_path.exists():
        input_path = file_path
        break

if input_path is None:
    csv_files = list(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            "No processed CSV file found in data/processed/. "
            "Run Step E first to create the clean student dataset."
        )
    input_path = csv_files[0]

print(f"Loading dataset from: {input_path}")

df = pd.read_csv(input_path)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")


# ---------------------------------------------------------------------
# 3. Validate target
# ---------------------------------------------------------------------

TARGET = "student_success"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' was not found. "
        f"Available columns are: {list(df.columns)}"
    )

# Convert target to binary 0/1 if needed
if df[TARGET].dtype == "object":
    target_map = {
        "yes": 1,
        "y": 1,
        "true": 1,
        "success": 1,
        "successful": 1,
        "1": 1,
        "no": 0,
        "n": 0,
        "false": 0,
        "not_success": 0,
        "not successful": 0,
        "0": 0,
    }

    df[TARGET] = (
        df[TARGET]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(target_map)
    )

if df[TARGET].isna().any():
    before = len(df)
    df = df.dropna(subset=[TARGET]).copy()
    after = len(df)
    print(f"Dropped {before - after} rows with missing target values.")

df[TARGET] = df[TARGET].astype(int)

print("\nTarget distribution:")
print(df[TARGET].value_counts(dropna=False))
print("\nTarget distribution percentage:")
print((df[TARGET].value_counts(normalize=True) * 100).round(2))


# ---------------------------------------------------------------------
# 4. Remove leakage and ID-like columns
# ---------------------------------------------------------------------

def is_leakage_column(col_name: str) -> bool:
    """
    Detect columns that should not be used as predictors.

    The main target is student_success. If it was defined from math scores,
    then math scores and plausible values are leakage.
    """
    c = col_name.lower()

    direct_leakage = {
        TARGET.lower(),
        "math_score_mean_pv",
        "student_success_label",
        "success_label",
    }

    if c in direct_leakage:
        return True

    # PISA plausible values and achievement scores
    leakage_patterns = [
        r"^pv\d+math",
        r"^pv\d+read",
        r"^pv\d+scie",
        r"math_score",
        r"reading_score",
        r"science_score",
        r"score_mean",
        r"achievement",
    ]

    for pattern in leakage_patterns:
        if re.search(pattern, c):
            return True

    return False


def is_id_like_column(col_name: str, series: pd.Series) -> bool:
    """
    Remove columns that are IDs or extremely high-cardinality identifiers.
    These often cause overfitting and do not generalize well.
    """
    c = col_name.lower()

    id_keywords = [
        "student_id",
        "school_id",
        "cntstuid",
        "cntschid",
        "stuid",
        "schid",
        "id",
    ]

    if any(keyword == c or keyword in c for keyword in id_keywords):
        return True

    # High-cardinality categorical columns are often IDs or near-IDs
    if series.dtype == "object":
        unique_count = series.nunique(dropna=True)
        if unique_count > 100:
            return True

    return False


drop_cols = []

for col in df.columns:
    if is_leakage_column(col):
        drop_cols.append(col)
    elif is_id_like_column(col, df[col]):
        drop_cols.append(col)

drop_cols = sorted(set(drop_cols))

print("\nColumns removed from predictors:")
for col in drop_cols:
    print(f"  - {col}")

X = df.drop(columns=drop_cols, errors="ignore")
y = df[TARGET]

# Remove all-missing and constant columns
all_missing_cols = [col for col in X.columns if X[col].isna().all()]
constant_cols = [col for col in X.columns if X[col].nunique(dropna=True) <= 1]

X = X.drop(columns=all_missing_cols + constant_cols, errors="ignore")

print(f"\nPredictor matrix shape after cleaning: {X.shape}")

if X.shape[1] == 0:
    raise ValueError("No usable predictor columns remain after leakage and ID removal.")


# ---------------------------------------------------------------------
# 5. Identify numeric and categorical predictors
# ---------------------------------------------------------------------

numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()

print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# ---------------------------------------------------------------------
# 6. Train/test split
# ---------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(f"\nTraining rows: {X_train.shape[0]}")
print(f"Testing rows: {X_test.shape[0]}")


# ---------------------------------------------------------------------
# 7. Preprocessing pipelines
# ---------------------------------------------------------------------

try:
    categorical_encoder = OneHotEncoder(
        handle_unknown="ignore",
        min_frequency=10,
    )
except TypeError:
    categorical_encoder = OneHotEncoder(
        handle_unknown="ignore",
    )

numeric_preprocess_scaled = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

numeric_preprocess_tree = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

categorical_preprocess = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", categorical_encoder),
    ]
)

preprocessor_scaled = ColumnTransformer(
    transformers=[
        ("num", numeric_preprocess_scaled, numeric_features),
        ("cat", categorical_preprocess, categorical_features),
    ],
    remainder="drop",
)

preprocessor_tree = ColumnTransformer(
    transformers=[
        ("num", numeric_preprocess_tree, numeric_features),
        ("cat", categorical_preprocess, categorical_features),
    ],
    remainder="drop",
)


# ---------------------------------------------------------------------
# 8. Define models
# ---------------------------------------------------------------------

models = {
    "logistic_regression": Pipeline(
        steps=[
            ("preprocessor", preprocessor_scaled),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    ),
    "random_forest": Pipeline(
        steps=[
            ("preprocessor", preprocessor_tree),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=5,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    ),
    "gradient_boosting": Pipeline(
        steps=[
            ("preprocessor", preprocessor_tree),
            (
                "model",
                GradientBoostingClassifier(
                    random_state=42,
                ),
            ),
        ]
    ),
}


# ---------------------------------------------------------------------
# 9. Train and evaluate models
# ---------------------------------------------------------------------

metrics_rows = []
confusion_rows = []
prediction_frames = []
roc_data = {}

best_model_name = None
best_model = None
best_auc = -np.inf
best_f1 = -np.inf

for model_name, pipeline in models.items():
    print(f"\nTraining model: {model_name}")

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    if hasattr(pipeline.named_steps["model"], "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
    else:
        y_prob = None

    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    if y_prob is not None:
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = np.nan

    metrics_rows.append(
        {
            "model": model_name,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": auc,
            "train_rows": X_train.shape[0],
            "test_rows": X_test.shape[0],
            "num_features_before_encoding": X.shape[1],
        }
    )

    cm = confusion_matrix(y_test, y_pred)
    confusion_rows.append(
        {
            "model": model_name,
            "true_negative": int(cm[0, 0]),
            "false_positive": int(cm[0, 1]),
            "false_negative": int(cm[1, 0]),
            "true_positive": int(cm[1, 1]),
        }
    )

    pred_frame = X_test.copy()
    pred_frame["actual_student_success"] = y_test.values
    pred_frame[f"predicted_{model_name}"] = y_pred

    if y_prob is not None:
        pred_frame[f"probability_success_{model_name}"] = y_prob

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[model_name] = {
            "fpr": fpr,
            "tpr": tpr,
            "auc": auc,
        }

    prediction_frames.append(pred_frame)

    print(classification_report(y_test, y_pred, zero_division=0))

    # Select best model by ROC-AUC when available; otherwise by F1
    if not np.isnan(auc):
        if auc > best_auc:
            best_auc = auc
            best_f1 = f1
            best_model_name = model_name
            best_model = pipeline
    else:
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = model_name
            best_model = pipeline


metrics_df = pd.DataFrame(metrics_rows).sort_values(
    by=["roc_auc", "f1"],
    ascending=False,
)

confusion_df = pd.DataFrame(confusion_rows)

metrics_path = REPORTS_TABLES_DIR / "08_model_metrics.csv"
confusion_path = REPORTS_TABLES_DIR / "08_confusion_matrices.csv"

metrics_df.to_csv(metrics_path, index=False)
confusion_df.to_csv(confusion_path, index=False)

print("\nModel metrics:")
print(metrics_df)

print(f"\nSaved model metrics to: {metrics_path}")
print(f"Saved confusion matrices to: {confusion_path}")


# ---------------------------------------------------------------------
# 10. Save test predictions
# ---------------------------------------------------------------------

# Start with the test set and actual outcome
test_predictions = X_test.copy()
test_predictions["actual_student_success"] = y_test.values

for model_name, pipeline in models.items():
    y_pred = pipeline.predict(X_test)
    test_predictions[f"predicted_{model_name}"] = y_pred

    if hasattr(pipeline.named_steps["model"], "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        test_predictions[f"probability_success_{model_name}"] = y_prob

predictions_path = REPORTS_TABLES_DIR / "08_test_predictions.csv"
test_predictions.to_csv(predictions_path, index=False)

print(f"Saved test predictions to: {predictions_path}")


# ---------------------------------------------------------------------
# 11. Plot ROC curves
# ---------------------------------------------------------------------

if roc_data:
    plt.figure(figsize=(8, 6))

    for model_name, data in roc_data.items():
        plt.plot(
            data["fpr"],
            data["tpr"],
            label=f"{model_name} AUC={data['auc']:.3f}",
        )

    plt.plot([0, 1], [0, 1], linestyle="--", label="random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Student-Success Prediction Models")
    plt.legend()
    plt.tight_layout()

    roc_path = REPORTS_FIGURES_DIR / "08_roc_curves.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()

    print(f"Saved ROC curve figure to: {roc_path}")


# ---------------------------------------------------------------------
# 12. Plot best-model confusion matrix
# ---------------------------------------------------------------------

best_y_pred = best_model.predict(X_test)
best_cm = confusion_matrix(y_test, best_y_pred)

plt.figure(figsize=(6, 5))
plt.imshow(best_cm)
plt.title(f"Best Model Confusion Matrix: {best_model_name}")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.xticks([0, 1], ["Not Success", "Success"])
plt.yticks([0, 1], ["Not Success", "Success"])

for i in range(best_cm.shape[0]):
    for j in range(best_cm.shape[1]):
        plt.text(j, i, str(best_cm[i, j]), ha="center", va="center")

plt.tight_layout()

best_cm_path = REPORTS_FIGURES_DIR / "08_best_confusion_matrix.png"
plt.savefig(best_cm_path, dpi=300)
plt.close()

print(f"Saved best-model confusion matrix to: {best_cm_path}")


# ---------------------------------------------------------------------
# 13. Feature importance / model interpretation
# ---------------------------------------------------------------------

def get_feature_names_from_pipeline(pipeline):
    preprocessor = pipeline.named_steps["preprocessor"]

    try:
        return preprocessor.get_feature_names_out()
    except Exception:
        return np.array([f"feature_{i}" for i in range(preprocessor.transform(X_train).shape[1])])


feature_importance_rows = []

for model_name, pipeline in models.items():
    model = pipeline.named_steps["model"]
    feature_names = get_feature_names_from_pipeline(pipeline)

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        importance_type = "feature_importance"

    elif hasattr(model, "coef_"):
        values = model.coef_[0]
        importance_type = "coefficient"

    else:
        continue

    temp = pd.DataFrame(
        {
            "model": model_name,
            "feature": feature_names,
            "importance_value": values,
            "absolute_importance_value": np.abs(values),
            "importance_type": importance_type,
        }
    )

    temp = temp.sort_values("absolute_importance_value", ascending=False).head(50)

    feature_importance_rows.append(temp)

if feature_importance_rows:
    feature_importance_df = pd.concat(feature_importance_rows, ignore_index=True)
else:
    feature_importance_df = pd.DataFrame(
        columns=[
            "model",
            "feature",
            "importance_value",
            "absolute_importance_value",
            "importance_type",
        ]
    )

feature_importance_path = REPORTS_TABLES_DIR / "08_feature_importance.csv"
feature_importance_df.to_csv(feature_importance_path, index=False)

print(f"Saved feature importance table to: {feature_importance_path}")


# ---------------------------------------------------------------------
# 14. Save best model and manifest
# ---------------------------------------------------------------------

best_model_path = MODELS_DIR / "08_best_student_success_model.joblib"
joblib.dump(best_model, best_model_path)

manifest = {
    "step": "Step G",
    "script": "scripts/08_train_student_success_models.py",
    "input_file": str(input_path.relative_to(PROJECT_ROOT)),
    "target": TARGET,
    "best_model": best_model_name,
    "best_model_path": str(best_model_path.relative_to(PROJECT_ROOT)),
    "metrics_file": str(metrics_path.relative_to(PROJECT_ROOT)),
    "confusion_matrix_file": str(confusion_path.relative_to(PROJECT_ROOT)),
    "predictions_file": str(predictions_path.relative_to(PROJECT_ROOT)),
    "feature_importance_file": str(feature_importance_path.relative_to(PROJECT_ROOT)),
    "roc_curve_file": str((REPORTS_FIGURES_DIR / "08_roc_curves.png").relative_to(PROJECT_ROOT)),
    "best_confusion_matrix_file": str(best_cm_path.relative_to(PROJECT_ROOT)),
    "leakage_columns_removed": drop_cols,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "model_selection_rule": "Highest ROC-AUC; F1 used only if ROC-AUC is unavailable.",
}

manifest_path = MODELS_DIR / "08_model_manifest.json"

with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=4)

print(f"Saved best model to: {best_model_path}")
print(f"Saved model manifest to: {manifest_path}")

print("\nStep G complete.")