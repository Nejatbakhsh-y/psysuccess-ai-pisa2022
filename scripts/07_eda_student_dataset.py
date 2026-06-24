"""
Step F: Exploratory Data Analysis for the clean PISA student dataset

This script creates summary tables and charts for:
- student_success
- math_score_mean_pv
- country distribution
- ESCS and student success
- gender and student success
- school-level variation
- missing values
- correlation structure

Outputs are saved to:

outputs/eda/
"""

from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.switch_backend("Agg")
warnings.filterwarnings("ignore")


# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

DEFAULT_INPUT_CANDIDATES = [
    "data/processed/pisa_student_analysis.parquet",
    "data/processed/pisa_student_clean.parquet",
    "data/processed/pisa_student_analysis_dataset.parquet",
    "data/processed/student_analysis_dataset.parquet",
    "data/processed/pisa_student_analysis.csv",
    "data/processed/pisa_student_clean.csv",
    "data/processed/pisa_student_analysis_dataset.csv",
    "data/processed/student_analysis_dataset.csv",
]


GLOB_PATTERNS = [
    "data/processed/*student*analysis*.parquet",
    "data/processed/*student*clean*.parquet",
    "data/processed/*pisa*student*.parquet",
    "data/processed/*student*analysis*.csv",
    "data/processed/*student*clean*.csv",
    "data/processed/*pisa*student*.csv",
]


def resolve_input_path(input_arg: str | None) -> Path:
    if input_arg:
        path = Path(input_arg)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        return path

    for candidate in DEFAULT_INPUT_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return path

    matched_files = []
    for pattern in GLOB_PATTERNS:
        matched_files.extend(Path(".").glob(pattern))

    matched_files = [p for p in matched_files if p.is_file()]

    if matched_files:
        matched_files = sorted(matched_files, key=lambda p: p.stat().st_mtime, reverse=True)
        return matched_files[0]

    raise FileNotFoundError(
        "Could not find the clean PISA student dataset.\n"
        "Check your data/processed folder, or run the script with:\n\n"
        "python scripts/07_eda_student_dataset.py --input data/processed/YOUR_FILE_NAME.parquet"
    )


def read_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path)

    if suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported file format: {suffix}. Use .parquet or .csv.")


# ------------------------------------------------------------
# 2. Column detection helpers
# ------------------------------------------------------------

def normalize_name(name: str) -> str:
    return (
        str(name)
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
        .replace(".", "")
    )


def pick_column(
    df: pd.DataFrame,
    candidates: list[str],
    required: bool = True,
    label: str = "column",
) -> str | None:
    lower_map = {str(c).lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    normalized_map = {normalize_name(c): c for c in df.columns}

    for candidate in candidates:
        key = normalize_name(candidate)
        if key in normalized_map:
            return normalized_map[key]

    if required:
        raise ValueError(
            f"Could not find required {label}. Tried: {candidates}\n"
            f"Available columns include:\n{list(df.columns)[:80]}"
        )

    return None


def normalize_binary_success(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().mean() > 0.70:
        return numeric

    mapping = {
        "1": 1,
        "true": 1,
        "yes": 1,
        "success": 1,
        "successful": 1,
        "pass": 1,
        "passed": 1,
        "0": 0,
        "false": 0,
        "no": 0,
        "not success": 0,
        "not_success": 0,
        "fail": 0,
        "failed": 0,
    }

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )


def normalize_gender(series: pd.Series) -> pd.Series:
    def one_value(x):
        if pd.isna(x):
            return np.nan

        text = str(x).strip().lower()

        if text in {"1", "1.0"}:
            return "Female"

        if text in {"2", "2.0"}:
            return "Male"

        if "female" in text:
            return "Female"

        if "male" in text:
            return "Male"

        if text in {"f", "woman", "girl"}:
            return "Female"

        if text in {"m", "man", "boy"}:
            return "Male"

        return str(x)

    return series.apply(one_value)


def save_table(df: pd.DataFrame, output_path: Path) -> None:
    df.to_csv(output_path, index=False)


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------
# 3. EDA functions
# ------------------------------------------------------------

def create_overall_summary(
    df: pd.DataFrame,
    math_col: str,
    success_col_num: str,
    escs_col: str | None,
    output_dir: Path,
) -> None:
    rows = []

    rows.append({"metric": "n_rows", "value": len(df)})
    rows.append({"metric": "n_columns", "value": df.shape[1]})
    rows.append({"metric": "math_score_mean", "value": df[math_col].mean()})
    rows.append({"metric": "math_score_std", "value": df[math_col].std()})
    rows.append({"metric": "math_score_min", "value": df[math_col].min()})
    rows.append({"metric": "math_score_max", "value": df[math_col].max()})
    rows.append({"metric": "student_success_rate", "value": df[success_col_num].mean()})

    if escs_col:
        rows.append({"metric": "escs_mean", "value": df[escs_col].mean()})
        rows.append({"metric": "escs_std", "value": df[escs_col].std()})
        rows.append({"metric": "escs_min", "value": df[escs_col].min()})
        rows.append({"metric": "escs_max", "value": df[escs_col].max()})

    summary = pd.DataFrame(rows)
    save_table(summary, output_dir / "summary_overall.csv")


def create_math_score_chart(df: pd.DataFrame, math_col: str, output_dir: Path) -> None:
    plt.figure(figsize=(9, 5))
    df[math_col].dropna().hist(bins=40)
    plt.title("Distribution of Math Score Mean Plausible Value")
    plt.xlabel("math_score_mean_pv")
    plt.ylabel("Number of students")
    save_plot(output_dir / "fig_math_score_distribution.png")


def create_country_distribution(
    df: pd.DataFrame,
    country_col: str | None,
    success_col_num: str,
    math_col: str,
    output_dir: Path,
) -> None:
    if country_col is None:
        return

    country_distribution = (
        df[country_col]
        .value_counts(dropna=False)
        .reset_index()
    )

    country_distribution.columns = ["country", "n_students"]
    country_distribution["percent"] = (
        100 * country_distribution["n_students"] / country_distribution["n_students"].sum()
    )

    save_table(country_distribution, output_dir / "country_distribution.csv")

    top_countries = country_distribution.head(20).sort_values("n_students")

    plt.figure(figsize=(9, 7))
    plt.barh(top_countries["country"].astype(str), top_countries["n_students"])
    plt.title("Top 20 Countries by Student Count")
    plt.xlabel("Number of students")
    plt.ylabel("Country")
    save_plot(output_dir / "fig_country_distribution_top20.png")

    country_success = (
        df.groupby(country_col)
        .agg(
            n_students=(success_col_num, "size"),
            success_rate=(success_col_num, "mean"),
            math_score_mean=(math_col, "mean"),
            math_score_std=(math_col, "std"),
        )
        .reset_index()
        .rename(columns={country_col: "country"})
        .sort_values("n_students", ascending=False)
    )

    save_table(country_success, output_dir / "success_by_country.csv")


def create_gender_success(
    df: pd.DataFrame,
    gender_col: str | None,
    success_col_num: str,
    math_col: str,
    output_dir: Path,
) -> None:
    if gender_col is None:
        return

    df = df.copy()
    df["gender_label"] = normalize_gender(df[gender_col])

    gender_summary = (
        df.groupby("gender_label", dropna=False)
        .agg(
            n_students=(success_col_num, "size"),
            success_rate=(success_col_num, "mean"),
            math_score_mean=(math_col, "mean"),
            math_score_std=(math_col, "std"),
        )
        .reset_index()
        .sort_values("n_students", ascending=False)
    )

    save_table(gender_summary, output_dir / "success_by_gender.csv")

    plot_df = gender_summary.dropna(subset=["gender_label"]).copy()

    plt.figure(figsize=(7, 5))
    plt.bar(plot_df["gender_label"].astype(str), plot_df["success_rate"])
    plt.title("Student Success Rate by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Success rate")
    plt.ylim(0, 1)
    save_plot(output_dir / "fig_success_by_gender.png")


def create_escs_success(
    df: pd.DataFrame,
    escs_col: str | None,
    success_col_num: str,
    math_col: str,
    output_dir: Path,
) -> None:
    if escs_col is None:
        return

    df = df.copy()

    valid_escs = df[escs_col].dropna()

    if valid_escs.nunique() < 5:
        return

    df["escs_quintile"] = pd.qcut(
        df[escs_col],
        q=5,
        labels=["Q1 lowest ESCS", "Q2", "Q3", "Q4", "Q5 highest ESCS"],
        duplicates="drop",
    )

    escs_summary = (
        df.groupby("escs_quintile", dropna=False)
        .agg(
            n_students=(success_col_num, "size"),
            success_rate=(success_col_num, "mean"),
            math_score_mean=(math_col, "mean"),
            escs_mean=(escs_col, "mean"),
        )
        .reset_index()
    )

    save_table(escs_summary, output_dir / "success_by_escs_quintile.csv")

    plot_df = escs_summary.dropna(subset=["escs_quintile"]).copy()

    plt.figure(figsize=(9, 5))
    plt.bar(plot_df["escs_quintile"].astype(str), plot_df["success_rate"])
    plt.title("Student Success Rate by ESCS Quintile")
    plt.xlabel("ESCS quintile")
    plt.ylabel("Success rate")
    plt.ylim(0, 1)
    plt.xticks(rotation=25, ha="right")
    save_plot(output_dir / "fig_success_by_escs_quintile.png")

    plt.figure(figsize=(9, 5))
    plt.bar(plot_df["escs_quintile"].astype(str), plot_df["math_score_mean"])
    plt.title("Mean Math Score by ESCS Quintile")
    plt.xlabel("ESCS quintile")
    plt.ylabel("Mean math score")
    plt.xticks(rotation=25, ha="right")
    save_plot(output_dir / "fig_math_score_by_escs_quintile.png")


def create_school_variation(
    df: pd.DataFrame,
    school_col: str | None,
    success_col_num: str,
    math_col: str,
    output_dir: Path,
) -> None:
    if school_col is None:
        return

    school_summary = (
        df.groupby(school_col)
        .agg(
            n_students=(success_col_num, "size"),
            success_rate=(success_col_num, "mean"),
            math_score_mean=(math_col, "mean"),
            math_score_std=(math_col, "std"),
        )
        .reset_index()
        .rename(columns={school_col: "school_id"})
        .sort_values("n_students", ascending=False)
    )

    save_table(school_summary, output_dir / "school_level_variation.csv")

    eligible_schools = school_summary[school_summary["n_students"] >= 20].copy()

    if len(eligible_schools) == 0:
        eligible_schools = school_summary.copy()

    plt.figure(figsize=(9, 5))
    eligible_schools["success_rate"].dropna().hist(bins=30)
    plt.title("Distribution of School-Level Success Rates")
    plt.xlabel("School success rate")
    plt.ylabel("Number of schools")
    save_plot(output_dir / "fig_school_success_rate_distribution.png")

    top_schools = (
        eligible_schools
        .sort_values("n_students", ascending=False)
        .head(20)
        .sort_values("success_rate")
    )

    plt.figure(figsize=(9, 7))
    plt.barh(top_schools["school_id"].astype(str), top_schools["success_rate"])
    plt.title("Success Rate for 20 Largest Schools")
    plt.xlabel("Success rate")
    plt.ylabel("School ID")
    plt.xlim(0, 1)
    save_plot(output_dir / "fig_school_variation_largest20.png")


def create_missing_values(df: pd.DataFrame, output_dir: Path) -> None:
    missing = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_percent": 100 * df.isna().sum().values / len(df),
        }
    ).sort_values("missing_percent", ascending=False)

    save_table(missing, output_dir / "missing_values.csv")

    top_missing = missing.head(25).sort_values("missing_percent")

    plt.figure(figsize=(9, 8))
    plt.barh(top_missing["column"].astype(str), top_missing["missing_percent"])
    plt.title("Top 25 Columns by Missing Percentage")
    plt.xlabel("Missing percentage")
    plt.ylabel("Column")
    save_plot(output_dir / "fig_missing_values_top25.png")


def create_correlation_structure(
    df: pd.DataFrame,
    math_col: str,
    success_col_num: str,
    escs_col: str | None,
    output_dir: Path,
) -> None:
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    numeric_df = numeric_df.loc[:, numeric_df.nunique(dropna=True) > 1]

    if numeric_df.shape[1] < 2:
        return

    priority_cols = [math_col, success_col_num]

    if escs_col and escs_col in numeric_df.columns:
        priority_cols.append(escs_col)

    priority_cols = [c for c in priority_cols if c in numeric_df.columns]

    corr_full = numeric_df.corr(numeric_only=True)

    selected_cols = []

    for col in priority_cols:
        if col not in selected_cols:
            selected_cols.append(col)

    if math_col in corr_full.columns:
        top_related = (
            corr_full[math_col]
            .abs()
            .dropna()
            .sort_values(ascending=False)
            .index
            .tolist()
        )
    else:
        top_related = corr_full.columns.tolist()

    for col in top_related:
        if col not in selected_cols:
            selected_cols.append(col)
        if len(selected_cols) >= 20:
            break

    corr_selected = numeric_df[selected_cols].corr(numeric_only=True)

    save_table(
        corr_selected.reset_index().rename(columns={"index": "variable"}),
        output_dir / "correlation_matrix_selected.csv",
    )

    plt.figure(figsize=(11, 9))
    plt.imshow(corr_selected, aspect="auto")
    plt.colorbar(label="Correlation")
    plt.title("Correlation Structure: Selected Numeric Variables")
    plt.xticks(range(len(corr_selected.columns)), corr_selected.columns, rotation=90)
    plt.yticks(range(len(corr_selected.index)), corr_selected.index)
    save_plot(output_dir / "fig_correlation_heatmap_selected.png")


# ------------------------------------------------------------
# 4. Main script
# ------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Optional path to the clean student dataset."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/eda",
        help="Output folder for EDA tables and charts."
    )

    args = parser.parse_args()

    input_path = resolve_input_path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Step F: Exploratory Data Analysis")
    print("=" * 70)
    print(f"Reading dataset from: {input_path}")

    df = read_dataset(input_path)

    print(f"Dataset shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns")

    math_col = pick_column(
        df,
        candidates=[
            "math_score_mean_pv",
            "math_score_mean",
            "math_score",
            "pv_math_mean",
            "PV_MATH_MEAN",
        ],
        required=True,
        label="math score column",
    )

    success_col = pick_column(
        df,
        candidates=[
            "student_success",
            "success",
            "student_success_flag",
            "math_success",
            "is_success",
        ],
        required=True,
        label="student success column",
    )

    country_col = pick_column(
        df,
        candidates=[
            "country",
            "CNT",
            "cnt",
            "CNTRYID",
            "country_code",
            "country_name",
        ],
        required=False,
        label="country column",
    )

    escs_col = pick_column(
        df,
        candidates=[
            "ESCS",
            "escs",
            "economic_social_cultural_status",
            "socioeconomic_status",
        ],
        required=False,
        label="ESCS column",
    )

    gender_col = pick_column(
        df,
        candidates=[
            "gender",
            "GENDER",
            "ST004D01T",
            "st004d01t",
            "sex",
        ],
        required=False,
        label="gender column",
    )

    school_col = pick_column(
        df,
        candidates=[
            "school_id",
            "schoolid",
            "CNTSCHID",
            "cntschid",
            "SCHLID",
            "schlid",
        ],
        required=False,
        label="school ID column",
    )

    df = df.copy()

    df[math_col] = pd.to_numeric(df[math_col], errors="coerce")

    success_col_num = "student_success_numeric"
    df[success_col_num] = normalize_binary_success(df[success_col])

    print("\nDetected columns:")
    print(f"Math score column:      {math_col}")
    print(f"Student success column: {success_col}")
    print(f"Country column:         {country_col}")
    print(f"ESCS column:            {escs_col}")
    print(f"Gender column:          {gender_col}")
    print(f"School ID column:       {school_col}")

    create_overall_summary(df, math_col, success_col_num, escs_col, output_dir)
    create_math_score_chart(df, math_col, output_dir)
    create_country_distribution(df, country_col, success_col_num, math_col, output_dir)
    create_gender_success(df, gender_col, success_col_num, math_col, output_dir)
    create_escs_success(df, escs_col, success_col_num, math_col, output_dir)
    create_school_variation(df, school_col, success_col_num, math_col, output_dir)
    create_missing_values(df, output_dir)
    create_correlation_structure(df, math_col, success_col_num, escs_col, output_dir)

    print("\nEDA complete.")
    print(f"Output folder: {output_dir}")
    print("\nCreated files:")
    for path in sorted(output_dir.glob("*")):
        print(f"- {path}")


if __name__ == "__main__":
    main()