"""
Step E: Create the first clean PISA 2022 student analysis dataset.

Input:
    data/raw/extracted/**/CY08MSP_STU_QQQ.SAV

Outputs:
    data/processed/pisa2022_student_clean.csv
    data/processed/pisa2022_student_clean.parquet
    reports/pisa2022_student_clean_metadata.json

Target:
    student_success = 1 if mean(PV1MATH ... PV10MATH) >= 420.07
"""

from pathlib import Path
import json

import pandas as pd
import pyreadstat


ROOT = Path(__file__).resolve().parents[1]
RAW_EXTRACTED_DIR = ROOT / "data" / "raw" / "extracted"
OUT_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def find_student_sav() -> Path:
    matches = list(RAW_EXTRACTED_DIR.rglob("CY08MSP_STU_QQQ.SAV"))
    if not matches:
        raise FileNotFoundError(
            "CY08MSP_STU_QQQ.SAV was not found under data/raw/extracted/. "
            "Run Step D again or check the extracted PISA folder."
        )
    return matches[0]


def keep_existing(requested_columns, available_columns):
    return [col for col in requested_columns if col in available_columns]


def add_mean_score(df, pv_columns, new_column):
    existing = [col for col in pv_columns if col in df.columns]
    if existing:
        df[new_column] = df[existing].mean(axis=1, skipna=True)
    else:
        df[new_column] = pd.NA


def binary_cut_score(series, cut_score):
    result = series.ge(cut_score)
    result = result.where(series.notna(), pd.NA)
    return result.astype("Int64")


student_sav = find_student_sav()

print(f"Reading metadata from: {student_sav}")

_, meta = pyreadstat.read_sav(str(student_sav), metadataonly=True)
available_columns = set(meta.column_names)

id_columns = [
    "CNT",
    "CNTRYID",
    "CNTSCHID",
    "CNTSTUID",
]

weight_columns = [
    "W_FSTUWT",
]

plausible_math = [f"PV{i}MATH" for i in range(1, 11)]
plausible_reading = [f"PV{i}READ" for i in range(1, 11)]
plausible_science = [f"PV{i}SCIE" for i in range(1, 11)]

feature_columns = [
    "ST004D01T",   # gender
    "AGE",
    "GRADE",
    "ESCS",
    "HOMEPOS",
    "HISEI",
    "PAREDINT",
    "IMMIG",
    "REPEAT",
    "BELONG",
    "BULLIED",
    "FEELSAFE",
    "ANXMAT",
    "MATHEFF",
    "MATHEASE",
    "MATHMOT",
    "ICTRES",
    "HEDRES",
    "CULTPOSS",
    "WEALTH",
]

requested_columns = (
    id_columns
    + weight_columns
    + feature_columns
    + plausible_math
    + plausible_reading
    + plausible_science
)

selected_columns = keep_existing(requested_columns, available_columns)
missing_columns = [col for col in requested_columns if col not in available_columns]

print(f"Total columns in SAV: {len(available_columns):,}")
print(f"Selected columns: {len(selected_columns):,}")
print(f"Missing requested columns: {len(missing_columns):,}")

df, _ = pyreadstat.read_sav(
    str(student_sav),
    usecols=selected_columns,
    apply_value_formats=False,
    user_missing=False,
)

print(f"Loaded selected data: {df.shape[0]:,} rows x {df.shape[1]:,} columns")

df = df.replace(r"^\s*$", pd.NA, regex=True)

for col in df.columns:
    if col != "CNT":
        df[col] = pd.to_numeric(df[col], errors="coerce")

add_mean_score(df, plausible_math, "math_score_mean_pv")
add_mean_score(df, plausible_reading, "reading_score_mean_pv")
add_mean_score(df, plausible_science, "science_score_mean_pv")

MATH_LEVEL2_CUT = 420.07
READING_LEVEL2_CUT = 407.47
SCIENCE_LEVEL2_CUT = 409.54

df["math_success_level2"] = binary_cut_score(df["math_score_mean_pv"], MATH_LEVEL2_CUT)
df["reading_success_level2"] = binary_cut_score(df["reading_score_mean_pv"], READING_LEVEL2_CUT)
df["science_success_level2"] = binary_cut_score(df["science_score_mean_pv"], SCIENCE_LEVEL2_CUT)

success_cols = [
    "math_success_level2",
    "reading_success_level2",
    "science_success_level2",
]

has_all_success_values = df[success_cols].notna().all(axis=1)
all_three_success = df[success_cols].eq(1).all(axis=1)

df["success_all_three_level2"] = all_three_success.where(
    has_all_success_values,
    pd.NA,
).astype("Int64")

df["student_success"] = df["math_success_level2"]

rename_map = {
    "CNT": "country_code",
    "CNTRYID": "country_id",
    "CNTSCHID": "school_id",
    "CNTSTUID": "student_id",
    "W_FSTUWT": "student_weight",
    "ST004D01T": "gender_raw",
    "AGE": "age",
    "GRADE": "grade",
    "ESCS": "escs",
    "HOMEPOS": "home_possessions",
    "HISEI": "highest_parent_isei",
    "PAREDINT": "parent_education_years",
    "IMMIG": "immigrant_status",
    "REPEAT": "grade_repetition",
    "BELONG": "school_belonging",
    "BULLIED": "bullied_index",
    "FEELSAFE": "feeling_safe",
    "ANXMAT": "math_anxiety",
    "MATHEFF": "math_self_efficacy",
    "MATHEASE": "math_ease",
    "MATHMOT": "math_motivation",
    "ICTRES": "ict_resources",
    "HEDRES": "home_educational_resources",
    "CULTPOSS": "cultural_possessions",
    "WEALTH": "wealth_index",
}

df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

rows_before_drop = len(df)

df = df.dropna(subset=["math_score_mean_pv", "student_success"]).copy()

rows_after_drop = len(df)

main_columns = [
    "country_code",
    "country_id",
    "school_id",
    "student_id",
    "student_weight",
    "math_score_mean_pv",
    "reading_score_mean_pv",
    "science_score_mean_pv",
    "math_success_level2",
    "reading_success_level2",
    "science_success_level2",
    "success_all_three_level2",
    "student_success",
]

ordered_columns = [col for col in main_columns if col in df.columns]
ordered_columns += [col for col in df.columns if col not in ordered_columns]

df = df[ordered_columns]

csv_output = OUT_DIR / "pisa2022_student_clean.csv"
parquet_output = OUT_DIR / "pisa2022_student_clean.parquet"
metadata_output = REPORT_DIR / "pisa2022_student_clean_metadata.json"

df.to_csv(csv_output, index=False)
df.to_parquet(parquet_output, index=False)

metadata_report = {
    "input_file": str(student_sav.relative_to(ROOT)),
    "rows_before_target_drop": int(rows_before_drop),
    "rows_after_target_drop": int(rows_after_drop),
    "columns_output": int(df.shape[1]),
    "csv_output": str(csv_output.relative_to(ROOT)),
    "parquet_output": str(parquet_output.relative_to(ROOT)),
    "metadata_output": str(metadata_output.relative_to(ROOT)),
    "target_definition": "student_success = 1 if mean(PV1MATH...PV10MATH) >= 420.07",
    "math_level2_cut_score": MATH_LEVEL2_CUT,
    "reading_level2_cut_score": READING_LEVEL2_CUT,
    "science_level2_cut_score": SCIENCE_LEVEL2_CUT,
    "selected_columns": selected_columns,
    "missing_requested_columns": missing_columns,
}

metadata_output.write_text(
    json.dumps(metadata_report, indent=2),
    encoding="utf-8",
)

print("\nStep E complete.")
print(f"CSV saved to: {csv_output}")
print(f"Parquet saved to: {parquet_output}")
print(f"Metadata saved to: {metadata_output}")
print(f"Final dataset shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns")

print("\nTarget distribution:")
print(df["student_success"].value_counts(dropna=False).sort_index())

print("\nFirst 10 columns:")
print(df.columns[:10].tolist())
