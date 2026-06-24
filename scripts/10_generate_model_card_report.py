"""
Step I: Generate final model card, project report, and README-ready summary.

This script converts previous project outputs into professional Markdown artifacts:

1. docs/model_card.md
2. reports/final_project_report.md
3. README_results_summary.md
4. reports/final/artifact_index.csv

It is designed to be robust even if some earlier outputs have different filenames.
"""

from pathlib import Path
from datetime import datetime
import json
import pandas as pd


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
DOCS_DIR = ROOT / "docs"
FINAL_DIR = REPORTS_DIR / "final"

DOCS_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def relpath(path: Path) -> str:
    """Return a clean relative path for Markdown links."""
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def find_files(patterns, base_dir=ROOT):
    """Find files matching a list of glob patterns."""
    files = []
    for pattern in patterns:
        files.extend(base_dir.glob(pattern))
    return sorted(set([p for p in files if p.is_file()]))


def safe_read_csv(path: Path):
    """Read CSV safely."""
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def safe_read_json(path: Path):
    """Read JSON safely."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def dataframe_to_markdown(df: pd.DataFrame, max_rows=12) -> str:
    """Convert a DataFrame to a Markdown table safely."""
    if df is None or df.empty:
        return "_No table content available._"

    display_df = df.head(max_rows).copy()

    for col in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[col]):
            display_df[col] = display_df[col].round(4)

    return display_df.to_markdown(index=False)


def file_list_markdown(files, empty_message="_No matching files found._"):
    """Create Markdown bullet list of files."""
    if not files:
        return empty_message

    lines = []
    for file in files:
        lines.append(f"- [`{relpath(file)}`]({relpath(file)})")
    return "\n".join(lines)


def image_gallery_markdown(files, max_images=20):
    """Create Markdown image links."""
    if not files:
        return "_No figures found._"

    lines = []
    for file in files[:max_images]:
        name = file.stem.replace("_", " ").replace("-", " ").title()
        lines.append(f"### {name}\n\n![{name}]({relpath(file)})\n")
    return "\n".join(lines)


def summarize_dataset():
    """Summarize processed dataset if available."""
    processed_files = find_files([
        "data/processed/*.csv",
        "data/clean/*.csv",
        "data/interim/*.csv"
    ])

    if not processed_files:
        return {
            "dataset_path": None,
            "n_rows": None,
            "n_cols": None,
            "columns": [],
            "target_found": None,
            "preview_table": "_No processed dataset found._"
        }

    # Use the largest CSV as the main analysis dataset.
    main_file = max(processed_files, key=lambda p: p.stat().st_size)
    df = safe_read_csv(main_file)

    if df is None:
        return {
            "dataset_path": main_file,
            "n_rows": None,
            "n_cols": None,
            "columns": [],
            "target_found": None,
            "preview_table": "_Dataset was found but could not be read._"
        }

    possible_targets = [
        "student_success",
        "success",
        "target",
        "label",
        "y"
    ]

    target_found = None
    for col in possible_targets:
        if col in df.columns:
            target_found = col
            break

    preview_cols = list(df.columns[:12])
    preview = df[preview_cols].head(8)

    return {
        "dataset_path": main_file,
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "columns": list(df.columns),
        "target_found": target_found,
        "preview_table": dataframe_to_markdown(preview, max_rows=8)
    }


def collect_artifacts():
    """Collect relevant project outputs."""
    metric_files = find_files([
        "reports/**/*metric*.csv",
        "reports/**/*metrics*.csv",
        "reports/**/*metric*.json",
        "reports/**/*metrics*.json",
        "outputs/**/*metric*.csv",
        "outputs/**/*metrics*.csv",
        "outputs/**/*metric*.json",
        "outputs/**/*metrics*.json"
    ])

    model_files = find_files([
        "reports/**/*model*.csv",
        "reports/**/*model*.json",
        "outputs/**/*model*.csv",
        "outputs/**/*model*.json",
        "models/**/*"
    ])

    fairness_files = find_files([
        "reports/**/*fair*.csv",
        "reports/**/*fair*.json",
        "reports/**/*bias*.csv",
        "reports/**/*bias*.json",
        "reports/**/*diagnostic*.csv",
        "reports/**/*diagnostic*.json",
        "outputs/**/*fair*.csv",
        "outputs/**/*bias*.csv",
        "outputs/**/*diagnostic*.csv"
    ])

    interpretation_files = find_files([
        "reports/**/*feature*.csv",
        "reports/**/*importance*.csv",
        "reports/**/*shap*.csv",
        "reports/**/*interpret*.csv",
        "outputs/**/*feature*.csv",
        "outputs/**/*importance*.csv",
        "outputs/**/*shap*.csv",
        "outputs/**/*interpret*.csv"
    ])

    figure_files = find_files([
        "reports/**/*.png",
        "reports/**/*.jpg",
        "reports/**/*.jpeg",
        "reports/**/*.svg"
    ])

    table_files = find_files([
        "reports/**/*.csv",
        "outputs/**/*.csv"
    ])

    json_files = find_files([
        "reports/**/*.json",
        "outputs/**/*.json"
    ])

    return {
        "metric_files": metric_files,
        "model_files": model_files,
        "fairness_files": fairness_files,
        "interpretation_files": interpretation_files,
        "figure_files": figure_files,
        "table_files": table_files,
        "json_files": json_files
    }


def best_table_section(title, files, max_files=3, max_rows=12):
    """Create a Markdown section with preview tables."""
    if not files:
        return f"## {title}\n\n_No files found._\n"

    section = [f"## {title}\n"]

    for file in files[:max_files]:
        section.append(f"### `{relpath(file)}`\n")

        if file.suffix.lower() == ".csv":
            df = safe_read_csv(file)
            section.append(dataframe_to_markdown(df, max_rows=max_rows))
            section.append("")
        elif file.suffix.lower() == ".json":
            data = safe_read_json(file)
            if data is None:
                section.append("_Could not read JSON file._\n")
            else:
                json_text = json.dumps(data, indent=2)
                if len(json_text) > 3000:
                    json_text = json_text[:3000] + "\n... truncated ..."
                section.append(f"```json\n{json_text}\n```\n")
        else:
            section.append("_Preview not available for this file type._\n")

    return "\n".join(section)


def create_artifact_index(artifacts):
    """Save artifact index CSV."""
    rows = []

    for category, files in artifacts.items():
        for file in files:
            rows.append({
                "category": category,
                "path": relpath(file),
                "file_name": file.name,
                "extension": file.suffix.lower(),
                "size_kb": round(file.stat().st_size / 1024, 2)
            })

    index_df = pd.DataFrame(rows)

    if index_df.empty:
        index_df = pd.DataFrame(columns=[
            "category", "path", "file_name", "extension", "size_kb"
        ])

    index_path = FINAL_DIR / "artifact_index.csv"
    index_df.to_csv(index_path, index=False)
    return index_path, index_df


# ---------------------------------------------------------------------
# Generate Markdown artifacts
# ---------------------------------------------------------------------

def generate_model_card(dataset_summary, artifacts):
    today = datetime.now().strftime("%B %d, %Y")

    metric_section = best_table_section(
        "Model Performance Evidence",
        artifacts["metric_files"],
        max_files=4,
        max_rows=12
    )

    fairness_section = best_table_section(
        "Fairness and Measurement Diagnostics",
        artifacts["fairness_files"],
        max_files=4,
        max_rows=12
    )

    interpretation_section = best_table_section(
        "Interpretability Evidence",
        artifacts["interpretation_files"],
        max_files=4,
        max_rows=12
    )

    target_text = dataset_summary["target_found"] or "Not automatically detected"

    model_card = f"""# Model Card: PISA 2022 Student Success Prediction

Generated on: {today}

## 1. Model Overview

This model card documents a student-success prediction workflow built from the PISA 2022 student-level dataset. The project combines educational measurement, statistical modeling, machine learning, fairness diagnostics, and reproducible analytics.

The modeling objective is to estimate student success using cleaned student-level predictors and to evaluate the model through predictive performance, interpretability, and fairness/measurement checks.

## 2. Intended Use

The model is intended for research, educational analytics, and portfolio demonstration purposes. Appropriate uses include:

- analyzing predictors associated with student success;
- comparing interpretable and machine-learning models;
- studying educational measurement concepts such as construct validity, subgroup performance, and fairness;
- producing reproducible GitHub-ready evidence for future academic or applied work.

This model should not be used as a standalone high-stakes decision system for individual students.

## 3. Dataset Summary

Main detected dataset:

`{relpath(dataset_summary["dataset_path"]) if dataset_summary["dataset_path"] else "No processed dataset detected"}`

Rows: {dataset_summary["n_rows"]}

Columns: {dataset_summary["n_cols"]}

Detected target variable: `{target_text}`

### Dataset Preview

{dataset_summary["preview_table"]}

## 4. Target Definition

The expected target variable is `student_success`, defined in earlier project steps from PISA achievement and/or success-related criteria. The exact construction should be traceable in the data-preparation script.

For educational-measurement validity, the target should be interpreted as an operational research label, not as a complete measure of student ability or long-term educational outcome.

## 5. Model Inputs

Model inputs may include student-level demographic, socioeconomic, school-context, and achievement-related variables produced during the cleaned dataset step.

Important measurement caution: PISA plausible values are not ordinary test scores. When using plausible values, interpretation should acknowledge that they represent multiple imputed proficiency estimates designed for population-level inference.

## 6. Model Outputs

The model output is a predicted probability or classification of student success. For reporting, probability bands and subgroup-level summaries are preferable to deterministic individual-level decisions.

{metric_section}

{interpretation_section}

{fairness_section}

## 7. Fairness and Measurement Considerations

The project evaluates subgroup performance and fairness-related diagnostics. These checks are important because educational prediction models can reproduce structural differences in access, resources, socioeconomic status, language background, gender, or school context.

Recommended interpretation:

- Compare performance across subgroups, not only overall accuracy.
- Examine false positive and false negative rates when classification thresholds are used.
- Treat subgroup gaps as signals requiring measurement and policy interpretation.
- Avoid causal claims unless the analysis design explicitly supports causal inference.

## 8. Limitations

Key limitations include:

- PISA is cross-sectional and does not establish causal effects.
- The `student_success` label is an operational project label.
- Plausible values require careful interpretation.
- Country, school, and socioeconomic effects may be confounded.
- Fairness metrics do not alone determine whether a model is educationally valid.
- The model should not be used for high-stakes decisions about individual students.

## 9. Recommended Monitoring

If the workflow is extended, recommended monitoring includes:

- subgroup performance drift;
- missing-data drift;
- feature-distribution drift;
- calibration drift;
- stability of top predictors;
- threshold sensitivity;
- documentation updates after each model retraining.

## 10. Reproducibility

The project is organized as a reproducible GitHub workflow. The expected sequence is:

1. download and store raw PISA files locally;
2. inspect `.sav` files;
3. create a cleaned student-level dataset;
4. run exploratory data analysis;
5. train prediction models;
6. run diagnostics and fairness checks;
7. generate this model card and final report.

"""

    return model_card


def generate_final_report(dataset_summary, artifacts, index_path):
    today = datetime.now().strftime("%B %d, %Y")

    metric_section = best_table_section(
        "Predictive Modeling Results",
        artifacts["metric_files"],
        max_files=5,
        max_rows=15
    )

    fairness_section = best_table_section(
        "Diagnostics, Fairness, and Measurement Checks",
        artifacts["fairness_files"],
        max_files=5,
        max_rows=15
    )

    interpretation_section = best_table_section(
        "Feature Importance and Interpretation Outputs",
        artifacts["interpretation_files"],
        max_files=5,
        max_rows=15
    )

    figure_gallery = image_gallery_markdown(
        artifacts["figure_files"],
        max_images=25
    )

    report = f"""# Final Project Report: PsySuccess AI for PISA 2022

Generated on: {today}

## Executive Summary

This project builds a reproducible educational measurement and machine-learning workflow using the PISA 2022 student dataset. The workflow moves from raw international assessment data to a cleaned student-level analysis dataset, exploratory data analysis, student-success prediction models, model diagnostics, fairness checks, and final documentation.

The central research question is:

> Can student success be predicted from PISA 2022 student-level variables while maintaining interpretable, measurement-aware, and fairness-conscious reporting?

The project is suitable for GitHub publication, portfolio demonstration, and future paper development because it combines:

- real public educational assessment data;
- psychometric and educational measurement framing;
- classical exploratory statistics;
- supervised machine learning;
- model diagnostics;
- fairness and subgroup performance analysis;
- model-card documentation.

## 1. Project Purpose

The purpose of this project is to demonstrate how educational measurement and artificial intelligence can be combined responsibly. Instead of only maximizing predictive accuracy, the workflow emphasizes validity, fairness, interpretability, and reproducibility.

The project is designed to support future research directions in:

- student-success prediction;
- assessment analytics;
- fairness-aware educational AI;
- subgroup measurement diagnostics;
- international large-scale assessment analysis;
- AI governance for educational decision support.

## 2. Data Source and Analysis Dataset

The project uses PISA 2022 student-level data. Raw files should remain local and should not be committed to GitHub if they are large or restricted by repository-size rules.

Detected main processed dataset:

`{relpath(dataset_summary["dataset_path"]) if dataset_summary["dataset_path"] else "No processed dataset detected"}`

Rows: {dataset_summary["n_rows"]}

Columns: {dataset_summary["n_cols"]}

Detected target variable: `{dataset_summary["target_found"] or "Not automatically detected"}`

### Dataset Preview

{dataset_summary["preview_table"]}

## 3. Workflow Summary

The completed workflow contains the following major steps:

1. Create GitHub repository and local project structure.
2. Download PISA 2022 raw data.
3. Store raw files locally under `data/raw/`.
4. Unzip and inspect `.sav` files.
5. Create a clean student-level analysis dataset.
6. Run exploratory data analysis.
7. Train student-success prediction models.
8. Run model diagnostics, interpretation, and fairness checks.
9. Generate model card, final report, and README-ready result summary.

## 4. Exploratory Data Analysis Outputs

The EDA stage should include descriptive statistics and visual summaries for:

- `student_success`;
- achievement score distributions;
- country distribution;
- socioeconomic status and student success;
- gender and student success;
- school-level variation;
- missing values;
- correlation structure.

Detected figure files:

{file_list_markdown(artifacts["figure_files"])}

{metric_section}

{interpretation_section}

{fairness_section}

## 5. Figures

{figure_gallery}

## 6. Measurement and Fairness Interpretation

The project should be interpreted through both predictive and educational-measurement lenses.

A model can have acceptable overall performance while still showing subgroup instability or validity concerns. Therefore, the fairness and diagnostic outputs should be read as core evidence, not as optional secondary analysis.

Important interpretation principles:

- Overall performance is insufficient without subgroup checks.
- Accuracy should be considered alongside calibration, false positive rates, and false negative rates.
- Socioeconomic and school-level variables may encode structural inequality.
- PISA plausible values should not be interpreted as direct observed test scores.
- Predictive success does not imply causal explanation.

## 7. GitHub-Ready Deliverables

This final step created the following documentation artifacts:

- `docs/model_card.md`
- `reports/final_project_report.md`
- `README_results_summary.md`
- `{relpath(index_path)}`

These files can be committed to GitHub and used to document the project professionally.

## 8. Recommended README Result Summary

Use the generated file below as the result section for your main README:

`README_results_summary.md`

## 9. Limitations

The project has several important limitations:

- PISA is cross-sectional.
- The target label is project-defined.
- Student success is broader than test performance.
- Subgroup differences require educational and social interpretation.
- The model should not be used for individual high-stakes decisions.
- Public GitHub repositories should not include large raw data files.

## 10. Future Research Extensions

Strong extensions for a future paper include:

- country-specific model comparison;
- hierarchical or multilevel modeling;
- DIF-style subgroup measurement checks;
- calibration analysis by country and subgroup;
- comparison of classical statistical models and modern ML models;
- SHAP-based interpretation with measurement-theory discussion;
- robustness checks using alternative success-label definitions.

## 11. Artifact Index

A full artifact index was saved to:

`{relpath(index_path)}`

"""

    return report


def generate_readme_summary(dataset_summary, artifacts):
    metric_files_text = file_list_markdown(artifacts["metric_files"])
    fairness_files_text = file_list_markdown(artifacts["fairness_files"])
    figure_files_text = file_list_markdown(artifacts["figure_files"][:10])

    summary = f"""# README Results Summary

## Project Result Summary

This project develops a reproducible educational measurement and machine-learning workflow for predicting student success using the PISA 2022 student dataset.

The workflow includes:

- raw data organization;
- `.sav` file inspection;
- clean student-level dataset construction;
- exploratory data analysis;
- supervised prediction modeling;
- model diagnostics;
- fairness and subgroup checks;
- model-card documentation.

## Main Processed Dataset

Detected dataset:

`{relpath(dataset_summary["dataset_path"]) if dataset_summary["dataset_path"] else "No processed dataset detected"}`

Rows: {dataset_summary["n_rows"]}

Columns: {dataset_summary["n_cols"]}

Detected target variable: `{dataset_summary["target_found"] or "Not automatically detected"}`

## Modeling Evidence

Detected model/metric files:

{metric_files_text}

## Fairness and Diagnostic Evidence

Detected fairness/diagnostic files:

{fairness_files_text}

## Key Figures

{figure_files_text}

## Interpretation

The project should be interpreted as a research and portfolio workflow, not as a high-stakes student decision system. The strongest contribution is the combination of real educational assessment data, student-success prediction, interpretability, and fairness/measurement diagnostics.

## Main GitHub Documentation Files

- `docs/model_card.md`
- `reports/final_project_report.md`
- `README_results_summary.md`
- `reports/final/artifact_index.csv`

"""

    return summary


def main():
    print("=" * 80)
    print("Step I: Generating final model card, report, and README summary")
    print("=" * 80)

    dataset_summary = summarize_dataset()
    artifacts = collect_artifacts()
    index_path, index_df = create_artifact_index(artifacts)

    model_card = generate_model_card(dataset_summary, artifacts)
    final_report = generate_final_report(dataset_summary, artifacts, index_path)
    readme_summary = generate_readme_summary(dataset_summary, artifacts)

    model_card_path = DOCS_DIR / "model_card.md"
    final_report_path = REPORTS_DIR / "final_project_report.md"
    readme_summary_path = ROOT / "README_results_summary.md"

    model_card_path.write_text(model_card, encoding="utf-8")
    final_report_path.write_text(final_report, encoding="utf-8")
    readme_summary_path.write_text(readme_summary, encoding="utf-8")

    print("\nCreated files:")
    print(f"  - {relpath(model_card_path)}")
    print(f"  - {relpath(final_report_path)}")
    print(f"  - {relpath(readme_summary_path)}")
    print(f"  - {relpath(index_path)}")

    print("\nArtifact inventory:")
    print(f"  Metric files: {len(artifacts['metric_files'])}")
    print(f"  Model files: {len(artifacts['model_files'])}")
    print(f"  Fairness/diagnostic files: {len(artifacts['fairness_files'])}")
    print(f"  Interpretation files: {len(artifacts['interpretation_files'])}")
    print(f"  Figure files: {len(artifacts['figure_files'])}")
    print(f"  Table files: {len(artifacts['table_files'])}")
    print(f"  JSON files: {len(artifacts['json_files'])}")

    if dataset_summary["dataset_path"]:
        print("\nDetected main dataset:")
        print(f"  - {relpath(dataset_summary['dataset_path'])}")
        print(f"  - Rows: {dataset_summary['n_rows']}")
        print(f"  - Columns: {dataset_summary['n_cols']}")
        print(f"  - Target: {dataset_summary['target_found']}")
    else:
        print("\nWarning: No processed dataset was detected.")

    print("\nStep I completed successfully.")


if __name__ == "__main__":
    main()