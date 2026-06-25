# PsySuccess AI: Educational Measurement and Student Success Prediction Using PISA 2022

Last updated: June 24, 2026

## Key Project Outputs

- [Final Project Report](reports/final/final_project_report.md)
- [Model Card](docs/model_card.md)
- [Results Summary](README_results_summary.md)
- [README Finalization Script](scripts/11_finalize_github_readme.py)
- [Public Repository Polish Check](scripts/12_github_public_polish_check.py)

## Project Overview

**PsySuccess AI** is a reproducible educational measurement and machine learning project using the real OECD PISA 2022 student dataset.

The project combines educational measurement, statistical analysis, machine learning, model diagnostics, subgroup fairness checks, and model-card-style reporting to predict and interpret student success in large-scale assessment data.

The purpose is not only to build a predictive model. The broader purpose is to demonstrate a responsible educational AI workflow that connects prediction accuracy with validity, interpretability, subgroup fairness diagnostics, and transparent reporting.

## Research Problem

Large-scale educational assessments contain rich information about student achievement, background, school context, and learning outcomes.

This project studies whether student-success status can be predicted responsibly from PISA 2022 data while preserving educational measurement logic.

The project addresses the following research questions:

1. How accurately can student-success status be predicted from PISA 2022 student, school, and background variables?
2. Which student, background, and school-context variables are associated with predicted student success?
3. Which machine learning models provide useful and interpretable prediction?
4. Do model errors differ across gender, socioeconomic, country, or school-level groups?
5. How can educational measurement validity and fairness diagnostics be integrated into a student-success prediction pipeline?

## Dataset

The project uses the PISA 2022 student questionnaire and achievement data.

Typical source file:

```text
CY08MSP_STU_QQQ.SAV
```

Raw data should be stored locally in:

```text
data/raw/
```

Raw PISA files are not pushed to GitHub.

This repository is designed so that code, reports, figures, and model artifacts can be public, while large raw data files remain local and excluded through `.gitignore`.

## Target Variable

The main supervised-learning target is:

```text
student_success
```

The target is created from student achievement information, including mathematics plausible values and a project-defined success threshold.

## Methods

### Educational Measurement

- PISA achievement-score interpretation
- Student-success threshold construction
- Plausible-value-aware reasoning
- Group-based measurement checks
- Validity-oriented interpretation
- Fairness and subgroup diagnostics

### Statistical Analysis

- Descriptive statistics
- Missing-value analysis
- Distributional analysis
- Group comparisons
- Correlation analysis
- School-level and country-level summaries

### Machine Learning

The project trains and compares student-success prediction models, including:

- Logistic Regression
- Random Forest
- Gradient Boosting / HistGradientBoosting
- Additional scikit-learn compatible classifiers as the project evolves

## Model Evaluation

Model performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC
- Confusion matrix
- Classification report
- Calibration diagnostics
- Probability-decile analysis
- Feature-importance analysis

## Fairness and Measurement Diagnostics

The project evaluates whether predictive performance differs across educationally meaningful groups, including:

- Gender
- Socioeconomic status
- Country or education system
- School-level variation
- Achievement-score bands
- Model error groups

The goal is not only prediction accuracy. The broader goal is responsible educational AI supported by diagnostics, transparency, and measurement validity.

## Main Generated Outputs

### Final Reports

- [Final Project Report](reports/final/final_project_report.md)
- [Model Card](docs/model_card.md)
- [Results Summary](README_results_summary.md)

### Diagnostic Reports

- `reports/model_diagnostics_fairness/09_model_diagnostics_fairness_summary.txt`
- `reports/final/artifact_index.csv`

### Tables and JSON Outputs

- `reports/model_diagnostics_fairness/09_calibration_by_decile.csv`
- `reports/model_diagnostics_fairness/09_fairness_flags.csv`
- `reports/model_diagnostics_fairness/09_measurement_checks.csv`
- `reports/model_diagnostics_fairness/09_model_diagnostics_metrics.csv`
- `reports/model_diagnostics_fairness/09_permutation_feature_importance.csv`
- `reports/model_diagnostics_fairness/09_prediction_math_score_alignment_by_decile.csv`
- `reports/model_diagnostics_fairness/09_subgroup_fairness_metrics.csv`
- `reports/model_diagnostics_fairness/09_success_rate_by_probability_decile.csv`
- `reports/pisa2022_student_clean_metadata.json`
- `reports/tables/08_confusion_matrices.csv`
- `reports/tables/08_feature_importance.csv`

### Figures

- `figures/model_diagnostics_fairness/09_calibration_curve.png`
- `figures/model_diagnostics_fairness/09_confusion_matrix.png`
- `figures/model_diagnostics_fairness/09_fairness_equal_opportunity_escs_quartile.png`
- `figures/model_diagnostics_fairness/09_mean_math_score_by_probability_decile.png`
- `figures/model_diagnostics_fairness/09_permutation_feature_importance.png`
- `figures/model_diagnostics_fairness/09_precision_recall_curve.png`
- `figures/model_diagnostics_fairness/09_probability_distribution_by_actual_outcome.png`
- `figures/model_diagnostics_fairness/09_roc_curve.png`
- `figures/model_diagnostics_fairness/09_success_rate_by_probability_decile.png`

### Model Files

- `models/08_best_student_success_model.joblib`
- `models/08_model_manifest.json`
- `models/09_best_diagnostic_student_success_model.joblib`

## Repository Structure

```text
psysuccess-ai-pisa2022/
|-- configs/
|-- dashboard/
|-- data/
|   |-- raw/              # Local only; raw files are not pushed
|   |-- interim/
|   |-- processed/
|-- docs/
|   |-- model_card.md
|-- figures/
|-- models/
|-- notebooks/
|-- outputs/
|-- reports/
|   |-- final/
|   |   |-- final_project_report.md
|   |-- figures/
|   |-- tables/
|-- scripts/
|   |-- 06_inspect_sav_files.py
|   |-- 07_clean_student_dataset.py
|   |-- 07_eda_student_dataset.py
|   |-- 08_train_student_success_models.py
|   |-- 09_model_diagnostics_fairness.py
|   |-- 10_generate_model_card_report.py
|   |-- 11_finalize_github_readme.py
|   |-- 12_github_public_polish_check.py
|-- src/
|-- tests/
|-- README.md
|-- README_results_summary.md
|-- LICENSE
|-- environment.yml
|-- requirements.txt
|-- .gitignore
```

## How to Reproduce the Project

These instructions are written for **Windows PowerShell**, not Bash.

### 1. Clone the repository

Open PowerShell or the VS Code PowerShell terminal.

Then run:

```powershell
git clone https://github.com/Nejatbakhsh-y/psysuccess-ai-pisa2022.git

Set-Location .\psysuccess-ai-pisa2022
```

If the repository is already on your computer, do not clone it again. Instead, go to the existing local folder:

```powershell
Set-Location "C:\Users\nejat\OneDrive\Desktop\UN\Skills\GitHub 2026\psysuccess-ai-pisa2022"

git status

git pull
```

### 2. Create and activate a virtual environment

Create the virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip

pip install -r requirements.txt
```

If the project uses Conda instead:

```powershell
conda env create -f environment.yml

conda activate psysuccess-ai-pisa2022
```

### 4. Add raw PISA data locally

Place the downloaded PISA 2022 files in:

```text
data/raw/
```

The main student file should include:

```text
CY08MSP_STU_QQQ.SAV
```

Do not commit raw data files to GitHub.

### 5. Run the workflow scripts

Run the scripts in order:

```powershell
python scripts/06_inspect_sav_files.py

python scripts/07_clean_student_dataset.py

python scripts/07_eda_student_dataset.py

python scripts/08_train_student_success_models.py

python scripts/09_model_diagnostics_fairness.py

python scripts/10_generate_model_card_report.py

python scripts/11_finalize_github_readme.py

python scripts/12_github_public_polish_check.py
```

### 6. Check Git status

Before pushing anything to GitHub, check the repository status:

```powershell
git status
```

### 7. Confirm raw data is not tracked

Run:

```powershell
git ls-files data/raw
```

Expected result:

```text

```

The command should return nothing.

### 8. Push updates to GitHub

After editing files, use:

```powershell
git add README.md

git commit -m "Polish public README"

git push
```

If there are no changes to commit, Git will report that the working tree is clean.

## Paper-Writing Direction

Possible manuscript title:

**Responsible Student-Success Prediction in Large-Scale Educational Assessment: A PISA 2022 Measurement and Machine Learning Study**

Possible research questions:

1. How accurately can student-success status be predicted from PISA 2022 student, school, and background variables?
2. Which variables are most associated with predicted student success?
3. Do model errors differ across gender, socioeconomic, country, or school-level groups?
4. How can educational measurement validity and fairness diagnostics be integrated into a student-success prediction pipeline?

Possible contributions:

- A reproducible open-source educational measurement and machine learning workflow.
- A student-success prediction pipeline using real international assessment data.
- Interpretable modeling and diagnostic outputs.
- Fairness-oriented subgroup evaluation.
- Model-card-style reporting for responsible educational AI.

## Project Status

Completed stages:

- Raw data organization
- PISA file inspection
- Clean student-level dataset construction
- Exploratory data analysis
- Student-success modeling
- Model diagnostics and fairness checks
- Model card and final project report generation
- README integration for GitHub presentation
- Public README and repository polish checks

## Responsible Use Statement

This project is intended for research, portfolio development, and responsible AI demonstration.

Predictions should not be used for high-stakes student decisions without additional validity evidence, external review, stakeholder input, institutional approval, and operational safeguards.

## License

This repository is released under the MIT License.

Dataset use must follow the official OECD/PISA data-use terms.

## Author

**Yousef Nejatbakhsh, Ph.D.**