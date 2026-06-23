# PsySuccess AI: Validity-Gated Psychometric Machine Learning for Student Success Prediction Using PISA 2022

This repository implements a research-grade educational measurement and AI pipeline using OECD PISA 2022 public-use data.

## Research question

Can psychometric measurement features improve AI-based student-success prediction compared with standard machine-learning models that use only demographic, questionnaire, and school-context variables?

## Main contribution

The project treats classical test theory, Rasch/IRT ability, DIF diagnostics, plausible-value uncertainty, calibration, and subgroup fairness as first-class components of a predictive modeling workflow. The final model is not only an accuracy model; it is a validity-gated student-success prediction system.

## Dataset

Primary dataset: OECD PISA 2022 public-use files.

Raw PISA data files are not committed to this repository. Download the public-use files from OECD and place them under `data/raw/`.

Recommended files:

- Student questionnaire data file
- School questionnaire data file
- Cognitive item data file
- Codebook and compendia
- PISA 2022 Technical Report

See: `data/raw/README_download_instructions.md`.

## Target definition

The main target is mathematics proficiency success. A student is treated as successful when the student reaches at least a selected PISA mathematics threshold, such as Level 2.

Because PISA uses plausible values, this repository supports two label strategies:

1. Binary label from the average of PV1MATH to PV10MATH.
2. Measurement-aware success probability:

```text
success_probability = mean(PVkMATH >= threshold), k = 1,...,10
```

## Repository structure

```text
psysuccess-ai-pisa2022/
├── README.md
├── requirements.txt
├── environment.yml
├── pyproject.toml
├── .gitignore
├── LICENSE
├── configs/
│   └── project_config.yaml
├── data/
│   ├── raw/
│   │   └── README_download_instructions.md
│   ├── interim/
│   └── processed/
├── notebooks/
│   ├── 01_download_and_data_dictionary.ipynb
│   ├── 02_data_cleaning_and_merging.ipynb
│   ├── 03_classical_test_theory.ipynb
│   ├── 04_irt_modeling.ipynb
│   ├── 05_dif_fairness_analysis.ipynb
│   ├── 06_success_label_construction.ipynb
│   ├── 07_ai_prediction_models.ipynb
│   ├── 08_explainable_ai_and_calibration.ipynb
│   └── 09_validity_argument_report.ipynb
├── scripts/
│   ├── 00_check_environment.py
│   ├── 01_prepare_data.py
│   ├── 02_run_ctt.py
│   ├── 03_run_irt.py
│   ├── 04_run_dif.py
│   └── 05_train_models.py
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── ctt.py
│   ├── irt.py
│   ├── dif.py
│   ├── features.py
│   ├── modeling.py
│   ├── fairness.py
│   ├── explainability.py
│   ├── evaluation.py
│   └── validity_gate.py
├── dashboard/
│   └── app.py
├── reports/
│   ├── figures/
│   └── tables/
└── tests/
    ├── test_ctt.py
    ├── test_irt.py
    └── test_validity_gate.py
```

## Quick start

### 1. Create the GitHub repository locally

```bash
mkdir psysuccess-ai-pisa2022
cd psysuccess-ai-pisa2022
git init
```

If using this scaffold, unzip it and then run:

```bash
git init
git add .
git commit -m "Initial PsySuccess AI repository scaffold"
```

### 2. Create the Python environment

Using conda:

```bash
conda env create -f environment.yml
conda activate psysuccess-ai
```

Or using pip:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add the PISA files

Download the PISA 2022 files and put them in `data/raw/`. Do not push raw data to GitHub.

Expected examples:

```text
data/raw/student_questionnaire.*
data/raw/school_questionnaire.*
data/raw/cognitive_item.*
```

The scripts are designed to work with `.sav`, `.sas7bdat`, `.csv`, `.txt`, `.parquet`, or `.feather` where practical.

### 4. Run the pipeline

```bash
python scripts/00_check_environment.py
python scripts/01_prepare_data.py --config configs/project_config.yaml
python scripts/02_run_ctt.py --config configs/project_config.yaml
python scripts/03_run_irt.py --config configs/project_config.yaml
python scripts/04_run_dif.py --config configs/project_config.yaml
python scripts/05_train_models.py --config configs/project_config.yaml
```

## Main outputs

```text
data/processed/student_features.parquet
data/processed/success_labels.parquet
reports/tables/ctt_item_statistics.csv
reports/tables/irt_student_ability.csv
reports/tables/dif_logistic_results.csv
reports/tables/model_metrics.csv
reports/tables/validity_gate_summary.csv
```

## Model comparison design

| Model | Feature set | Purpose |
|---|---|---|
| M1 | Demographics only | Minimum baseline |
| M2 | Questionnaire and school variables | Contextual baseline |
| M3 | Raw score plus questionnaire variables | Traditional analytics baseline |
| M4 | CTT features plus questionnaire variables | Measurement-enhanced baseline |
| M5 | Rasch/IRT ability plus questionnaire variables | Psychometric AI model |
| M6 | IRT plus DIF-aware features | Fairness-aware psychometric AI |
| M7 | Full model plus calibration and SHAP | Final validity-gated model |

## Validity gate

The final model can flag a prediction as low validity when any of the following hold:

- Too few valid item responses.
- IRT standard error is too large.
- Predicted probability is poorly calibrated for the subgroup.
- Prediction confidence interval crosses the success threshold.
- High dependence on DIF-flagged item patterns.

## Dashboard

Run:

```bash
streamlit run dashboard/app.py
```

## Publication direction

A strong paper angle is:

**Validity-Gated Psychometric Machine Learning for Student Success Prediction in Large-Scale Assessment Data**

The GitHub version should emphasize reproducibility, measurement validity, model calibration, subgroup fairness, and transparent limitations.
