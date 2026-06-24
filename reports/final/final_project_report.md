# PsySuccess AI PISA 2022: Final Project Report

## Project purpose

This project develops a reproducible educational measurement and AI workflow for predicting student success using the OECD PISA 2022 student dataset. The project combines educational measurement concepts, exploratory data analysis, predictive modeling, model interpretation, and fairness-oriented diagnostic checks.

## Research problem

The central research question is whether student background, learning-context, and assessment-related variables can be used to predict student success in a transparent and reproducible way while preserving measurement-awareness and avoiding careless interpretation of predictive performance.

## Dataset

The project uses the public OECD PISA 2022 student dataset. Raw data files are intentionally excluded from Git tracking and must be downloaded locally by the user into the appropriate raw-data directory.

## Methods

The project workflow includes:

1. Repository and environment setup.
2. Raw-data download and local storage.
3. PISA file inspection.
4. Clean student-level analysis dataset construction.
5. Exploratory data analysis.
6. Student-success prediction modeling.
7. Model diagnostics, interpretation, and fairness/measurement checks.
8. Model card and README-ready reporting.
9. GitHub public-release safety checks.

## Models

The project trains baseline and machine-learning models for student-success prediction. The modeling workflow is designed to compare predictive performance, inspect feature importance, and evaluate whether model behavior is reasonable for educational measurement use.

## Fairness and measurement checks

The project includes checks related to subgroup performance, interpretability, and measurement-oriented caution. These checks are intended to support responsible interpretation rather than to claim causal effects.

## Main outputs

Important repository outputs include:

- `README.md`
- `README_results_summary.md`
- `docs/model_card.md`
- `reports/final/artifact_index.csv`
- `scripts/12_github_public_polish_check.py`

## Reproducibility

To reproduce the project, a user should:

1. Clone the GitHub repository.
2. Create and activate the Python environment.
3. Install the required packages.
4. Download the PISA 2022 raw data locally.
5. Run the numbered scripts in order from `scripts/00_check_environment.py` through `scripts/12_github_public_polish_check.py`.

## Public-release status

The repository has been checked so that raw data files are not tracked by Git. The `data/raw` directory is excluded from public version control to avoid accidentally publishing large raw data files.

## Paper-writing direction

This project can be developed into a paper focused on educational measurement, predictive validity, student-success modeling, and responsible AI diagnostics using large-scale international assessment data.
