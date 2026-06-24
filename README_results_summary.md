# README Results Summary

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

`data/processed/pisa2022_student_clean.csv`

Rows: 613744

Columns: 60

Detected target variable: `student_success`

## Modeling Evidence

Detected model/metric files:

- [`reports/model_diagnostics_fairness/09_model_diagnostics_metrics.csv`](reports/model_diagnostics_fairness/09_model_diagnostics_metrics.csv)
- [`reports/model_diagnostics_fairness/09_subgroup_fairness_metrics.csv`](reports/model_diagnostics_fairness/09_subgroup_fairness_metrics.csv)
- [`reports/tables/08_model_metrics.csv`](reports/tables/08_model_metrics.csv)

## Fairness and Diagnostic Evidence

Detected fairness/diagnostic files:

- [`reports/model_diagnostics_fairness/09_fairness_flags.csv`](reports/model_diagnostics_fairness/09_fairness_flags.csv)
- [`reports/model_diagnostics_fairness/09_model_diagnostics_metrics.csv`](reports/model_diagnostics_fairness/09_model_diagnostics_metrics.csv)
- [`reports/model_diagnostics_fairness/09_subgroup_fairness_metrics.csv`](reports/model_diagnostics_fairness/09_subgroup_fairness_metrics.csv)

## Key Figures

- [`reports/figures/08_best_confusion_matrix.png`](reports/figures/08_best_confusion_matrix.png)
- [`reports/figures/08_roc_curves.png`](reports/figures/08_roc_curves.png)

## Interpretation

The project should be interpreted as a research and portfolio workflow, not as a high-stakes student decision system. The strongest contribution is the combination of real educational assessment data, student-success prediction, interpretability, and fairness/measurement diagnostics.

## Main GitHub Documentation Files

- `docs/model_card.md`
- `reports/final_project_report.md`
- `README_results_summary.md`
- `reports/final/artifact_index.csv`

