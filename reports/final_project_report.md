# Final Project Report: PsySuccess AI for PISA 2022

Generated on: June 23, 2026

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

`data/processed/pisa2022_student_clean.csv`

Rows: 613744

Columns: 60

Detected target variable: `student_success`

### Dataset Preview

| country_code   |   country_id |   school_id |   student_id |   student_weight |   math_score_mean_pv |   reading_score_mean_pv |   science_score_mean_pv |   math_success_level2 |   reading_success_level2 |   science_success_level2 |   success_all_three_level2 |
|:---------------|-------------:|------------:|-------------:|-----------------:|---------------------:|------------------------:|------------------------:|----------------------:|-------------------------:|-------------------------:|---------------------------:|
| ALB            |            8 |      800282 |       800001 |           3.1587 |              223.035 |                 249.803 |                 301.26  |                     0 |                        0 |                        0 |                          0 |
| ALB            |            8 |      800115 |       800002 |           4.3452 |              308.494 |                 288.9   |                 303.531 |                     0 |                        0 |                        0 |                          0 |
| ALB            |            8 |      800242 |       800003 |           7.8386 |              313.735 |                 311.779 |                 323.649 |                     0 |                        0 |                        0 |                          0 |
| ALB            |            8 |      800245 |       800005 |           8.4915 |              298.732 |                 300.775 |                 210.15  |                     0 |                        0 |                        0 |                          0 |
| ALB            |            8 |      800285 |       800006 |           3.7096 |              475.751 |                 486.669 |                 466.757 |                     1 |                        1 |                        1 |                          1 |
| ALB            |            8 |      800172 |       800007 |           4.2595 |              521.283 |                 430.567 |                 501.594 |                     1 |                        1 |                        1 |                          1 |
| ALB            |            8 |      800082 |       800008 |           5.1236 |              346.857 |                 356.243 |                 328.338 |                     0 |                        0 |                        0 |                          0 |
| ALB            |            8 |      800274 |       800009 |           6.3052 |              258.299 |                 308.504 |                 332.082 |                     0 |                        0 |                        0 |                          0 |

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

- [`reports/figures/08_best_confusion_matrix.png`](reports/figures/08_best_confusion_matrix.png)
- [`reports/figures/08_roc_curves.png`](reports/figures/08_roc_curves.png)

## Predictive Modeling Results

### `reports/model_diagnostics_fairness/09_model_diagnostics_metrics.csv`

| model                        |   threshold |   test_accuracy |   test_precision |   test_recall |   test_f1 |   test_roc_auc |   test_average_precision |   test_brier_score |   test_log_loss |   train_roc_auc |   train_average_precision |   roc_auc_gap_train_minus_test |
|:-----------------------------|------------:|----------------:|-----------------:|--------------:|----------:|---------------:|-------------------------:|-------------------:|----------------:|----------------:|--------------------------:|-------------------------------:|
| logistic_regression_balanced |      0.9908 |          1      |                1 |        0.9999 |    1      |              1 |                        1 |             0      |          0.0002 |               1 |                         1 |                              0 |
| random_forest_balanced       |      0.6924 |          0.9999 |                1 |        0.9998 |    0.9999 |              1 |                        1 |             0.0017 |          0.02   |               1 |                         1 |                              0 |

### `reports/model_diagnostics_fairness/09_subgroup_fairness_metrics.csv`

| group_variable   | group_value     |    n |   actual_success_rate |   average_predicted_probability |   predicted_positive_rate |   accuracy |   precision |   recall_equal_opportunity |   false_positive_rate |     f1 |   roc_auc |   predicted_positive_rate_gap |   recall_gap_equal_opportunity |   false_positive_rate_gap |   roc_auc_gap |   average_probability_gap |
|:-----------------|:----------------|-----:|----------------------:|--------------------------------:|--------------------------:|-----------:|------------:|---------------------------:|----------------------:|-------:|----------:|------------------------------:|-------------------------------:|--------------------------:|--------------:|--------------------------:|
| escs_quartile    | Missing         | 1257 |                0.4073 |                          0.4073 |                    0.4073 |     1      |           1 |                     1      |                     0 | 1      |         1 |                       -0.1372 |                         0.0001 |                         0 |             0 |                   -0.1372 |
| escs_quartile    | Q1_lowest_ESCS  | 7122 |                0.2731 |                          0.273  |                    0.273  |     0.9999 |           1 |                     0.9995 |                     0 | 0.9997 |         1 |                       -0.2716 |                        -0.0005 |                         0 |             0 |                   -0.2715 |
| escs_quartile    | Q2              | 7291 |                0.488  |                          0.488  |                    0.488  |     1      |           1 |                     1      |                     0 | 1      |         1 |                       -0.0566 |                         0.0001 |                         0 |             0 |                   -0.0566 |
| escs_quartile    | Q3              | 7092 |                0.6279 |                          0.6279 |                    0.6279 |     1      |           1 |                     1      |                     0 | 1      |         1 |                        0.0833 |                         0.0001 |                         0 |             0 |                    0.0833 |
| escs_quartile    | Q4_highest_ESCS | 7238 |                0.811  |                          0.811  |                    0.811  |     1      |           1 |                     1      |                     0 | 1      |         1 |                        0.2664 |                         0.0001 |                         0 |             0 |                    0.2664 |

### `reports/tables/08_model_metrics.csv`

| model               |   accuracy |   balanced_accuracy |   precision |   recall |   f1 |   roc_auc |   train_rows |   test_rows |   num_features_before_encoding |
|:--------------------|-----------:|--------------------:|------------:|---------:|-----:|----------:|-------------:|------------:|-------------------------------:|
| logistic_regression |          1 |                   1 |           1 |        1 |    1 |         1 |       490995 |      122749 |                             23 |
| random_forest       |          1 |                   1 |           1 |        1 |    1 |         1 |       490995 |      122749 |                             23 |
| gradient_boosting   |          1 |                   1 |           1 |        1 |    1 |         1 |       490995 |      122749 |                             23 |


## Feature Importance and Interpretation Outputs

### `reports/model_diagnostics_fairness/09_permutation_feature_importance.csv`

| feature                  |   importance_mean |   importance_std |
|:-------------------------|------------------:|-----------------:|
| math_success_level2      |            0.2571 |           0.0031 |
| country_code             |            0      |           0      |
| reading_success_level2   |            0      |           0      |
| science_success_level2   |            0      |           0      |
| success_all_three_level2 |            0      |           0      |
| gender_raw               |            0      |           0      |
| age                      |            0      |           0      |
| grade                    |            0      |           0      |
| immigrant_status         |            0      |           0      |
| grade_repetition         |            0      |           0      |
| math_ease                |            0      |           0      |
| math_motivation          |            0      |           0      |
| school_belonging         |            0      |           0      |
| bullied_index            |            0      |           0      |
| feeling_safe             |            0      |           0      |

### `reports/tables/08_feature_importance.csv`

| model               | feature                       |   importance_value |   absolute_importance_value | importance_type   |
|:--------------------|:------------------------------|-------------------:|----------------------------:|:------------------|
| logistic_regression | num__math_success_level2      |             8.4821 |                      8.4821 | coefficient       |
| logistic_regression | num__success_all_three_level2 |             1.1754 |                      1.1754 | coefficient       |
| logistic_regression | num__science_success_level2   |             0.3933 |                      0.3933 | coefficient       |
| logistic_regression | cat__country_code_KAZ         |             0.1939 |                      0.1939 | coefficient       |
| logistic_regression | cat__country_code_ESP         |             0.1347 |                      0.1347 | coefficient       |
| logistic_regression | num__reading_success_level2   |             0.1243 |                      0.1243 | coefficient       |
| logistic_regression | num__home_possessions         |             0.1079 |                      0.1079 | coefficient       |
| logistic_regression | cat__country_code_QAZ         |             0.1065 |                      0.1065 | coefficient       |
| logistic_regression | num__gender_raw               |             0.1028 |                      0.1028 | coefficient       |
| logistic_regression | num__math_self_efficacy       |             0.0995 |                      0.0995 | coefficient       |
| logistic_regression | cat__country_code_MNG         |             0.095  |                      0.095  | coefficient       |
| logistic_regression | num__grade                    |             0.0926 |                      0.0926 | coefficient       |
| logistic_regression | num__grade_repetition         |            -0.082  |                      0.082  | coefficient       |
| logistic_regression | num__math_anxiety             |            -0.0801 |                      0.0801 | coefficient       |
| logistic_regression | num__math_ease                |             0.0782 |                      0.0782 | coefficient       |


## Diagnostics, Fairness, and Measurement Checks

### `reports/model_diagnostics_fairness/09_fairness_flags.csv`

| group_variable   | group_value     |    n |   actual_success_rate |   average_predicted_probability |   predicted_positive_rate |   accuracy |   precision |   recall_equal_opportunity |   false_positive_rate |     f1 |   roc_auc |   predicted_positive_rate_gap |   recall_gap_equal_opportunity |   false_positive_rate_gap |   roc_auc_gap |   average_probability_gap |
|:-----------------|:----------------|-----:|----------------------:|--------------------------------:|--------------------------:|-----------:|------------:|---------------------------:|----------------------:|-------:|----------:|------------------------------:|-------------------------------:|--------------------------:|--------------:|--------------------------:|
| escs_quartile    | Q4_highest_ESCS | 7238 |                0.811  |                          0.811  |                    0.811  |     1      |           1 |                     1      |                     0 | 1      |         1 |                        0.2664 |                         0.0001 |                         0 |             0 |                    0.2664 |
| escs_quartile    | Q1_lowest_ESCS  | 7122 |                0.2731 |                          0.273  |                    0.273  |     0.9999 |           1 |                     0.9995 |                     0 | 0.9997 |         1 |                       -0.2716 |                        -0.0005 |                         0 |             0 |                   -0.2715 |
| escs_quartile    | Missing         | 1257 |                0.4073 |                          0.4073 |                    0.4073 |     1      |           1 |                     1      |                     0 | 1      |         1 |                       -0.1372 |                         0.0001 |                         0 |             0 |                   -0.1372 |

### `reports/model_diagnostics_fairness/09_model_diagnostics_metrics.csv`

| model                        |   threshold |   test_accuracy |   test_precision |   test_recall |   test_f1 |   test_roc_auc |   test_average_precision |   test_brier_score |   test_log_loss |   train_roc_auc |   train_average_precision |   roc_auc_gap_train_minus_test |
|:-----------------------------|------------:|----------------:|-----------------:|--------------:|----------:|---------------:|-------------------------:|-------------------:|----------------:|----------------:|--------------------------:|-------------------------------:|
| logistic_regression_balanced |      0.9908 |          1      |                1 |        0.9999 |    1      |              1 |                        1 |             0      |          0.0002 |               1 |                         1 |                              0 |
| random_forest_balanced       |      0.6924 |          0.9999 |                1 |        0.9998 |    0.9999 |              1 |                        1 |             0.0017 |          0.02   |               1 |                         1 |                              0 |

### `reports/model_diagnostics_fairness/09_subgroup_fairness_metrics.csv`

| group_variable   | group_value     |    n |   actual_success_rate |   average_predicted_probability |   predicted_positive_rate |   accuracy |   precision |   recall_equal_opportunity |   false_positive_rate |     f1 |   roc_auc |   predicted_positive_rate_gap |   recall_gap_equal_opportunity |   false_positive_rate_gap |   roc_auc_gap |   average_probability_gap |
|:-----------------|:----------------|-----:|----------------------:|--------------------------------:|--------------------------:|-----------:|------------:|---------------------------:|----------------------:|-------:|----------:|------------------------------:|-------------------------------:|--------------------------:|--------------:|--------------------------:|
| escs_quartile    | Missing         | 1257 |                0.4073 |                          0.4073 |                    0.4073 |     1      |           1 |                     1      |                     0 | 1      |         1 |                       -0.1372 |                         0.0001 |                         0 |             0 |                   -0.1372 |
| escs_quartile    | Q1_lowest_ESCS  | 7122 |                0.2731 |                          0.273  |                    0.273  |     0.9999 |           1 |                     0.9995 |                     0 | 0.9997 |         1 |                       -0.2716 |                        -0.0005 |                         0 |             0 |                   -0.2715 |
| escs_quartile    | Q2              | 7291 |                0.488  |                          0.488  |                    0.488  |     1      |           1 |                     1      |                     0 | 1      |         1 |                       -0.0566 |                         0.0001 |                         0 |             0 |                   -0.0566 |
| escs_quartile    | Q3              | 7092 |                0.6279 |                          0.6279 |                    0.6279 |     1      |           1 |                     1      |                     0 | 1      |         1 |                        0.0833 |                         0.0001 |                         0 |             0 |                    0.0833 |
| escs_quartile    | Q4_highest_ESCS | 7238 |                0.811  |                          0.811  |                    0.811  |     1      |           1 |                     1      |                     0 | 1      |         1 |                        0.2664 |                         0.0001 |                         0 |             0 |                    0.2664 |


## 5. Figures

### 08 Best Confusion Matrix

![08 Best Confusion Matrix](reports/figures/08_best_confusion_matrix.png)

### 08 Roc Curves

![08 Roc Curves](reports/figures/08_roc_curves.png)


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
- `reports/final/artifact_index.csv`

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

`reports/final/artifact_index.csv`

