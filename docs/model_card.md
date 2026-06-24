# Model Card: PISA 2022 Student Success Prediction

Generated on: June 23, 2026

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

## 4. Target Definition

The expected target variable is `student_success`, defined in earlier project steps from PISA achievement and/or success-related criteria. The exact construction should be traceable in the data-preparation script.

For educational-measurement validity, the target should be interpreted as an operational research label, not as a complete measure of student ability or long-term educational outcome.

## 5. Model Inputs

Model inputs may include student-level demographic, socioeconomic, school-context, and achievement-related variables produced during the cleaned dataset step.

Important measurement caution: PISA plausible values are not ordinary test scores. When using plausible values, interpretation should acknowledge that they represent multiple imputed proficiency estimates designed for population-level inference.

## 6. Model Outputs

The model output is a predicted probability or classification of student success. For reporting, probability bands and subgroup-level summaries are preferable to deterministic individual-level decisions.

## Model Performance Evidence

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


## Interpretability Evidence

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


## Fairness and Measurement Diagnostics

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

