# PsySuccess AI: Educational Measurement and Student Success Prediction Using PISA 2022

Last updated: June 24, 2026

## Key Project Outputs

- [Final Project Report](reports/final/final_project_report.md)
- [Model Card](docs/model_card.md)
- [Results Summary](README_results_summary.md)
- [README Finalization Script](scripts/11_finalize_github_readme.py)

## Project Overview

**PsySuccess AI** is a reproducible educational measurement and machine learning project using the real OECD PISA 2022 student dataset.

The project combines educational measurement, statistical analysis, machine learning, model diagnostics, subgroup fairness checks, and model-card-style reporting to predict and interpret student success in large-scale assessment data.

The goal is not only to build a predictive model. The goal is to demonstrate a responsible educational AI workflow that connects prediction accuracy with validity, interpretability, fairness diagnostics, and transparent reporting.

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