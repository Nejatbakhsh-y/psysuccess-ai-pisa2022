# Raw data download instructions

Do not commit raw PISA files to GitHub.

## Official sources

Use the official OECD PISA 2022 database and data-file pages:

- OECD PISA 2022 Database: https://www.oecd.org/en/data/datasets/pisa-2022-database.html
- OECD PISA 2022 Data Files index: https://webfs.oecd.org/pisa2022/index.html
- PISA 2022 Technical Report: https://www.oecd.org/content/dam/oecd/en/publications/reports/2024/03/pisa-2022-technical-report_599753f0/01820d6d-en.pdf

## Files to download

For the first reproducible version, download these files:

1. Student questionnaire data file
2. School questionnaire data file
3. Cognitive item data file
4. Codebook
5. Questionnaire-item compendium
6. Cognitive-item compendium
7. PISA 2022 Technical Report

The student-questionnaire file usually contains student-level background variables and plausible values. The cognitive item file contains item-response information. The school file contains school-context variables.

## Suggested local names

After downloading, rename files into simple local names:

```text
data/raw/student_questionnaire.sav
data/raw/school_questionnaire.sav
data/raw/cognitive_items.sav
```

Then edit `configs/project_config.yaml`:

```yaml
files:
  student_questionnaire: data/raw/student_questionnaire.sav
  school_questionnaire: data/raw/school_questionnaire.sav
  cognitive_items: data/raw/cognitive_items.sav
```

## Important data-use note

This repository is code-only. Raw PISA files should remain local. Do not imply that OECD endorses this project. Cite OECD materials properly in any report, paper, or GitHub README.
