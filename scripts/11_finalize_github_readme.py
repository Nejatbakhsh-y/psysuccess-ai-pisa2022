from pathlib import Path
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[1]
README_PATH = PROJECT_ROOT / "README.md"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "figures"
MODELS_DIR = PROJECT_ROOT / "models"


def find_files(folder: Path, suffixes: tuple[str, ...]) -> list[Path]:
    if not folder.exists():
        return []

    files = []
    for suffix in suffixes:
        files.extend(folder.rglob(f"*{suffix}"))

    return sorted(set(files))


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def file_list(files: list[Path], limit: int = 12) -> str:
    if not files:
        return "- No generated files detected yet."

    selected = files[:limit]
    lines = [f"- `{rel(path)}`" for path in selected]

    if len(files) > limit:
        lines.append(f"- ... and {len(files) - limit} additional file(s).")

    return "\n".join(lines)


def build_readme() -> str:
    today = datetime.now().strftime("%B %d, %Y")

    reports = find_files(REPORTS_DIR, (".md", ".txt", ".pdf", ".docx"))
    tables = find_files(REPORTS_DIR, (".csv", ".xlsx", ".json"))
    figures = find_files(FIGURES_DIR, (".png", ".jpg", ".jpeg", ".svg"))
    models = find_files(MODELS_DIR, (".pkl", ".joblib", ".json"))

    lines = [
        "# PsySuccess AI: Educational Measurement and Student Success Prediction Using PISA 2022",
        "",
        f"Last updated: {today}",
        "",
        "## Project Overview",
        "",
        "PsySuccess AI is an educational measurement and machine learning project using the real OECD PISA 2022 student dataset.",
        "",
        "The project combines educational measurement, statistical analysis, machine learning, model diagnostics, fairness checks, and model-card-style reporting to predict and interpret student success.",
        "",
        "## Research Problem",
        "",
        "Large-scale educational assessments contain rich information about student achievement, background, school context, and learning outcomes.",
        "",
        "This project studies whether student success can be predicted responsibly from PISA 2022 data while preserving educational measurement logic.",
        "",
        "The project addresses the following questions:",
        "",
        "- Which students are more likely to meet a defined success threshold?",
        "- Which student, background, and school-context variables are associated with success?",
        "- Which machine learning models provide useful and interpretable prediction?",
        "- Do model errors differ across student groups?",
        "- How can fairness and measurement diagnostics be integrated into an educational AI workflow?",
        "",
        "## Dataset",
        "",
        "The project uses the PISA 2022 student questionnaire and achievement data.",
        "",
        "Typical source file:",
        "",
        "```text",
        "CY08MSP_STU_QQQ.SAV",
        "```",
        "",
        "Raw data should be stored locally in:",
        "",
        "```text",
        "data/raw/",
        "```",
        "",
        "Raw PISA files are not pushed to GitHub.",
        "",
        "## Target Variable",
        "",
        "The main supervised-learning target is:",
        "",
        "```text",
        "student_success",
        "```",
        "",
        "The target is created from student achievement information, including mathematics plausible values and a project-defined success threshold.",
        "",
        "## Methods",
        "",
        "### Educational Measurement",
        "",
        "- PISA achievement-score interpretation",
        "- Student-success threshold construction",
        "- Plausible-value-aware reasoning",
        "- Group-based measurement checks",
        "- Validity-oriented interpretation",
        "- Fairness and subgroup diagnostics",
        "",
        "### Statistical Analysis",
        "",
        "- Descriptive statistics",
        "- Missing-value analysis",
        "- Distributional analysis",
        "- Group comparisons",
        "- Correlation analysis",
        "- School-level and country-level summaries",
        "",
        "### Machine Learning",
        "",
        "The project trains and compares student-success prediction models such as:",
        "",
        "- Logistic Regression",
        "- Random Forest",
        "- Gradient Boosting / HistGradientBoosting",
        "- Additional scikit-learn compatible classifiers as the project evolves",
        "",
        "## Model Evaluation",
        "",
        "Model performance is evaluated using:",
        "",
        "- Accuracy",
        "- Precision",
        "- Recall",
        "- F1 score",
        "- ROC-AUC",
        "- Confusion matrix",
        "- Classification report",
        "",
        "## Fairness and Measurement Diagnostics",
        "",
        "The project evaluates whether predictive performance differs across educationally meaningful groups, including:",
        "",
        "- Gender",
        "- Socioeconomic status",
        "- Country or education system",
        "- School-level variation",
        "- Achievement-score bands",
        "- Model error groups",
        "",
        "The goal is not only prediction accuracy. The goal is responsible educational AI supported by diagnostics, transparency, and measurement validity.",
        "",
        "## Main Generated Outputs",
        "",
        "### Reports",
        "",
        file_list(reports),
        "",
        "### Tables and JSON Outputs",
        "",
        file_list(tables),
        "",
        "### Figures",
        "",
        file_list(figures),
        "",
        "### Model Files",
        "",
        file_list(models),
        "",
        "## Repository Structure",
        "",
        "```text",
        "psysuccess-ai-pisa2022/",
        "|-- configs/",
        "|-- dashboard/",
        "|-- data/",
        "|   |-- raw/",
        "|   |-- interim/",
        "|   |-- processed/",
        "|-- docs/",
        "|-- figures/",
        "|-- models/",
        "|-- notebooks/",
        "|-- outputs/",
        "|-- reports/",
        "|   |-- figures/",
        "|   |-- tables/",
        "|   |-- final/",
        "|-- scripts/",
        "|   |-- 06_create_clean_student_dataset.py",
        "|   |-- 07_eda_student_dataset.py",
        "|   |-- 08_train_student_success_models.py",
        "|   |-- 09_model_diagnostics_fairness.py",
        "|   |-- 10_generate_model_card_report.py",
        "|   |-- 11_finalize_github_readme.py",
        "|-- src/",
        "|-- tests/",
        "|-- README.md",
        "|-- environment.yml",
        "|-- .gitignore",
        "```",
        "",
        "## How to Reproduce the Project",
        "",
        "### 1. Clone the repository",
        "",
        "```bash",
        "git clone <your-repository-url>",
        "cd psysuccess-ai-pisa2022",
        "```",
        "",
        "### 2. Activate the virtual environment",
        "",
        "```powershell",
        ".\\.venv\\Scripts\\Activate.ps1",
        "```",
        "",
        "### 3. Install dependencies",
        "",
        "```powershell",
        "pip install -r requirements.txt",
        "```",
        "",
        "If the project uses Conda:",
        "",
        "```powershell",
        "conda env create -f environment.yml",
        "conda activate psysuccess-ai-pisa2022",
        "```",
        "",
        "### 4. Add raw PISA data locally",
        "",
        "Place the downloaded PISA 2022 files in:",
        "",
        "```text",
        "data/raw/",
        "```",
        "",
        "The main student file should include:",
        "",
        "```text",
        "CY08MSP_STU_QQQ.SAV",
        "```",
        "",
        "### 5. Run the workflow scripts",
        "",
        "```powershell",
        "python scripts/06_create_clean_student_dataset.py",
        "python scripts/07_eda_student_dataset.py",
        "python scripts/08_train_student_success_models.py",
        "python scripts/09_model_diagnostics_fairness.py",
        "python scripts/10_generate_model_card_report.py",
        "python scripts/11_finalize_github_readme.py",
        "```",
        "",
        "## Paper-Writing Direction",
        "",
        "Possible manuscript title:",
        "",
        "**Responsible Student-Success Prediction in Large-Scale Educational Assessment: A PISA 2022 Measurement and Machine Learning Study**",
        "",
        "Possible research questions:",
        "",
        "1. How accurately can student-success status be predicted from PISA 2022 student, school, and background variables?",
        "2. Which variables are most associated with predicted student success?",
        "3. Do model errors differ across gender, socioeconomic, country, or school-level groups?",
        "4. How can educational measurement validity and fairness diagnostics be integrated into a student-success prediction pipeline?",
        "",
        "Possible contributions:",
        "",
        "- A reproducible open-source educational measurement and machine learning workflow.",
        "- A student-success prediction pipeline using real international assessment data.",
        "- Interpretable modeling and diagnostic outputs.",
        "- Fairness-oriented subgroup evaluation.",
        "- Model-card-style reporting for responsible educational AI.",
        "",
        "## Project Status",
        "",
        "Completed stages:",
        "",
        "- Raw data organization",
        "- PISA file inspection",
        "- Clean student-level dataset construction",
        "- Exploratory data analysis",
        "- Student-success modeling",
        "- Model diagnostics and fairness checks",
        "- Model card and final project report generation",
        "- README integration for GitHub presentation",
        "",
        "## Responsible Use Statement",
        "",
        "This project is intended for research, portfolio development, and responsible AI demonstration. Predictions should not be used for high-stakes student decisions without additional validity evidence, external review, stakeholder input, and institutional safeguards.",
        "",
        "## License",
        "",
        "Add a license before public release. A common code license is the MIT License. Dataset use must follow the official OECD/PISA terms.",
        "",
        "## Author",
        "",
        "**Yousef Nejatbakhsh, Ph.D.**",
        "",
    ]

    return "\n".join(lines)


def main() -> None:
    readme_text = build_readme()

    with open(README_PATH, "w", encoding="utf-8") as file:
        file.write(readme_text)

    print("Step J completed successfully.")
    print(f"README updated: {README_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()