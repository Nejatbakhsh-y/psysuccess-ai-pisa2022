from pathlib import Path
import subprocess
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "final" / "public_repository_review.md"
OUT.parent.mkdir(parents=True, exist_ok=True)

required_files = [
    "README.md",
    "LICENSE",
    ".gitignore",
    "README_results_summary.md",
    "docs/model_card.md",
    "reports/final/final_project_report.md",
    "scripts/06_create_clean_student_dataset.py",
    "scripts/07_eda_student_dataset.py",
    "scripts/08_train_student_success_models.py",
    "scripts/09_model_diagnostics_fairness.py",
    "scripts/10_generate_model_card_report.py",
    "scripts/11_finalize_github_readme.py",
    "scripts/12_github_public_polish_check.py",
]

risky_extensions = [".sav", ".zip", ".parquet", ".pkl", ".joblib"]


def run_git(args):
    result = subprocess.run(
        ["git"] + args,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip()


def file_exists(path):
    return (ROOT / path).exists()


def main():
    tracked_files = run_git(["ls-files"]).splitlines()
    git_status = run_git(["status", "--short"])

    missing = [f for f in required_files if not file_exists(f)]

    risky_files = []
    for f in tracked_files:
        lower = f.lower()
        if lower.startswith("data/raw/") or any(lower.endswith(ext) for ext in risky_extensions):
            risky_files.append(f)

    lines = []
    lines.append("# Final Public Repository Review")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("## Repository Status")
    lines.append("")
    lines.append(f"- Git working tree clean: {'PASS' if git_status == '' else 'REVIEW'}")
    lines.append(f"- Required files present: {'PASS' if not missing else 'REVIEW'}")
    lines.append(f"- Raw data / large file safety: {'PASS' if not risky_files else 'REVIEW'}")
    lines.append("")

    lines.append("## Required File Check")
    lines.append("")
    for f in required_files:
        lines.append(f"- {'PASS' if file_exists(f) else 'REVIEW'}: `{f}`")
    lines.append("")

    lines.append("## Raw Data Safety Check")
    lines.append("")
    if risky_files:
        lines.append("The following tracked files require review before public sharing:")
        lines.append("")
        for f in risky_files:
            lines.append(f"- `{f}`")
    else:
        lines.append("No tracked raw PISA files or large model/data artifacts were found.")
    lines.append("")

    lines.append("## Reviewer-Perspective Review")
    lines.append("")
    lines.append("### 1. Professor")
    lines.append("- Strength: The project has a clear educational measurement and applied data science structure.")
    lines.append("- Review point: Ensure the README clearly explains the research problem, dataset, and reproducibility steps.")
    lines.append("")

    lines.append("### 2. Recruiter")
    lines.append("- Strength: The repository shows a complete end-to-end workflow: cleaning, EDA, modeling, diagnostics, fairness, and reporting.")
    lines.append("- Review point: Ensure the key results are visible quickly near the top of the README.")
    lines.append("")

    lines.append("### 3. Journal Reviewer")
    lines.append("- Strength: The project connects prediction, fairness, reproducibility, and educational measurement.")
    lines.append("- Review point: Avoid causal claims unless causal identification is formally added.")
    lines.append("")

    lines.append("### 4. Educational Measurement Researcher")
    lines.append("- Strength: The project uses real PISA 2022 large-scale assessment data.")
    lines.append("- Review point: Clearly state measurement-validity limitations of the constructed student_success target.")
    lines.append("")

    lines.append("### 5. Data Science Hiring Manager")
    lines.append("- Strength: The project demonstrates Python, GitHub workflow, machine learning, reporting, and AI governance documentation.")
    lines.append("- Review point: Ensure model metrics and fairness checks are easy to locate.")
    lines.append("")

    lines.append("## Final Decision")
    lines.append("")
    if not git_status and not missing and not risky_files:
        lines.append("Status: READY FOR PUBLIC REVIEW.")
    else:
        lines.append("Status: REVIEW BEFORE PUBLIC SHARING.")
    lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Created: {OUT}")


if __name__ == "__main__":
    main()