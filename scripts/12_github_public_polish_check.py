"""
Step K: GitHub public polish and repository safety check.

This script verifies that the repository is ready for public GitHub presentation.
It checks:

1. Required public-facing documentation files exist.
2. Raw data files are not accidentally tracked by Git.
3. Suspicious large/raw file formats are not tracked.
4. Very large tracked files are flagged.
5. The Git working tree status is shown for final review.

Run from the repository root:

    python scripts/12_github_public_polish_check.py
"""

from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "LICENSE",
    ".gitignore",
    "reports/final/final_project_report.md",
    "docs/model_card.md",
    "README_results_summary.md",
    "scripts/11_finalize_github_readme.py",
]

SUSPICIOUS_EXTENSIONS = {
    ".sav",
    ".zip",
    ".parquet",
    ".7z",
    ".rar",
    ".gz",
    ".xz",
}

LARGE_FILE_LIMIT_MB = 50


def run_git_command(args):
    """Run a Git command from the repository root."""
    result = subprocess.run(
        ["git"] + args,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        print(f"[ERROR] Git command failed: git {' '.join(args)}")
        print(result.stderr.strip())
        sys.exit(1)

    return result.stdout.strip()


def normalize_path(path_text):
    """Normalize paths for cross-platform checking."""
    return path_text.replace("\\", "/")


def check_required_files():
    print("\n[1] Checking required public-facing files...")

    missing_files = []

    for relative_path in REQUIRED_FILES:
        full_path = REPO_ROOT / relative_path
        if full_path.exists():
            print(f"    OK: {relative_path}")
        else:
            print(f"    MISSING: {relative_path}")
            missing_files.append(relative_path)

    return missing_files


def get_tracked_files():
    output = run_git_command(["ls-files"])
    if not output:
        return []

    return [line.strip() for line in output.splitlines() if line.strip()]


def check_raw_data_not_tracked(tracked_files):
    print("\n[2] Checking that data/raw files are not tracked...")

    tracked_raw_files = [
        path for path in tracked_files
        if normalize_path(path).startswith("data/raw/")
    ]

    if tracked_raw_files:
        print("    ERROR: The following raw data files are tracked by Git:")
        for path in tracked_raw_files:
            print(f"    - {path}")
    else:
        print("    OK: No tracked files found under data/raw/")

    return tracked_raw_files


def check_suspicious_file_extensions(tracked_files):
    print("\n[3] Checking for suspicious tracked raw-data file formats...")

    suspicious_files = []

    for path in tracked_files:
        suffix = Path(path).suffix.lower()
        if suffix in SUSPICIOUS_EXTENSIONS:
            suspicious_files.append(path)

    if suspicious_files:
        print("    ERROR: Suspicious tracked files found:")
        for path in suspicious_files:
            print(f"    - {path}")
    else:
        print("    OK: No tracked .sav, .zip, .parquet, .7z, .rar, .gz, or .xz files found")

    return suspicious_files


def check_large_tracked_files(tracked_files):
    print(f"\n[4] Checking for tracked files larger than {LARGE_FILE_LIMIT_MB} MB...")

    large_files = []

    for path in tracked_files:
        full_path = REPO_ROOT / path
        if full_path.exists() and full_path.is_file():
            size_mb = full_path.stat().st_size / (1024 * 1024)
            if size_mb > LARGE_FILE_LIMIT_MB:
                large_files.append((path, size_mb))

    if large_files:
        print("    WARNING: Large tracked files found:")
        for path, size_mb in large_files:
            print(f"    - {size_mb:.1f} MB: {path}")
    else:
        print("    OK: No tracked files larger than the threshold found")

    return large_files


def show_git_status():
    print("\n[5] Current Git status:")
    status = run_git_command(["status", "--short"])

    if status:
        print(status)
    else:
        print("    OK: Working tree is clean")


def main():
    print("=" * 72)
    print("Step K: GitHub Public Polish and Safety Check")
    print("=" * 72)
    print(f"Repository root: {REPO_ROOT}")

    missing_files = check_required_files()

    tracked_files = get_tracked_files()

    tracked_raw_files = check_raw_data_not_tracked(tracked_files)
    suspicious_files = check_suspicious_file_extensions(tracked_files)
    large_files = check_large_tracked_files(tracked_files)

    show_git_status()

    print("\n" + "=" * 72)
    print("Final Step K Result")
    print("=" * 72)

    blocking_issues = False

    if missing_files:
        blocking_issues = True
        print("\nMissing required files:")
        for path in missing_files:
            print(f"- {path}")

    if tracked_raw_files:
        blocking_issues = True
        print("\nRaw data files are tracked. These should be removed from Git tracking:")
        for path in tracked_raw_files:
            print(f"- {path}")

    if suspicious_files:
        blocking_issues = True
        print("\nSuspicious raw-data-style files are tracked:")
        for path in suspicious_files:
            print(f"- {path}")

    if large_files:
        print("\nLarge tracked files were found. Review before public GitHub release:")
        for path, size_mb in large_files:
            print(f"- {size_mb:.1f} MB: {path}")

    if blocking_issues:
        print("\nRESULT: FAIL")
        print("Fix the issues above before presenting the repository publicly.")
        sys.exit(1)

    print("\nRESULT: PASS")
    print("The repository structure and tracked-file safety checks look ready for GitHub presentation.")


if __name__ == "__main__":
    main()