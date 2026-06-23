from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.data_loader import first_existing_column
from src.dif import logistic_dif, mantel_haenszel_dif


def main(config_path: str) -> None:
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    processed_dir = Path(cfg["paths"]["processed_dir"])
    tables_dir = Path(cfg["paths"]["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    student = pd.read_parquet(processed_dir / "student_questionnaire_clean.parquet")
    items = pd.read_parquet(processed_dir / "student_item_matrix.parquet")
    ctt_scores = pd.read_parquet(processed_dir / "ctt_student_scores.parquet")

    id_col = first_existing_column(items, cfg["id_columns"]["student_id_candidates"])
    item_cols = [c for c in items.columns if c != id_col]
    group_col = cfg["features"].get("group_for_dif")
    if group_col not in student.columns:
        raise KeyError(f"DIF group column not found in student file: {group_col}")

    merged = items[[id_col] + item_cols].merge(student[[id_col, group_col]], on=id_col, how="left").merge(
        ctt_scores[[id_col, "raw_score"]], on=id_col, how="left"
    )

    logistic = logistic_dif(merged[item_cols], total_score=merged["raw_score"], group=merged[group_col])
    mh = mantel_haenszel_dif(merged[item_cols], total_score=merged["raw_score"], group=merged[group_col])
    logistic.to_csv(tables_dir / "dif_logistic_results.csv", index=False)
    mh.to_csv(tables_dir / "dif_mantel_haenszel_results.csv", index=False)
    print("DIF outputs written.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/project_config.yaml")
    args = parser.parse_args()
    main(args.config)
