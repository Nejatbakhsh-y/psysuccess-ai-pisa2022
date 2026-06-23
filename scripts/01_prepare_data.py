from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.data_loader import first_existing_column, read_table, write_parquet
from src.preprocessing import binary_item_matrix, find_item_columns, find_pv_columns, make_success_labels, normalize_columns


def main(config_path: str) -> None:
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    paths = cfg["paths"]
    files = cfg["files"]
    processed_dir = Path(paths["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    student_path = files.get("student_questionnaire")
    cognitive_path = files.get("cognitive_items")
    if not student_path or not cognitive_path:
        raise ValueError("Set files.student_questionnaire and files.cognitive_items in configs/project_config.yaml")

    student = normalize_columns(read_table(student_path))
    cognitive = normalize_columns(read_table(cognitive_path))

    id_col = first_existing_column(student, cfg["id_columns"]["student_id_candidates"])
    cognitive_id_col = first_existing_column(cognitive, cfg["id_columns"]["student_id_candidates"])

    pv_cols = [c for c in cfg["pisa"].get("mathematics_pv_columns", []) if c in student.columns]
    if not pv_cols:
        pv_cols = find_pv_columns(student, cfg["pisa"].get("plausible_value_domain", "MATH"))

    labels = make_success_labels(
        student,
        pv_columns=pv_cols,
        threshold=float(cfg["pisa"]["success_threshold"]),
        id_column=id_col,
    )

    item_cols = find_item_columns(
        cognitive,
        id_columns=[cognitive_id_col],
        prefixes=cfg["features"].get("item_prefix_candidates", ["CM", "CR", "DM", "M"]),
        min_nonmissing_rate=float(cfg["features"].get("minimum_item_valid_rate", 0.20)),
        max_missing_rate=float(cfg["features"].get("maximum_item_missing_rate", 0.80)),
    )
    items = binary_item_matrix(cognitive, item_cols, cognitive_id_col)

    write_parquet(student, processed_dir / "student_questionnaire_clean.parquet")
    write_parquet(items, processed_dir / "student_item_matrix.parquet")
    write_parquet(labels, processed_dir / "success_labels.parquet")

    print(f"Student rows: {len(student):,}")
    print(f"Detected plausible-value columns: {pv_cols}")
    print(f"Detected item columns: {len(item_cols):,}")
    print("Prepared files written to data/processed/.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/project_config.yaml")
    args = parser.parse_args()
    main(args.config)
