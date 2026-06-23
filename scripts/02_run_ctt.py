from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.ctt import ctt_item_table, kr20, student_raw_scores
from src.data_loader import first_existing_column


def main(config_path: str) -> None:
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    processed_dir = Path(cfg["paths"]["processed_dir"])
    tables_dir = Path(cfg["paths"]["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    items = pd.read_parquet(processed_dir / "student_item_matrix.parquet")
    id_col = first_existing_column(items, cfg["id_columns"]["student_id_candidates"])
    item_cols = [c for c in items.columns if c != id_col]
    response = items[item_cols]

    item_table = ctt_item_table(response)
    student_scores = student_raw_scores(response)
    student_scores.insert(0, id_col, items[id_col].values)

    alpha = kr20(response)
    reliability = pd.DataFrame([{"coefficient": "KR20/Cronbach_alpha", "value": alpha, "n_items": len(item_cols), "n_students": len(items)}])

    item_table.to_csv(tables_dir / "ctt_item_statistics.csv", index=False)
    student_scores.to_parquet(processed_dir / "ctt_student_scores.parquet", index=False)
    reliability.to_csv(tables_dir / "ctt_reliability.csv", index=False)

    print(reliability.to_string(index=False))
    print("CTT outputs written.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/project_config.yaml")
    args = parser.parse_args()
    main(args.config)
