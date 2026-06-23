from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.data_loader import first_existing_column
from src.irt import fit_rasch_jml


def main(config_path: str) -> None:
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    processed_dir = Path(cfg["paths"]["processed_dir"])
    tables_dir = Path(cfg["paths"]["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    items = pd.read_parquet(processed_dir / "student_item_matrix.parquet")
    id_col = first_existing_column(items, cfg["id_columns"]["student_id_candidates"])
    item_cols = [c for c in items.columns if c != id_col]
    response = items[item_cols]

    result = fit_rasch_jml(response, max_iter=75)
    student_table = result.student_table(id_values=items[id_col].values).rename(columns={"student_id": id_col})
    item_table = result.item_table(item_names=item_cols)

    student_table.to_parquet(processed_dir / "irt_student_ability.parquet", index=False)
    item_table.to_csv(tables_dir / "irt_item_parameters.csv", index=False)
    print(f"Rasch iterations: {result.n_iter}")
    print("IRT outputs written.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/project_config.yaml")
    args = parser.parse_args()
    main(args.config)
