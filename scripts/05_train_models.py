from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.data_loader import first_existing_column
from src.features import assemble_student_features
from src.modeling import train_model_suite
from src.validity_gate import apply_validity_gate


def main(config_path: str) -> None:
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    processed_dir = Path(cfg["paths"]["processed_dir"])
    tables_dir = Path(cfg["paths"]["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    student = pd.read_parquet(processed_dir / "student_questionnaire_clean.parquet")
    labels = pd.read_parquet(processed_dir / "success_labels.parquet")
    ctt_scores = pd.read_parquet(processed_dir / "ctt_student_scores.parquet")
    irt_scores = pd.read_parquet(processed_dir / "irt_student_ability.parquet")

    id_col = first_existing_column(student, cfg["id_columns"]["student_id_candidates"])
    candidate_cols = []
    for key in ["demographic_candidates", "socioeconomic_candidates", "school_candidates"]:
        candidate_cols.extend(cfg["features"].get(key, []))
    candidate_cols = [c for c in candidate_cols if c in student.columns]

    ctt = ctt_scores.set_index(id_col)
    features = assemble_student_features(
        student_df=student,
        id_column=id_col,
        candidate_columns=candidate_cols,
        ctt_scores=ctt,
        irt_scores=irt_scores,
    )
    modeling = features.merge(labels[[id_col, "success_label", "success_probability"]], on=id_col, how="inner")
    y = modeling["success_label"]
    x = modeling.drop(columns=["success_label", "success_probability"])

    metrics, fitted = train_model_suite(
        x,
        y,
        id_column=id_col,
        test_size=float(cfg["modeling"].get("test_size", 0.2)),
        random_state=int(cfg["modeling"].get("random_state", 42)),
    )
    metrics.to_csv(tables_dir / "model_metrics.csv", index=False)

    # Example full-data prediction with best model for dashboard prototyping.
    best_name = metrics.iloc[0]["model"]
    best_model = fitted[best_name]
    probs = best_model.predict_proba(x.drop(columns=[id_col], errors="ignore"))[:, 1]
    pred = modeling[[id_col, "success_label", "success_probability"]].copy()
    pred["predicted_success_probability"] = probs
    for col in ["valid_item_count", "rasch_theta_se", "rasch_theta", "raw_score", "percent_correct"]:
        if col in modeling.columns:
            pred[col] = modeling[col]
    gated = apply_validity_gate(
        pred,
        min_valid_item_count=int(cfg["modeling"].get("min_valid_item_count", 8)),
        max_irt_se=float(cfg["modeling"].get("max_irt_se_for_valid_prediction", 1.25)),
        probability_margin=float(cfg["modeling"].get("probability_gate_margin", 0.05)),
    )
    gated.to_parquet(processed_dir / "student_predictions_validity_gated.parquet", index=False)
    gated["validity_gate_reason"].value_counts().rename_axis("reason").reset_index(name="count").to_csv(
        tables_dir / "validity_gate_summary.csv", index=False
    )

    print(metrics.to_string(index=False))
    print(f"Best starter model: {best_name}")
    print("Model outputs written.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/project_config.yaml")
    args = parser.parse_args()
    main(args.config)
