from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="PsySuccess AI Dashboard", layout="wide")
st.title("PsySuccess AI Dashboard")
st.caption("Validity-gated psychometric machine learning for PISA 2022 student-success prediction")

processed = Path("data/processed")
tables = Path("reports/tables")

st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Page",
    [
        "Dataset Overview",
        "Reliability and Item Quality",
        "IRT Ability Distribution",
        "DIF and Fairness Audit",
        "Student Success Prediction",
        "Validity Gate",
    ],
)


def safe_read(path: Path):
    if not path.exists():
        st.warning(f"Missing file: {path}. Run the pipeline first.")
        st.stop()
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


if page == "Dataset Overview":
    labels = safe_read(processed / "success_labels.parquet")
    st.metric("Students with labels", f"{len(labels):,}")
    st.dataframe(labels.head(20), use_container_width=True)

if page == "Reliability and Item Quality":
    reliability = safe_read(tables / "ctt_reliability.csv")
    items = safe_read(tables / "ctt_item_statistics.csv")
    st.dataframe(reliability, use_container_width=True)
    fig = px.scatter(items, x="difficulty_p_value", y="discrimination_item_total_corr", hover_name="item")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(items, use_container_width=True)

if page == "IRT Ability Distribution":
    irt = safe_read(processed / "irt_student_ability.parquet")
    fig = px.histogram(irt, x="rasch_theta", nbins=50)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(irt.head(50), use_container_width=True)

if page == "DIF and Fairness Audit":
    dif = safe_read(tables / "dif_logistic_results.csv")
    st.metric("Uniform DIF flags, p < .01", int(dif.get("uniform_dif_flag_0_01", pd.Series(dtype=bool)).sum()))
    st.metric("Nonuniform DIF flags, p < .01", int(dif.get("nonuniform_dif_flag_0_01", pd.Series(dtype=bool)).sum()))
    st.dataframe(dif, use_container_width=True)

if page == "Student Success Prediction":
    metrics = safe_read(tables / "model_metrics.csv")
    st.dataframe(metrics, use_container_width=True)
    preds = safe_read(processed / "student_predictions_validity_gated.parquet")
    fig = px.histogram(preds, x="predicted_success_probability", nbins=50)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(preds.head(100), use_container_width=True)

if page == "Validity Gate":
    summary = safe_read(tables / "validity_gate_summary.csv")
    st.dataframe(summary, use_container_width=True)
    preds = safe_read(processed / "student_predictions_validity_gated.parquet")
    st.metric("Validity gate pass rate", f"{preds['validity_gate_pass'].mean():.1%}")
    st.dataframe(preds.head(100), use_container_width=True)
