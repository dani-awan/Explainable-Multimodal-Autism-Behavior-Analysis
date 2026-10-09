from pathlib import Path
import json
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "docs/ASD_FEAT_E6_FROZEN_TEST_EVIDENCE.json"
st.title("E6 Selected Research Model: Frozen Test Results")
if not PATH.is_file():
    st.error("Frozen E6 aggregate evidence is missing.")
    st.stop()
evidence = json.loads(PATH.read_text(encoding="utf-8"))
comparison = pd.DataFrame(evidence["comparison"])
primary = comparison.loc[comparison["model"].eq(evidence["selected_model"])].iloc[0]
st.caption("RGB + human-coded behavior | Fixed threshold: " + str(evidence["threshold"]))
columns = st.columns(4)
columns[0].metric("Test recordings", evidence["test_recordings"])
columns[1].metric("Participants", evidence["test_participants"])
columns[2].metric("ASD F1", f"{primary['asd_f1']:.4f}")
columns[3].metric("Balanced accuracy", f"{primary['balanced_accuracy']:.4f}")
st.warning(evidence["interpretation_limits"])
st.subheader("Frozen recording-level comparison")
st.dataframe(comparison, hide_index=True, width="stretch")
st.caption("ROC-AUC is a ranking metric. F1 is not an accuracy percentage. High dummy accuracy reflects class imbalance.")
st.subheader("Selected-model prediction outcomes")
columns = st.columns(4)
for column, key, label in zip(columns, ["tp", "fp", "fn", "tn"], ["True positives", "False positives", "False negatives", "True negatives"]):
    column.metric(label, int(primary[key]))
st.subheader("Model and verification")
st.write("Input features:", evidence["feature_count"])
st.write("Development recordings:", evidence["development_recordings"])
st.write("Development participants:", evidence["development_participants"])
st.write("Saved metric recalculation:", evidence["metric_recalculation"])
st.caption("This page displays saved aggregate research results; it performs no inference or model fitting.")
