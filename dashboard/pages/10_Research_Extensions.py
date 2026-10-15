"""Aggregate development extension results; no model loading."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st

st.title("Research Extensions: XGBoost, SHAP and Temporal Models")
path = Path(__file__).resolve().parents[2] / "docs/RESEARCH_EXTENSION_EVIDENCE.json"
if not path.is_file():
    st.info("Run the extension documentation stage first.")
    st.stop()
evidence = json.loads(path.read_text(encoding="utf-8"))
if evidence.get("status") != "PASS":
    st.error("Validated extension evidence is unavailable.")
    st.stop()
st.caption("ASD-FEAT exploratory development comparison | 1,510 recordings | 259 participants | five participant folds | threshold 0.5")
summary = pd.DataFrame(evidence["summary"])
columns = ["model", "asd_f1_mean", "asd_f1_std", "precision_mean", "recall_mean", "roc_auc_mean", "average_precision_mean"]
st.dataframe(summary[columns].round(4), hide_index=True, width="stretch")
st.bar_chart(summary.set_index("model")[["asd_f1_mean"]], horizontal=True)
st.write("LSTM and TCN did not exceed the matched mean-pooling LR control in this fixed configuration. Existing frozen models are preserved.")
st.write("XGBoost SHAP reconstruction, temporal saved inference and OOF coverage checks passed. Fold metrics were recalculated from saved predictions.")
st.info("Development research only. Scores do not diagnose autism or determine severity. Behavior features are human-coded; stored RGB order was retained without independently verified timestamps. No official test was evaluated for these extensions.")
st.caption("Team: Muhammad Danial, Hamza Rasheed, Muhammad Freeh")
