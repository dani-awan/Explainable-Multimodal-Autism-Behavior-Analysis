"""DREAM aggregate activity results; no model inference."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st

st.title("DREAM: Therapy-Activity Recognition")
path = Path(__file__).resolve().parents[2] / "docs/DREAM_ACTIVITY_EVIDENCE.json"
if not path.is_file():
    st.info("Run the DREAM results documentation stage first.")
    st.stop()
evidence = json.loads(path.read_text(encoding="utf-8"))
if evidence.get("status") != "METRICS_RECALCULATED_PASS":
    st.error("Validated activity evidence is unavailable.")
    st.stop()
st.caption("IM / JA / TT activity labels | 2,615 sessions | 56 participants | five participant folds")
summary = pd.DataFrame(evidence["summary"])
columns = ["model", "macro_f1_mean", "macro_f1_std", "balanced_accuracy_mean", "accuracy_mean", "macro_ovr_roc_auc_mean"]
st.dataframe(summary[columns].round(4), hide_index=True, width="stretch")
st.bar_chart(summary.set_index("model")[["macro_f1_mean"]], horizontal=True)
st.subheader("Pooled out-of-fold metrics")
st.dataframe(pd.DataFrame(evidence["pooled_metrics"]).round(4), hide_index=True, width="stretch")
st.write("Both learned models improve Macro F1 over the dummy baseline. Random Forest has higher Macro F1 in this run; LR has higher balanced accuracy. No significance claim is made.")
model = st.selectbox("Pooled confusion matrix", summary["model"].tolist())
labels = evidence["labels"]
st.caption("Rows: true activity. Columns: predicted activity.")
st.dataframe(pd.DataFrame(evidence["pooled_confusion_matrices"][model], index=labels, columns=labels), width="stretch")
st.info("All participants have ASD. This page reports therapy-activity recognition, not diagnosis, severity or treatment effectiveness. It does not externally validate the AV-ASD or ASD-FEAT classifiers.")
st.caption("Team: Muhammad Danial, Hamza Rasheed, Muhammad Freeh")
