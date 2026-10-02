from pathlib import Path
import pandas as pd
import streamlit as st

PROJECT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT / "results/modality_robustness"

st.set_page_config(page_title="Modality Robustness", layout="wide")
st.title("Modality Robustness")
st.caption(
    "Frozen combined logistic regression | Validation split | "
    "Threshold: 0.30 | No retraining"
)

path = RESULTS / "validation_missing_modality_comparison.csv"
if not path.is_file():
    st.error("The robustness comparison report is missing.")
    st.stop()

results = pd.read_csv(path)

names = {
    "both_available": "Both modalities available",
    "audio_mean_imputed": "Audio replaced with training means",
    "visual_mean_imputed": "Visual replaced with training means",
}
display = results.copy()
display["condition"] = display["condition"].replace(names)

st.info(
    "This is a synthetic sensitivity check. Missing feature groups "
    "were replaced with saved training means. It does not establish "
    "performance on real recordings with unavailable modalities."
)

columns = st.columns(3)
columns[0].metric("Validation clips", int(results.iloc[0]["clips"]))
columns[1].metric("Frozen threshold", f'{results.iloc[0]["threshold"]:.2f}')
columns[2].metric("Conditions", len(results))

st.subheader("Validation results")
st.dataframe(display.round(4), hide_index=True)

left, right = st.columns(2)
with left:
    st.markdown("**Macro F1 — higher is better**")
    st.bar_chart(display.set_index("condition")[["macro_f1"]])
with right:
    st.markdown("**Micro F1 — higher is better**")
    st.bar_chart(display.set_index("condition")[["micro_f1"]])

st.subheader("False positives and missed labels")
st.dataframe(
    display[[
        "condition",
        "false_positives",
        "false_negatives",
        "predicted_labels_per_clip",
    ]].round(4),
    hide_index=True,
)

st.caption(
    "Replacing audio reduced false positives but increased missed labels. "
    "Lower Hamming loss alone does not establish an overall improvement. "
    "Both replacement conditions reduced Macro F1 and Micro F1."
)

st.download_button(
    "Download robustness comparison",
    data=results.to_csv(index=False),
    file_name="validation_missing_modality_comparison.csv",
    mime="text/csv",
)

report = PROJECT / "docs/MODALITY_ROBUSTNESS.md"
if report.is_file():
    with st.expander("Experiment notes"):
        st.markdown(report.read_text(encoding="utf-8"))
