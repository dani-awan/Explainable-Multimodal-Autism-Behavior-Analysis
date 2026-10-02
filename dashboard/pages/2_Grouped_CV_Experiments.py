from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT / "results"
PCA_RESULTS = RESULTS / "embedding_pca_grouped_cv"

st.set_page_config(
    page_title="Grouped CV Experiments",
    layout="wide",
)

st.title("Grouped CV Experiments")
st.caption(
    "Training-only experiments using the same five source-video folds "
    "and a fixed prediction threshold of 0.5."
)

paths = {
    "comparison": PCA_RESULTS / "cv_comparison.csv",
    "paired": PCA_RESULTS / "audio_pca_paired_summary.csv",
    "labels": PCA_RESULTS / "audio_pca_label_comparison.csv",
    "diagnostics": (
        RESULTS / "embedding_grouped_cv/oof_prediction_diagnostics.csv"
    ),
}

missing = [path.name for path in paths.values() if not path.is_file()]
if missing:
    st.error("Required reports are missing: " + ", ".join(missing))
    st.stop()

# Read the saved results on each rerun.
comparison = pd.read_csv(paths["comparison"])
paired = pd.read_csv(paths["paired"])
labels = pd.read_csv(paths["labels"])
diagnostics = pd.read_csv(paths["diagnostics"])

columns = st.columns(4)
columns[0].metric("Training clips", 312)
columns[1].metric("Source-video groups", 185)
columns[2].metric("Evaluation folds", 5)
columns[3].metric("Prediction threshold", "0.5")

st.info(
    "PCA improved Micro F1 over audio-only in 4 of 5 folds. "
    "Macro F1 improved in only 1 of 5 folds. "
    "The frozen primary baseline remains unchanged."
)

tabs = st.tabs([
    "Model comparison",
    "Paired folds",
    "Behavior labels",
    "Diagnostics",
    "Progress report",
])

with tabs[0]:
    st.subheader("Mean scores across five folds")

    available_models = comparison["model"].tolist()
    selected_models = st.multiselect(
        "Models to compare",
        options=available_models,
        default=available_models,
    )

    selected = comparison[
        comparison["model"].isin(selected_models)
    ].copy()

    if selected.empty:
        st.info("Select at least one model.")
    else:
        st.dataframe(selected.round(4), hide_index=True)

        left, right = st.columns(2)
        with left:
            st.markdown("**Macro F1 — higher is better**")
            st.bar_chart(
                selected.set_index("model")[["macro_f1_mean"]]
            )
        with right:
            st.markdown("**Micro F1 — higher is better**")
            st.bar_chart(
                selected.set_index("model")[["micro_f1_mean"]]
            )

    st.caption(
        "Standard deviations describe variation across folds. "
        "They are not confidence intervals."
    )
    st.download_button(
        "Download full CV comparison",
        data=comparison.to_csv(index=False),
        file_name="grouped_cv_comparison.csv",
        mime="text/csv",
    )

with tabs[1]:
    st.subheader("Audio + PCA embeddings versus audio-only")
    st.dataframe(paired.round(4), hide_index=True)
    st.caption(
        "Positive benefit means the PCA model performed better. "
        "Fold counts are descriptive, not significance tests."
    )

with tabs[2]:
    st.subheader("Per-label out-of-fold results")
    st.dataframe(
        labels.sort_values("f1_difference", ascending=False).round(4),
        hide_index=True,
    )

    selected_label = st.selectbox(
        "Inspect a behavior label",
        options=labels["label"].tolist(),
    )
    row = labels.set_index("label").loc[selected_label]

    left, middle, right = st.columns(3)
    left.metric("Positive training examples", int(row["support"]))
    middle.metric("Audio-only F1", f'{row["audio_f1"]:.4f}')
    right.metric(
        "Audio + PCA F1",
        f'{row["pca_f1"]:.4f}',
        delta=f'{row["f1_difference"]:+.4f}',
    )

    st.caption(
        "These scores pool all training out-of-fold predictions. "
        "Their average can differ from mean per-fold Macro F1."
    )

with tabs[3]:
    st.subheader("Raw embedding prediction diagnostics")
    st.dataframe(diagnostics.round(4), hide_index=True)
    st.caption(
        "This table covers the original models before PCA. "
        "The all-negative reference Hamming loss is approximately "
        "0.1615. Hamming loss alone does not establish useful detection."
    )

with tabs[4]:
    report_path = PROJECT / "docs/VISUAL_EXPERIMENT_PROGRESS.md"
    if report_path.is_file():
        report = report_path.read_text(encoding="utf-8")
        st.markdown(report)
        st.download_button(
            "Download experiment report",
            data=report,
            file_name="VISUAL_EXPERIMENT_PROGRESS.md",
            mime="text/markdown",
        )
    else:
        st.info("The progress report is not available.")

st.caption(
    "Exploratory model-development results. Source-video grouping "
    "does not establish participant independence."
)
