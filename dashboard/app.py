
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Multimodal Behavior Analysis",
    page_icon="📊",
    layout="wide",
)

# Resolve project folders relative to this app.
PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results" / "baselines"
EXPLANATIONS = PROJECT / "results" / "explainability"
INPUTS = PROJECT / "dataset" / "model_inputs"

st.title("Explainable Multimodal Behavior Analysis")
st.caption(
    "AV-ASD research dashboard | Audio and visual feature baselines"
)
st.info(
    "This dashboard displays saved research results. "
    "Predictions are not a clinical diagnosis, and model scores "
    "are not diagnostic confidence."
)


def read_csv(path):
    """Load a required CSV file."""
    if not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path.name}")
    return pd.read_csv(path)


def index_by_id(table):
    """Validate clip IDs before using them for alignment."""
    if "Video_ID" not in table.columns:
        raise ValueError("A table is missing the Video_ID column.")

    if table["Video_ID"].isna().any():
        raise ValueError("A table contains missing clip IDs.")

    if table["Video_ID"].duplicated().any():
        raise ValueError("A table contains duplicate clip IDs.")

    return table.set_index("Video_ID")


try:
    comparison = read_csv(
        RESULTS / "frozen_baseline_test_comparison.csv"
    )
    probabilities = index_by_id(
        read_csv(RESULTS / "combined_lr_test_probabilities.csv")
    )
    predictions = index_by_id(
        read_csv(RESULTS / "combined_lr_test_predictions.csv")
    )
    targets = index_by_id(read_csv(INPUTS / "test_y.csv"))

    labels = targets.columns.tolist()

    # Align predictions and probabilities by clip ID and label name.
    if set(probabilities.index) != set(targets.index):
        raise ValueError("Probability and annotation clip IDs differ.")

    if set(predictions.index) != set(targets.index):
        raise ValueError("Prediction and annotation clip IDs differ.")

    if not set(labels).issubset(probabilities.columns):
        raise ValueError("Probability label columns are incomplete.")

    if not set(labels).issubset(predictions.columns):
        raise ValueError("Prediction label columns are incomplete.")

    probabilities = probabilities.loc[targets.index, labels]
    predictions = predictions.loc[targets.index, labels]

    probability_values = probabilities.to_numpy(dtype=float)

    if not np.isfinite(probability_values).all():
        raise ValueError("Probabilities contain missing or infinite values.")

    if ((probability_values < 0) | (probability_values > 1)).any():
        raise ValueError("Probability values must be between zero and one.")

    if not targets.isin([0, 1]).all().all():
        raise ValueError("Annotations must contain binary values.")

    if not predictions.isin([0, 1]).all().all():
        raise ValueError("Predictions must contain binary values.")

    primary_rows = comparison.loc[
        comparison["model"] == "combined_lr"
    ]

    if len(primary_rows) != 1:
        raise ValueError("Expected one combined_lr result row.")

    primary = primary_rows.iloc[0]
    threshold = float(primary["threshold"])

    # Verify saved predictions against the frozen threshold.
    expected = (probability_values >= threshold).astype(int)

    if not np.array_equal(
        expected, predictions.to_numpy(dtype=int)
    ):
        raise ValueError(
            "Saved predictions do not match the recorded threshold."
        )

except Exception as error:
    st.error(str(error))
    st.stop()


st.sidebar.header("Experiment")
st.sidebar.write("Primary baseline: Combined Logistic Regression")
st.sidebar.write(f"Frozen threshold: {threshold:.2f}")
st.sidebar.write(f"Test clips: {len(targets)}")
st.sidebar.caption(
    "The primary model and thresholds were selected using validation "
    "results before test evaluation."
)

overview, explorer, explanations = st.tabs(
    ["Baseline Results", "Clip Explorer", "Explainability"]
)

with overview:
    columns = st.columns(4)
    columns[0].metric("Prepared clips", 527)
    columns[1].metric("Training clips", 312)
    columns[2].metric("Validation clips", 108)
    columns[3].metric("Test clips", len(targets))

    st.subheader("Primary baseline: Combined Logistic Regression")
    metrics = st.columns(3)
    metrics[0].metric("Test Macro F1", f"{primary['macro_f1']:.4f}")
    metrics[1].metric("Test Micro F1", f"{primary['micro_f1']:.4f}")
    metrics[2].metric("Test Hamming loss", f"{primary['hamming_loss']:.4f}")

    st.subheader("Frozen test comparison")
    st.dataframe(comparison, hide_index=True, width="stretch")

    st.bar_chart(
        comparison.set_index("model")[["macro_f1", "micro_f1"]],
        stack=False,
        horizontal=True,
        height=350,
    )

    st.caption(
        "Higher F1 is better; lower Hamming loss is better. "
        "F1 scores are not accuracy percentages. "
        "The primary model remains the validation-selected combined_lr."
    )

    with st.expander("Dataset and feature details"):
        st.write(
            "527 prepared clips: 312 training, 108 validation, "
            "and 107 test clips."
        )
        st.write(
            "58 input features: 44 audio features and 14 visual features. "
            "Each clip has 10 binary annotation targets."
        )
        st.write(
            "Clip-ID and original-video overlap between splits: zero. "
            "Participant independence has not been established."
        )
        st.write(
            "Seven clips have confirmed silent audio. "
            "No silent clips occur in the test split."
        )


with explorer:
    st.subheader("Explore saved primary-model predictions")

    clip_id = st.selectbox(
        "Select a test clip",
        targets.index.tolist(),
    )

    detail = pd.DataFrame(
        {
            "Label": labels,
            "Model score": probabilities.loc[clip_id].to_numpy(dtype=float),
            "Prediction": predictions.loc[clip_id].to_numpy(dtype=int),
            "Annotation": targets.loc[clip_id].to_numpy(dtype=int),
        }
    )

    def describe_result(row):
        """Describe the prediction relative to the annotation."""
        predicted = int(row["Prediction"])
        annotated = int(row["Annotation"])

        if predicted == 1 and annotated == 1:
            return "True positive"
        if predicted == 1 and annotated == 0:
            return "False positive"
        if predicted == 0 and annotated == 1:
            return "False negative"
        return "True negative"

    detail["Outcome"] = detail.apply(describe_result, axis=1)
    detail = detail.sort_values("Model score", ascending=False)

    st.caption(f"Frozen prediction threshold: {threshold:.2f}")
    st.dataframe(detail, hide_index=True, width="stretch")
    import matplotlib.pyplot as plt

    # Plot the same scores displayed in the prediction table.
    chart_data = detail.sort_values("Model score", ascending=True)
    scores = chart_data["Model score"].astype(float).to_numpy()
    colors = [
        "#f59e0b" if score >= threshold else "#60a5fa"
        for score in scores
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(chart_data["Label"], scores, color=colors)

    # Show the complete probability scale and frozen threshold.
    ax.set_xlim(0, 1)
    ax.axvline(
        threshold,
        color="#dc2626",
        linestyle="--",
        linewidth=1.5,
        label=f"Prediction threshold: {threshold:.2f}",
    )

    for position, score in enumerate(scores):
        ax.annotate(
            f"{score:.4f}",
            xy=(score, position),
            xytext=(5 if score < 0.9 else -5, 0),
            textcoords="offset points",
            va="center",
            ha="left" if score < 0.9 else "right",
            fontsize=9,
        )

    ax.set_xlabel("Model score")
    ax.set_title("Saved scores for the selected clip")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    predicted_labels = detail.loc[
        detail["Prediction"] == 1, "Label"
    ].tolist()
    annotated_labels = detail.loc[
        detail["Annotation"] == 1, "Label"
    ].tolist()

    st.write(
        "**Predicted labels:** "
        + (", ".join(predicted_labels) if predicted_labels else "None")
    )
    st.write(
        "**Annotated labels:** "
        + (", ".join(annotated_labels) if annotated_labels else "None")
    )

    st.download_button(
        "Download this clip's prediction table",
        data=detail.to_csv(index=False).encode("utf-8"),
        file_name=f"{clip_id}_prediction_review.csv",
        mime="text/csv",
    )


with explanations:
    st.subheader("Saved linear-model explanations")
    st.caption(
        "These charts explain model scores through feature contributions "
        "in logit units. They do not establish causes of behavior."
    )

    # Keep the saved validation example separate from test selections.
    st.write("**Validation example:** `_M8YkVVb5dg_0_16`")
    st.write("**Explained label:** Upper Limb Stereotypies")

    local_chart = EXPLANATIONS / "example_prediction_contributions.png"
    global_chart = EXPLANATIONS / "global_feature_contributions.png"

    if local_chart.is_file():
        st.image(str(local_chart))
    else:
        st.warning("The local explanation chart is missing.")

    st.caption(
        "Positive contributions increase the selected label's logit "
        "relative to the training-mean reference; negative contributions "
        "decrease it. This chart belongs to the validation example above."
    )

    st.subheader("Global feature contributions")

    if global_chart.is_file():
        st.image(str(global_chart))
    else:
        st.warning("The global explanation chart is missing.")

    st.caption(
        "Global contributions summarize the saved validation explanations."
    )
