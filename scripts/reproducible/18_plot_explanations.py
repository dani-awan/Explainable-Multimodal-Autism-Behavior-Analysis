"""Notebook-derived standalone stage. Source cells: [51]."""
from script_runtime import run_stage


def run(PROJECT, original_videos=None):
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt


    XAI_DIR = PROJECT / "results" / "explainability"

    local = pd.read_csv(
        XAI_DIR / "example_clip_linear_contributions.csv"
    )
    global_table = pd.read_csv(
        XAI_DIR / "global_linear_contributions.csv"
    )

    # Identify the highest-scoring label for the example clip
    label_scores = local.groupby("label")["probability"].first()
    selected_label = label_scores.idxmax()

    example = local.loc[local["label"].eq(selected_label)].copy()
    example["magnitude"] = example["logit_contribution"].abs()

    top_local = (
        example.nlargest(10, "magnitude")
        .sort_values("logit_contribution")
    )

    # Plot signed contributions to the example prediction
    colors = [
        "#219653" if value >= 0 else "#D9534F"
        for value in top_local["logit_contribution"]
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        top_local["feature"],
        top_local["logit_contribution"],
        color=colors
    )
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_xlabel("Logit contribution relative to training feature means")
    ax.set_title(
        f"Example prediction: {selected_label}\n"
        f"Clip: {example['Video_ID'].iloc[0]} | "
        f"Model score: {example['probability'].iloc[0]:.4f}"
    )
    fig.tight_layout()
    fig.savefig(
        XAI_DIR / "example_prediction_contributions.png",
        dpi=200, bbox_inches="tight"
    )
    # Figures are saved to disk; do not open a blocking window.
    plt.close(fig)

    # Rank features by contribution magnitude across validation clips
    global_ranking = (
        global_table.groupby("feature")["mean_abs_logit_contribution"]
        .mean()
        .nlargest(15)
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        global_ranking.index,
        global_ranking.values,
        color="#2878B5"
    )
    ax.set_xlabel("Mean absolute logit contribution, averaged across labels")
    ax.set_title("Primary baseline: feature contributions on validation clips")
    fig.tight_layout()
    fig.savefig(
        XAI_DIR / "global_feature_contributions.png",
        dpi=200, bbox_inches="tight"
    )
    # Figures are saved to disk; do not open a blocking window.
    plt.close(fig)

    # Compare the example prediction with its annotated target
    labels = pd.read_csv(
        PROJECT / "dataset" / "model_inputs" / "val_y.csv"
    )
    video_id = example["Video_ID"].iloc[0]
    matched = labels.loc[labels["Video_ID"].eq(video_id)]
    assert len(matched) == 1

    truth = int(matched.iloc[0][selected_label])
    predicted = int(
        example["probability"].iloc[0] >= example["threshold"].iloc[0]
    )

    print("Selected label:", selected_label)
    print("Annotated target:", truth)
    print("Predicted target:", predicted)
    print("Prediction matches annotation:", truth == predicted)
    print("\nCharts saved:", XAI_DIR)


if __name__ == "__main__":
    run_stage(
        __file__, run,
        requires=['results/explainability/example_clip_linear_contributions.csv', 'results/explainability/global_linear_contributions.csv', 'dataset/model_inputs/val_y.csv'],
        produces=['results/explainability/*.png'],
        external=(),
        originals=False,
    )
