"""Notebook-derived standalone stage. Source cells: [50]."""
from script_runtime import run_stage


def run(PROJECT, original_videos=None):
    from pathlib import Path
    import joblib
    import numpy as np
    import pandas as pd


    INPUT_DIR = PROJECT / "dataset" / "model_inputs"
    MODEL_DIR = PROJECT / "models" / "baselines"
    XAI_DIR = PROJECT / "results" / "explainability"
    XAI_DIR.mkdir(parents=True, exist_ok=True)

    # Load the frozen primary model and its feature order
    bundle = joblib.load(
        MODEL_DIR / "combined_logistic_regression_tuned.joblib"
    )
    model = bundle["model"]
    feature_cols = bundle["feature_columns"]
    target_cols = bundle["target_columns"]

    train = pd.read_csv(INPUT_DIR / "train_combined_X.csv")
    validation = pd.read_csv(INPUT_DIR / "val_combined_X.csv")

    # Use training feature means as the explanation reference
    scaler = model.named_steps["scaler"]
    classifier = model.named_steps["classifier"]

    z_train = scaler.transform(train[feature_cols])
    z_val = scaler.transform(validation[feature_cols])
    reference = z_train.mean(axis=0)

    global_rows = []
    local_rows = []

    # Explain the first validation clip without selecting by its outcome
    clip_index = 0
    video_id = validation.iloc[clip_index]["Video_ID"]
    probabilities = model.predict_proba(
        validation[feature_cols].iloc[[clip_index]]
    )[0]

    for label_index, label in enumerate(target_cols):
        estimator = classifier.estimators_[label_index]

        if not hasattr(estimator, "coef_"):
            raise RuntimeError(
                f"{label}: constant classifier cannot use this explanation."
            )

        weights = estimator.coef_[0]
        intercept = float(estimator.intercept_[0])

        # Decompose the logit relative to the training reference
        contributions = (z_val - reference) * weights
        baseline_logit = intercept + float(reference @ weights)

        reconstructed = baseline_logit + contributions.sum(axis=1)
        actual = estimator.decision_function(z_val)
        assert np.allclose(reconstructed, actual)

        # Summarize contribution magnitudes across validation clips
        magnitudes = np.abs(contributions).mean(axis=0)

        for feature, magnitude in zip(feature_cols, magnitudes):
            global_rows.append({
                "label": label,
                "feature": feature,
                "mean_abs_logit_contribution": float(magnitude)
            })

        # Save signed contributions for the example clip
        for feature_index, feature in enumerate(feature_cols):
            local_rows.append({
                "Video_ID": video_id,
                "label": label,
                "feature": feature,
                "feature_value": float(
                    validation.iloc[clip_index][feature]
                ),
                "logit_contribution": float(
                    contributions[clip_index, feature_index]
                ),
                "baseline_logit": baseline_logit,
                "probability": float(probabilities[label_index]),
                "threshold": float(bundle["threshold"])
            })

    global_table = pd.DataFrame(global_rows)
    local_table = pd.DataFrame(local_rows)

    global_table.to_csv(
        XAI_DIR / "global_linear_contributions.csv", index=False
    )
    local_table.to_csv(
        XAI_DIR / "example_clip_linear_contributions.csv", index=False
    )

    # Display the strongest contributions for the highest-scoring label
    label_index = int(np.argmax(probabilities))
    selected_label = target_cols[label_index]

    example = local_table.loc[
        local_table["label"].eq(selected_label)
    ].copy()
    example["absolute_contribution"] = example["logit_contribution"].abs()
    example = example.sort_values(
        "absolute_contribution", ascending=False
    ).head(10)

    print("Example clip:", video_id)
    print("Highest-scoring label:", selected_label)
    print("Model probability:", round(float(probabilities[label_index]), 4))
    print("Prediction threshold:", bundle["threshold"])
    print("\nTop feature contributions:")
    print(
        example[["feature", "logit_contribution"]]
        .round(4)
        .to_string(index=False)
    )
    print("\nLogit reconstruction checks: PASS")
    print("Explanation files saved:", XAI_DIR)


if __name__ == "__main__":
    run_stage(
        __file__, run,
        requires=['models/baselines/combined_logistic_regression_tuned.joblib', 'dataset/model_inputs/train_combined_X.csv', 'dataset/model_inputs/val_combined_X.csv'],
        produces=['results/explainability/*linear_contributions.csv'],
        external=(),
        originals=False,
    )
