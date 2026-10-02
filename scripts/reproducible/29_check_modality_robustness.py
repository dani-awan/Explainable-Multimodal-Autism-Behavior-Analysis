"""Notebook-derived standalone stage. Source cells: [77]."""
from script_runtime import run_stage


def run(PROJECT, original_videos=None):
    from pathlib import Path
    import json

    import joblib
    import numpy as np
    import pandas as pd

    from sklearn.metrics import f1_score, hamming_loss
    from sklearn.preprocessing import StandardScaler


    OUTPUT = PROJECT / "results" / "modality_robustness"
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Load the existing primary model trained on this project.
    bundle = joblib.load(
        PROJECT / "models/baselines/combined_logistic_regression_tuned.joblib"
    )

    model = bundle["model"]
    features = list(bundle["feature_columns"])
    labels = list(bundle["target_columns"])
    threshold = float(bundle["threshold"])

    assert np.isclose(threshold, 0.3)


    def load_indexed(path):
        table = pd.read_csv(path)
        if table["Video_ID"].isna().any() or not table["Video_ID"].is_unique:
            raise ValueError(f"Missing or duplicate IDs: {path.name}")
        return table.set_index("Video_ID")


    train = load_indexed(
        PROJECT / "dataset/model_inputs/train_combined_X.csv"
    )
    validation = load_indexed(
        PROJECT / "dataset/model_inputs/val_combined_X.csv"
    )
    targets = load_indexed(
        PROJECT / "dataset/model_inputs/val_y.csv"
    )

    assert set(validation.index) == set(targets.index)
    assert not set(train.index).intersection(validation.index)

    # Confirm source-video separation.
    train_groups = {clip.rsplit("_", 2)[0] for clip in train.index}
    val_groups = {clip.rsplit("_", 2)[0] for clip in validation.index}
    assert not train_groups.intersection(val_groups)

    X = validation.loc[:, features].copy()
    y = targets.loc[X.index, labels].to_numpy(dtype=int)

    assert np.isfinite(X.to_numpy(dtype=float)).all()
    assert np.isin(y, [0, 1]).all()

    # Use the scaler's saved training means.
    scalers = [
        step for _, step in model.steps
        if isinstance(step, StandardScaler)
    ]
    if len(scalers) != 1:
        raise ValueError("Expected exactly one fitted StandardScaler.")

    scaler = scalers[0]
    assert len(scaler.mean_) == len(features)

    if hasattr(scaler, "feature_names_in_"):
        assert list(scaler.feature_names_in_) == features

    training_means = pd.Series(scaler.mean_, index=features)
    assert np.isfinite(training_means.to_numpy()).all()

    visual_columns = [
        column for column in features if column.startswith("visual_")
    ]
    audio_columns = [
        column for column in features if column not in visual_columns
    ]

    assert len(audio_columns) == 44
    assert len(visual_columns) == 14

    conditions = {
        "both_available": [],
        "audio_mean_imputed": audio_columns,
        "visual_mean_imputed": visual_columns,
    }

    rows = []

    print(f"Validation clips: {len(X)}")
    print(f"Frozen threshold: {threshold}")
    print("Running simulated missing-modality checks...")

    for condition, missing_columns in conditions.items():
        changed = X.copy()

        # Replace unavailable features with their training means.
        for column in missing_columns:
            changed[column] = training_means[column]

        probabilities = model.predict_proba(changed)

        assert probabilities.shape == y.shape
        assert np.isfinite(probabilities).all()
        assert ((probabilities >= 0) & (probabilities <= 1)).all()

        predictions = (probabilities >= threshold).astype(int)

        rows.append({
            "condition": condition,
            "clips": len(X),
            "threshold": threshold,
            "macro_f1": f1_score(
                y, predictions, average="macro", zero_division=0
            ),
            "micro_f1": f1_score(
                y, predictions, average="micro", zero_division=0
            ),
            "hamming_loss": hamming_loss(y, predictions),
            "false_positives": int(
                ((predictions == 1) & (y == 0)).sum()
            ),
            "false_negatives": int(
                ((predictions == 0) & (y == 1)).sum()
            ),
            "predicted_labels_per_clip": predictions.sum(axis=1).mean(),
        })

        table = pd.DataFrame(probabilities, columns=labels)
        table.insert(0, "Video_ID", X.index.to_numpy())
        table.to_csv(
            OUTPUT / f"{condition}_validation_probabilities.csv",
            index=False,
        )

    results = pd.DataFrame(rows)
    reference = results.set_index("condition").loc["both_available"]

    # Positive F1 change means improvement over the unmodified inputs.
    for metric in ("macro_f1", "micro_f1", "hamming_loss"):
        results[f"{metric}_change"] = results[metric] - reference[metric]

    results.to_csv(
        OUTPUT / "validation_missing_modality_comparison.csv",
        index=False,
    )

    config = {
        "model": "Frozen combined logistic regression",
        "split": "Validation",
        "threshold": threshold,
        "replacement": "Saved training feature means",
        "retrained": False,
        "interpretation": (
            "Synthetic sensitivity check; not a retrained ablation "
            "or a new independent performance estimate."
        ),
    }
    (OUTPUT / "robustness_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print("\nValidation robustness comparison:")
    print(results.round(4).to_string(index=False))
    print(f"\nSaved in: {OUTPUT}")


if __name__ == "__main__":
    run_stage(
        __file__, run,
        requires=['models/baselines/combined_logistic_regression_tuned.joblib', 'dataset/model_inputs/train_combined_X.csv', 'dataset/model_inputs/val_combined_X.csv', 'dataset/model_inputs/val_y.csv'],
        produces=['results/modality_robustness/*.csv', 'results/modality_robustness/robustness_config.json'],
        external=(),
        originals=False,
    )
