"""Notebook-derived standalone stage. Source cells: [49]."""
from script_runtime import run_stage


def run(PROJECT, original_videos=None):
    from pathlib import Path
    import json
    import joblib
    import numpy as np
    import pandas as pd

    from sklearn.metrics import (
        f1_score, precision_score, recall_score,
        hamming_loss, accuracy_score, classification_report
    )


    INPUT_DIR = PROJECT / "dataset" / "model_inputs"
    FEATURE_DIR = PROJECT / "dataset" / "features"
    MODEL_DIR = PROJECT / "models" / "baselines"
    RESULT_DIR = PROJECT / "results" / "baselines"

    # Record model selection before evaluating the test set
    model_files = {
        "audio_lr": "audio_logistic_regression_tuned.joblib",
        "visual_lr": "visual_logistic_regression_tuned.joblib",
        "combined_lr": "combined_logistic_regression_tuned.joblib",
        "combined_rf": "combined_random_forest_tuned.joblib",
        "late_fusion_lr": "late_fusion_logistic_tuned.joblib"
    }

    bundles = {
        name: joblib.load(MODEL_DIR / filename)
        for name, filename in model_files.items()
    }

    frozen_config = {
        "primary_model": "combined_lr",
        "selection_metric": "Validation Macro F1",
        "training_split": "train",
        "thresholds": {
            name: float(bundle["threshold"])
            for name, bundle in bundles.items()
        }
    }

    config_path = RESULT_DIR / "frozen_baseline_config.json"
    if config_path.exists():
        with config_path.open(encoding="utf-8") as file:
            assert json.load(file) == frozen_config, (
                "The frozen configuration has changed. Review before evaluation."
            )
    else:
        with config_path.open("w", encoding="utf-8") as file:
            json.dump(frozen_config, file, indent=2)

    # Load and verify the held-out test data
    labels = pd.read_csv(INPUT_DIR / "test_y.csv")
    target_cols = bundles["combined_lr"]["target_columns"]

    assert labels["Video_ID"].is_unique
    assert labels.columns.drop("Video_ID").tolist() == target_cols
    assert labels[target_cols].isin([0, 1]).all().all()

    test_tables = {
        modality: pd.read_csv(INPUT_DIR / f"test_{modality}_X.csv")
        for modality in ["audio", "visual", "combined"]
    }

    for table in test_tables.values():
        assert table["Video_ID"].tolist() == labels["Video_ID"].tolist()

    # Confirm that test clips and source videos are separate
    test_ids = set(labels["Video_ID"])
    test_sources = set(labels["Video_ID"].str.rsplit("_", n=2).str[0])

    for split in ["train", "val"]:
        other = pd.read_csv(INPUT_DIR / f"{split}_y.csv")
        assert not test_ids.intersection(other["Video_ID"])
        other_sources = set(
            other["Video_ID"].str.rsplit("_", n=2).str[0]
        )
        assert not test_sources.intersection(other_sources)

    y_test = labels[target_cols].to_numpy(dtype=int)

    # Preserve the silent-audio flag for subgroup reporting
    audio_report = pd.read_csv(
        FEATURE_DIR / "audio_extraction_report.csv"
    )
    assert audio_report["Video_ID"].is_unique
    assert test_ids.issubset(set(audio_report["Video_ID"]))

    silent_ids = set(audio_report.loc[
        audio_report["all_zero_audio"].astype(str).str.lower().eq("true"),
        "Video_ID"
    ])
    silent_mask = labels["Video_ID"].isin(silent_ids).to_numpy()


    def component_probabilities(bundle, modality):
        """Predict using the saved feature order and fitted model."""
        assert bundle["target_columns"] == target_cols
        columns = bundle["feature_columns"]
        table = test_tables[modality]

        assert table.columns.drop("Video_ID").tolist() == columns
        X = table[columns]
        assert np.isfinite(X.to_numpy(dtype=float)).all()

        return bundle["model"].predict_proba(X)


    # Evaluate all frozen baselines without retraining
    metrics = []
    subgroups = []

    for name, bundle in bundles.items():
        if name == "late_fusion_lr":
            pa = component_probabilities(bundle["audio_bundle"], "audio")
            pv = component_probabilities(bundle["visual_bundle"], "visual")
            probabilities = (
                bundle["audio_weight"] * pa
                + bundle["visual_weight"] * pv
            )
        else:
            modality = (
                "combined" if name.startswith("combined")
                else name.split("_")[0]
            )
            probabilities = component_probabilities(bundle, modality)

        assert probabilities.shape == y_test.shape
        assert np.isfinite(probabilities).all()

        threshold = bundle["threshold"]
        predictions = (probabilities >= threshold).astype(int)

        metrics.append({
            "model": name,
            "primary": name == "combined_lr",
            "threshold": threshold,
            "macro_f1": f1_score(
                y_test, predictions, average="macro", zero_division=0
            ),
            "micro_f1": f1_score(
                y_test, predictions, average="micro", zero_division=0
            ),
            "precision": precision_score(
                y_test, predictions, average="micro", zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, average="micro", zero_division=0
            ),
            "hamming_loss": hamming_loss(y_test, predictions),
            "exact_match_accuracy": accuracy_score(y_test, predictions)
        })

        # Save per-label test results and predictions
        report = classification_report(
            y_test, predictions,
            target_names=target_cols,
            output_dict=True, zero_division=0
        )
        pd.DataFrame(report).T.to_csv(
            RESULT_DIR / f"{name}_test_report.csv"
        )

        for suffix, values in [
            ("probabilities", probabilities),
            ("predictions", predictions)
        ]:
            output = pd.DataFrame(values, columns=target_cols)
            output.insert(0, "Video_ID", labels["Video_ID"].to_numpy())
            output.to_csv(
                RESULT_DIR / f"{name}_test_{suffix}.csv", index=False
            )

        # Save descriptive silence subgroup results
        for group, mask in [
            ("non_silent", ~silent_mask), ("silent", silent_mask)
        ]:
            if mask.any():
                subgroups.append({
                    "model": name,
                    "group": group,
                    "clips": int(mask.sum()),
                    "micro_f1": f1_score(
                        y_test[mask], predictions[mask],
                        average="micro", zero_division=0
                    )
                })

    comparison = pd.DataFrame(metrics)
    comparison.to_csv(
        RESULT_DIR / "frozen_baseline_test_comparison.csv", index=False
    )
    pd.DataFrame(subgroups).to_csv(
        RESULT_DIR / "test_silence_groups.csv", index=False
    )

    print("Primary baseline: combined_lr")
    print("Test clips:", len(labels))
    print("Silent test clips:", int(silent_mask.sum()))
    print("\nFrozen baseline test results:")
    print(comparison.round(4).to_string(index=False))
    print("\nTest reports saved:", RESULT_DIR)


if __name__ == "__main__":
    run_stage(
        __file__, run,
        requires=['dataset/model_inputs/train_y.csv', 'dataset/model_inputs/val_y.csv', 'dataset/model_inputs/test_y.csv', 'dataset/features/audio_extraction_report.csv', 'dataset/model_inputs/test_audio_X.csv', 'dataset/model_inputs/test_visual_X.csv', 'dataset/model_inputs/test_combined_X.csv', 'models/baselines/audio_logistic_regression_tuned.joblib', 'models/baselines/visual_logistic_regression_tuned.joblib', 'models/baselines/combined_logistic_regression_tuned.joblib', 'models/baselines/combined_random_forest_tuned.joblib', 'models/baselines/late_fusion_logistic_tuned.joblib'],
        produces=['results/baselines/*_test_*.csv', 'results/baselines/test_silence_groups.csv'],
        external=(),
        originals=False,
    )
