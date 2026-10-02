"""Notebook-derived standalone stage. Source cells: [46]."""
from script_runtime import run_stage


def run(PROJECT, original_videos=None):
    from pathlib import Path
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.metrics import f1_score, precision_score, recall_score


    INPUT_DIR = PROJECT / "dataset" / "model_inputs"
    FEATURE_DIR = PROJECT / "dataset" / "features"
    MODEL_DIR = PROJECT / "models" / "baselines"
    RESULT_DIR = PROJECT / "results" / "baselines"

    labels = pd.read_csv(INPUT_DIR / "val_y.csv")
    audio_report = pd.read_csv(
        FEATURE_DIR / "audio_extraction_report.csv"
    )

    # Recover the silence flags from the extraction report
    assert audio_report["Video_ID"].is_unique
    assert set(labels["Video_ID"]).issubset(set(audio_report["Video_ID"]))

    silent_ids = set(
        audio_report.loc[
            audio_report["all_zero_audio"].astype(str).str.lower().eq("true"),
            "Video_ID"
        ]
    )
    silent_mask = labels["Video_ID"].isin(silent_ids).to_numpy()

    print("Validation clips:", len(labels))
    print("Silent validation clips:", int(silent_mask.sum()))

    per_label_rows = []
    group_rows = []

    for modality in ["audio", "visual", "combined"]:
        bundle = joblib.load(
            MODEL_DIR / f"{modality}_logistic_regression_tuned.joblib"
        )
        target_cols = bundle["target_columns"]
        threshold = bundle["threshold"]

        probabilities = pd.read_csv(
            RESULT_DIR / f"{modality}_validation_probabilities.csv"
        )
        assert probabilities["Video_ID"].tolist() == labels["Video_ID"].tolist()

        y = labels[target_cols].to_numpy(dtype=int)
        p = probabilities[target_cols].to_numpy(dtype=float)
        assert np.isfinite(p).all()
        pred = (p >= threshold).astype(int)

        # Count correct detections, false alarms, and missed positives
        for index, name in enumerate(target_cols):
            truth = y[:, index]
            prediction = pred[:, index]

            per_label_rows.append({
                "modality": modality,
                "label": name,
                "positive_support": int(truth.sum()),
                "predicted_positive": int(prediction.sum()),
                "TP": int(((truth == 1) & (prediction == 1)).sum()),
                "FP": int(((truth == 0) & (prediction == 1)).sum()),
                "FN": int(((truth == 1) & (prediction == 0)).sum()),
                "precision": precision_score(
                    truth, prediction, zero_division=0
                ),
                "recall": recall_score(
                    truth, prediction, zero_division=0
                ),
                "f1": f1_score(truth, prediction, zero_division=0)
            })

        # Report performance separately for silent and non-silent clips
        for group, mask in [
            ("all", np.ones(len(labels), dtype=bool)),
            ("non_silent", ~silent_mask),
            ("silent", silent_mask)
        ]:
            if not mask.any():
                continue

            group_rows.append({
                "modality": modality,
                "group": group,
                "clips": int(mask.sum()),
                "micro_f1": f1_score(
                    y[mask], pred[mask], average="micro", zero_division=0
                ),
                "macro_f1": f1_score(
                    y[mask], pred[mask], average="macro", zero_division=0
                )
            })

        # Compare against a simple predictor that marks every label positive
        print(
            f"{modality}: "
            f"average predicted labels per clip={pred.sum(axis=1).mean():.2f}"
        )

    # Save the diagnostic tables
    per_label = pd.DataFrame(per_label_rows)
    groups = pd.DataFrame(group_rows)

    per_label.to_csv(
        RESULT_DIR / "tuned_validation_per_label.csv", index=False
    )
    groups.to_csv(
        RESULT_DIR / "tuned_validation_silence_groups.csv", index=False
    )

    print("\nCombined model: per-label results")
    print(
        per_label.loc[per_label["modality"].eq("combined")]
        .drop(columns="modality")
        .round(4)
        .to_string(index=False)
    )

    print("\nSilence group results:")
    print(groups.round(4).to_string(index=False))

    all_positive = np.ones_like(y)
    print(
        "\nAlways-positive baseline Micro F1:",
        round(f1_score(y, all_positive, average="micro", zero_division=0), 4)
    )
    print(
        "Always-positive baseline Macro F1:",
        round(f1_score(y, all_positive, average="macro", zero_division=0), 4)
    )

    print("\nDiagnostic reports saved:", RESULT_DIR)


if __name__ == "__main__":
    run_stage(
        __file__, run,
        requires=['dataset/model_inputs/val_y.csv', 'dataset/features/audio_extraction_report.csv', 'models/baselines/audio_logistic_regression_tuned.joblib', 'models/baselines/visual_logistic_regression_tuned.joblib', 'models/baselines/combined_logistic_regression_tuned.joblib', 'results/baselines/audio_validation_probabilities.csv', 'results/baselines/visual_validation_probabilities.csv', 'results/baselines/combined_validation_probabilities.csv'],
        produces=['results/baselines/tuned_validation_per_label.csv', 'results/baselines/tuned_validation_silence_groups.csv'],
        external=(),
        originals=False,
    )
