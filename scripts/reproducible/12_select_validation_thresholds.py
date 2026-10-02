"""Notebook-derived standalone stage. Source cells: [45]."""
from script_runtime import run_stage


def run(PROJECT, original_videos=None):
    from pathlib import Path
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.metrics import f1_score, precision_score, recall_score


    INPUT_DIR = PROJECT / "dataset" / "model_inputs"
    MODEL_DIR = PROJECT / "models" / "baselines"
    RESULT_DIR = PROJECT / "results" / "baselines"

    labels = pd.read_csv(INPUT_DIR / "val_y.csv")
    search_rows = []

    # Compare one global threshold per model
    for modality in ["audio", "visual", "combined"]:
        bundle = joblib.load(
            MODEL_DIR / f"{modality}_logistic_regression.joblib"
        )
        target_cols = bundle["target_columns"]

        probabilities = pd.read_csv(
            RESULT_DIR / f"{modality}_validation_probabilities.csv"
        )

        assert probabilities["Video_ID"].tolist() == labels["Video_ID"].tolist()

        y = labels[target_cols].to_numpy(dtype=int)
        p = probabilities[target_cols].to_numpy(dtype=float)
        assert np.isfinite(p).all()

        candidates = []

        for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            predictions = (p >= threshold).astype(int)

            row = {
                "modality": modality,
                "threshold": threshold,
                "macro_f1": f1_score(
                    y, predictions, average="macro", zero_division=0
                ),
                "micro_f1": f1_score(
                    y, predictions, average="micro", zero_division=0
                ),
                "precision": precision_score(
                    y, predictions, average="micro", zero_division=0
                ),
                "recall": recall_score(
                    y, predictions, average="micro", zero_division=0
                )
            }
            candidates.append(row)
            search_rows.append(row)

        # Select by Macro F1, then Micro F1, then proximity to 0.5
        best = max(
            candidates,
            key=lambda row: (
                row["macro_f1"],
                row["micro_f1"],
                -abs(row["threshold"] - 0.5)
            )
        )

        # Save a separate tuned model and retain the original baseline
        bundle["threshold"] = best["threshold"]
        bundle["threshold_selection"] = "Validation Macro F1"
        joblib.dump(
            bundle,
            MODEL_DIR / f"{modality}_logistic_regression_tuned.joblib"
        )

        print(
            f"{modality}: threshold={best['threshold']} | "
            f"Macro F1={best['macro_f1']:.4f} | "
            f"Micro F1={best['micro_f1']:.4f}"
        )

    search = pd.DataFrame(search_rows)
    search.to_csv(
        RESULT_DIR / "validation_threshold_search.csv", index=False
    )

    print("\nThreshold comparison:")
    print(search.round(4).to_string(index=False))


if __name__ == "__main__":
    run_stage(
        __file__, run,
        requires=['dataset/model_inputs/val_y.csv', 'models/baselines/audio_logistic_regression.joblib', 'models/baselines/visual_logistic_regression.joblib', 'models/baselines/combined_logistic_regression.joblib', 'results/baselines/audio_validation_probabilities.csv', 'results/baselines/visual_validation_probabilities.csv', 'results/baselines/combined_validation_probabilities.csv'],
        produces=['models/baselines/*_logistic_regression_tuned.joblib', 'results/baselines/validation_threshold_search.csv'],
        external=(),
        originals=False,
    )
