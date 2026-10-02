"""Notebook-derived standalone stage. Source cells: [44]."""
from script_runtime import run_stage


def run(PROJECT, original_videos=None):
    from pathlib import Path
    import warnings
    import joblib
    import numpy as np
    import pandas as pd

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.metrics import (
        f1_score,
        precision_score,
        recall_score,
        hamming_loss,
        accuracy_score,
        classification_report
    )

    # Define the input and output folders

    INPUT_DIR = PROJECT / "dataset" / "model_inputs"
    MODEL_DIR = PROJECT / "models" / "baselines"
    RESULT_DIR = PROJECT / "results" / "baselines"

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the training and validation targets
    train_targets = pd.read_csv(INPUT_DIR / "train_y.csv")
    val_targets = pd.read_csv(INPUT_DIR / "val_y.csv")

    target_cols = train_targets.columns.drop("Video_ID").tolist()

    assert val_targets.columns.tolist() == train_targets.columns.tolist()
    assert train_targets["Video_ID"].is_unique
    assert val_targets["Video_ID"].is_unique
    assert train_targets[target_cols].isin([0, 1]).all().all()
    assert val_targets[target_cols].isin([0, 1]).all().all()

    y_train = train_targets[target_cols].to_numpy(dtype=int)
    y_val = val_targets[target_cols].to_numpy(dtype=int)

    # Display the positive training examples for each target
    print("Training positive counts:")
    print(train_targets[target_cols].sum().to_string())

    # Flag any target that cannot be learned from both classes
    constant_targets = [
        name for name in target_cols
        if train_targets[name].nunique() < 2
    ]
    if constant_targets:
        print("\nConstant training targets:", constant_targets)

    metrics = []

    for modality in ["audio", "visual", "combined"]:
        train_features = pd.read_csv(INPUT_DIR / f"train_{modality}_X.csv")
        val_features = pd.read_csv(INPUT_DIR / f"val_{modality}_X.csv")

        # Verify the feature-to-label row alignment
        assert train_features["Video_ID"].tolist() == (
            train_targets["Video_ID"].tolist()
        )
        assert val_features["Video_ID"].tolist() == (
            val_targets["Video_ID"].tolist()
        )

        feature_cols = train_features.columns.drop("Video_ID").tolist()
        assert val_features.columns.drop("Video_ID").tolist() == feature_cols

        X_train = train_features[feature_cols]
        X_val = val_features[feature_cols]

        assert np.isfinite(X_train.to_numpy(dtype=float)).all()
        assert np.isfinite(X_val.to_numpy(dtype=float)).all()

        # Fit the scaler and classifiers using training data only
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", OneVsRestClassifier(
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=3000,
                    random_state=42
                ),
                n_jobs=1
            ))
        ])

        print(f"\nTraining: {modality}")

        # Stop if optimization does not converge
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(X_train, y_train)

        # Use a fixed initial threshold for the validation comparison
        probabilities = model.predict_proba(X_val)
        predictions = (probabilities >= 0.5).astype(int)

        row = {
            "modality": modality,
            "features": len(feature_cols),
            "threshold": 0.5,
            "micro_f1": f1_score(
                y_val, predictions, average="micro", zero_division=0
            ),
            "macro_f1": f1_score(
                y_val, predictions, average="macro", zero_division=0
            ),
            "micro_precision": precision_score(
                y_val, predictions, average="micro", zero_division=0
            ),
            "micro_recall": recall_score(
                y_val, predictions, average="micro", zero_division=0
            ),
            "hamming_loss": hamming_loss(y_val, predictions),
            "exact_match_accuracy": accuracy_score(y_val, predictions)
        }
        metrics.append(row)

        # Save the model together with its feature and target order
        joblib.dump(
            {
                "model": model,
                "feature_columns": feature_cols,
                "target_columns": target_cols,
                "threshold": 0.5
            },
            MODEL_DIR / f"{modality}_logistic_regression.joblib"
        )

        # Save per-label validation results
        per_label = classification_report(
            y_val,
            predictions,
            target_names=target_cols,
            output_dict=True,
            zero_division=0
        )
        pd.DataFrame(per_label).T.to_csv(
            RESULT_DIR / f"{modality}_validation_report.csv"
        )

        # Preserve validation probabilities for later analysis
        probability_table = pd.DataFrame(
            probabilities, columns=target_cols
        )
        probability_table.insert(
            0, "Video_ID", val_targets["Video_ID"].to_numpy()
        )
        probability_table.to_csv(
            RESULT_DIR / f"{modality}_validation_probabilities.csv",
            index=False
        )

        # Save the comparison after each completed model
        pd.DataFrame(metrics).to_csv(
            RESULT_DIR / "validation_baseline_comparison.csv",
            index=False
        )

        print(
            f"Micro F1: {row['micro_f1']:.4f} | "
            f"Macro F1: {row['macro_f1']:.4f}"
        )

    comparison = pd.DataFrame(metrics)

    print("\nValidation comparison:")
    print(comparison.round(4).to_string(index=False))
    print("\nModels saved:", MODEL_DIR)
    print("Results saved:", RESULT_DIR)


if __name__ == "__main__":
    run_stage(
        __file__, run,
        requires=['dataset/model_inputs/train_y.csv', 'dataset/model_inputs/train_audio_X.csv', 'dataset/model_inputs/train_visual_X.csv', 'dataset/model_inputs/train_combined_X.csv', 'dataset/model_inputs/val_y.csv', 'dataset/model_inputs/val_audio_X.csv', 'dataset/model_inputs/val_visual_X.csv', 'dataset/model_inputs/val_combined_X.csv'],
        produces=['models/baselines/*_logistic_regression.joblib', 'results/baselines/validation_baseline_comparison.csv', 'results/baselines/*_validation_probabilities.csv', 'results/baselines/*_validation_report.csv'],
        external=(),
        originals=False,
    )
