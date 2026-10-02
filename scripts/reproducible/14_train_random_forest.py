"""Notebook-derived standalone stage. Source cells: [47]."""
from script_runtime import run_stage


def run(PROJECT, original_videos=None):
    from pathlib import Path
    import joblib
    import numpy as np
    import pandas as pd

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.metrics import (
        f1_score, precision_score, recall_score,
        hamming_loss, classification_report
    )


    INPUT_DIR = PROJECT / "dataset" / "model_inputs"
    MODEL_DIR = PROJECT / "models" / "baselines"
    RESULT_DIR = PROJECT / "results" / "baselines"

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # Load only training and validation data
    train_x = pd.read_csv(INPUT_DIR / "train_combined_X.csv")
    val_x = pd.read_csv(INPUT_DIR / "val_combined_X.csv")
    train_y = pd.read_csv(INPUT_DIR / "train_y.csv")
    val_y = pd.read_csv(INPUT_DIR / "val_y.csv")

    assert train_x["Video_ID"].tolist() == train_y["Video_ID"].tolist()
    assert val_x["Video_ID"].tolist() == val_y["Video_ID"].tolist()
    assert train_x.columns.tolist() == val_x.columns.tolist()
    assert train_y.columns.tolist() == val_y.columns.tolist()

    feature_cols = train_x.columns.drop("Video_ID").tolist()
    target_cols = train_y.columns.drop("Video_ID").tolist()

    X_train = train_x[feature_cols]
    X_val = val_x[feature_cols]
    y_train = train_y[target_cols].to_numpy(dtype=int)
    y_val = val_y[target_cols].to_numpy(dtype=int)

    assert np.isfinite(X_train.to_numpy(dtype=float)).all()
    assert np.isfinite(X_val.to_numpy(dtype=float)).all()
    assert train_y[target_cols].isin([0, 1]).all().all()
    assert val_y[target_cols].isin([0, 1]).all().all()

    # Train a separate binary forest for each behavior label
    model = OneVsRestClassifier(
        RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=1
        ),
        n_jobs=1
    )

    print("Training combined Random Forest...")
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_val)

    # Use the same threshold candidates as the logistic baseline
    rows = []

    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        pred = (probabilities >= threshold).astype(int)

        rows.append({
            "threshold": threshold,
            "macro_f1": f1_score(
                y_val, pred, average="macro", zero_division=0
            ),
            "micro_f1": f1_score(
                y_val, pred, average="micro", zero_division=0
            ),
            "precision": precision_score(
                y_val, pred, average="micro", zero_division=0
            ),
            "recall": recall_score(
                y_val, pred, average="micro", zero_division=0
            ),
            "hamming_loss": hamming_loss(y_val, pred)
        })

    best = max(
        rows,
        key=lambda row: (
            row["macro_f1"],
            row["micro_f1"],
            -abs(row["threshold"] - 0.5)
        )
    )

    # Save the model and validation-selected threshold
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_cols,
            "target_columns": target_cols,
            "threshold": best["threshold"],
            "threshold_selection": "Validation Macro F1"
        },
        MODEL_DIR / "combined_random_forest_tuned.joblib"
    )

    pd.DataFrame(rows).to_csv(
        RESULT_DIR / "random_forest_threshold_search.csv", index=False
    )

    probability_table = pd.DataFrame(
        probabilities, columns=target_cols
    )
    probability_table.insert(
        0, "Video_ID", val_y["Video_ID"].to_numpy()
    )
    probability_table.to_csv(
        RESULT_DIR / "random_forest_validation_probabilities.csv",
        index=False
    )

    best_predictions = (
        probabilities >= best["threshold"]
    ).astype(int)

    report = classification_report(
        y_val,
        best_predictions,
        target_names=target_cols,
        output_dict=True,
        zero_division=0
    )
    pd.DataFrame(report).T.to_csv(
        RESULT_DIR / "random_forest_validation_report.csv"
    )

    print("\nRandom Forest validation comparison:")
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    print(
        f"\nSelected threshold: {best['threshold']} | "
        f"Macro F1: {best['macro_f1']:.4f} | "
        f"Micro F1: {best['micro_f1']:.4f}"
    )
    print("Models and reports saved.")


if __name__ == "__main__":
    run_stage(
        __file__, run,
        requires=['dataset/model_inputs/train_y.csv', 'dataset/model_inputs/train_combined_X.csv', 'dataset/model_inputs/val_y.csv', 'dataset/model_inputs/val_combined_X.csv'],
        produces=['models/baselines/combined_random_forest_tuned.joblib', 'results/baselines/random_forest_*.csv'],
        external=(),
        originals=False,
    )
