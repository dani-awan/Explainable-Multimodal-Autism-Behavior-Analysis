"""Notebook-derived standalone stage. Source cells: [56]."""
from script_runtime import run_stage


def run(PROJECT, original_videos=None):
    from pathlib import Path

    import numpy as np
    import pandas as pd


    RESULTS = PROJECT / "results" / "baselines"
    OUTPUT = PROJECT / "results" / "error_analysis"
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Load the frozen primary-model predictions and test annotations.
    predictions = pd.read_csv(
        RESULTS / "combined_lr_test_predictions.csv"
    )
    targets = pd.read_csv(
        PROJECT / "dataset" / "model_inputs" / "test_y.csv"
    )

    # Validate IDs before aligning the tables.
    for name, table in [("Predictions", predictions), ("Targets", targets)]:
        if "Video_ID" not in table.columns:
            raise ValueError(f"{name}: Video_ID column is missing.")

        if table["Video_ID"].isna().any():
            raise ValueError(f"{name}: Missing clip IDs.")

        if table["Video_ID"].duplicated().any():
            raise ValueError(f"{name}: Duplicate clip IDs.")

    labels = [column for column in targets.columns if column != "Video_ID"]

    if set(predictions["Video_ID"]) != set(targets["Video_ID"]):
        raise ValueError("Prediction and annotation clip IDs differ.")

    if not set(labels).issubset(predictions.columns):
        raise ValueError("Some prediction label columns are missing.")

    predictions = predictions.set_index("Video_ID").loc[
        targets["Video_ID"], labels
    ]
    targets = targets.set_index("Video_ID")[labels]

    if not predictions.isin([0, 1]).all().all():
        raise ValueError("Predictions must contain only binary values.")

    if not targets.isin([0, 1]).all().all():
        raise ValueError("Targets must contain only binary values.")

    y_true = targets.to_numpy(dtype=int)
    y_pred = predictions.to_numpy(dtype=int)

    # Calculate error counts for each label.
    true_positive = ((y_true == 1) & (y_pred == 1)).sum(axis=0)
    false_positive = ((y_true == 0) & (y_pred == 1)).sum(axis=0)
    false_negative = ((y_true == 1) & (y_pred == 0)).sum(axis=0)
    true_negative = ((y_true == 0) & (y_pred == 0)).sum(axis=0)

    def safe_divide(numerator, denominator):
        """Return zero when the denominator is zero."""
        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=float),
            where=denominator != 0,
        )

    per_label = pd.DataFrame({
        "Label": labels,
        "Support": y_true.sum(axis=0),
        "Predicted_positive": y_pred.sum(axis=0),
        "TP": true_positive,
        "FP": false_positive,
        "FN": false_negative,
        "TN": true_negative,
        "Precision": safe_divide(
            true_positive, true_positive + false_positive
        ),
        "Recall": safe_divide(
            true_positive, true_positive + false_negative
        ),
        "F1": safe_divide(
            2 * true_positive,
            2 * true_positive + false_positive + false_negative,
        ),
    })

    per_label["Total_errors"] = per_label["FP"] + per_label["FN"]
    per_label = per_label.sort_values(
        ["Total_errors", "Label"], ascending=[False, True]
    )

    # Describe errors for each individual clip.
    clip_rows = []

    for row_number, clip_id in enumerate(targets.index):
        actual = y_true[row_number]
        predicted = y_pred[row_number]

        extra_labels = [
            label for index, label in enumerate(labels)
            if predicted[index] == 1 and actual[index] == 0
        ]
        missed_labels = [
            label for index, label in enumerate(labels)
            if predicted[index] == 0 and actual[index] == 1
        ]

        clip_rows.append({
            "Video_ID": clip_id,
            "Annotated_label_count": int(actual.sum()),
            "Predicted_label_count": int(predicted.sum()),
            "False_positive_count": len(extra_labels),
            "False_negative_count": len(missed_labels),
            "Total_errors": len(extra_labels) + len(missed_labels),
            "Extra_predictions": "; ".join(extra_labels),
            "Missed_annotations": "; ".join(missed_labels),
            "Exact_match": bool(np.array_equal(actual, predicted)),
        })

    per_clip = pd.DataFrame(clip_rows).sort_values(
        ["Total_errors", "Video_ID"], ascending=[False, True]
    )

    # Save descriptive reports without changing the model or threshold.
    label_path = OUTPUT / "combined_lr_test_label_errors.csv"
    clip_path = OUTPUT / "combined_lr_test_clip_errors.csv"

    per_label.to_csv(label_path, index=False)
    per_clip.to_csv(clip_path, index=False)

    print("Test clips:", len(targets))
    print("Total false positives:", int(false_positive.sum()))
    print("Total false negatives:", int(false_negative.sum()))
    print("Exact-match clips:", int(per_clip["Exact_match"].sum()))

    print("\nErrors by label:")
    print(per_label.round(4).to_string(index=False))

    print("\nTop 10 clips by error count:")
    print(
        per_clip[
            ["Video_ID", "False_positive_count",
             "False_negative_count", "Total_errors"]
        ].head(10).to_string(index=False)
    )

    print("\nSaved:", label_path)
    print("Saved:", clip_path)


if __name__ == "__main__":
    run_stage(
        __file__, run,
        requires=['results/baselines/combined_lr_test_predictions.csv', 'dataset/model_inputs/test_y.csv'],
        produces=['results/error_analysis/combined_lr_test_*errors.csv'],
        external=(),
        originals=False,
    )
