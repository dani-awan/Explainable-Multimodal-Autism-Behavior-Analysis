"""Notebook-derived standalone stage. Source cells: [70]."""
from script_runtime import run_stage


def run(PROJECT, original_videos=None):
    from pathlib import Path
    import numpy as np
    import pandas as pd


    OUTPUT = PROJECT / "results" / "embedding_grouped_cv"

    # Load the training targets.
    targets = pd.read_csv(
        PROJECT / "dataset" / "model_inputs" / "train_y.csv"
    )

    assert targets["Video_ID"].notna().all()
    assert targets["Video_ID"].is_unique

    labels = targets.columns.drop("Video_ID").tolist()
    ids = targets["Video_ID"].tolist()
    y = targets[labels].to_numpy(dtype=int)

    assert np.isin(y, [0, 1]).all()

    # Use out-of-fold predictions from the same training folds.
    files = {
        "audio": PROJECT / "results/grouped_cv/audio_oof_probabilities.csv",
        "combined": PROJECT / "results/grouped_cv/combined_oof_probabilities.csv",
        "embedding_lr": OUTPUT / "embedding_lr_oof_probabilities.csv",
        "audio_embedding_lr": OUTPUT / "audio_embedding_lr_oof_probabilities.csv",
    }

    summary_rows = []
    label_rows = []

    for model_name, path in files.items():
        probabilities = pd.read_csv(path)

        assert probabilities["Video_ID"].notna().all()
        assert probabilities["Video_ID"].is_unique
        assert set(probabilities["Video_ID"]) == set(ids)

        # Align predictions with the target rows.
        probabilities = probabilities.set_index("Video_ID").loc[ids, labels]
        p = probabilities.to_numpy(dtype=float)

        assert np.isfinite(p).all()
        assert ((p >= 0) & (p <= 1)).all()

        predictions = (p >= 0.5).astype(int)

        tp = ((y == 1) & (predictions == 1)).sum(axis=0)
        fp = ((y == 0) & (predictions == 1)).sum(axis=0)
        fn = ((y == 1) & (predictions == 0)).sum(axis=0)

        summary_rows.append({
            "model": model_name,
            "actual_positive_labels": int(y.sum()),
            "predicted_positive_labels": int(predictions.sum()),
            "actual_labels_per_clip": y.sum(axis=1).mean(),
            "predicted_labels_per_clip": predictions.sum(axis=1).mean(),
            "true_positives": int(tp.sum()),
            "false_positives": int(fp.sum()),
            "false_negatives": int(fn.sum()),
            "clips_with_no_predictions": int(
                (predictions.sum(axis=1) == 0).sum()
            ),
            "hamming_loss": float((predictions != y).mean()),
        })

        for index, label in enumerate(labels):
            label_rows.append({
                "model": model_name,
                "label": label,
                "support": int(y[:, index].sum()),
                "predicted_positive": int(predictions[:, index].sum()),
                "TP": int(tp[index]),
                "FP": int(fp[index]),
                "FN": int(fn[index]),
            })

    summary = pd.DataFrame(summary_rows)
    label_details = pd.DataFrame(label_rows)

    # Compare with an all-negative reference prediction.
    all_negative_loss = float(y.mean())

    print("Training out-of-fold prediction diagnostics:")
    print(summary.round(4).to_string(index=False))
    print(
        f"\nAll-negative reference Hamming loss: "
        f"{all_negative_loss:.4f}"
    )

    summary.to_csv(OUTPUT / "oof_prediction_diagnostics.csv", index=False)
    label_details.to_csv(OUTPUT / "oof_label_diagnostics.csv", index=False)

    print("\nDiagnostic reports saved successfully.")


if __name__ == "__main__":
    run_stage(
        __file__, run,
        requires=['dataset/model_inputs/train_y.csv', 'results/grouped_cv/audio_oof_probabilities.csv', 'results/grouped_cv/combined_oof_probabilities.csv', 'results/embedding_grouped_cv/embedding_lr_oof_probabilities.csv', 'results/embedding_grouped_cv/audio_embedding_lr_oof_probabilities.csv'],
        produces=['results/embedding_grouped_cv/oof_*diagnostics.csv'],
        external=(),
        originals=False,
    )
