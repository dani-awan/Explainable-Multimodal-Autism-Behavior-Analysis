"""Notebook-derived standalone stage. Source cells: [72]."""
from script_runtime import run_stage


def run(PROJECT, original_videos=None):
    from pathlib import Path
    import numpy as np
    import pandas as pd


    OUTPUT = PROJECT / "results" / "embedding_pca_grouped_cv"
    THRESHOLD = 0.5
    PCA_MODEL = "audio_embedding_pca32_lr"

    # Load training targets.
    targets = pd.read_csv(
        PROJECT / "dataset/model_inputs/train_y.csv"
    )
    assert targets["Video_ID"].notna().all()
    assert targets["Video_ID"].is_unique

    ids = targets["Video_ID"].tolist()
    labels = targets.columns.drop("Video_ID").tolist()
    y = targets[labels].to_numpy(dtype=int)
    assert np.isin(y, [0, 1]).all()

    prediction_files = {
        "audio": PROJECT / "results/grouped_cv/audio_oof_probabilities.csv",
        PCA_MODEL: OUTPUT / f"{PCA_MODEL}_oof_probabilities.csv",
    }

    label_rows = []

    for model_name, path in prediction_files.items():
        table = pd.read_csv(path)

        assert table["Video_ID"].notna().all()
        assert table["Video_ID"].is_unique
        assert set(table["Video_ID"]) == set(ids)

        # Align out-of-fold probabilities with the target rows.
        probabilities = (
            table.set_index("Video_ID").loc[ids, labels]
            .to_numpy(dtype=float)
        )

        assert np.isfinite(probabilities).all()
        assert ((probabilities >= 0) & (probabilities <= 1)).all()

        predictions = (probabilities >= THRESHOLD).astype(int)

        for index, label in enumerate(labels):
            actual = y[:, index]
            predicted = predictions[:, index]

            tp = int(((actual == 1) & (predicted == 1)).sum())
            fp = int(((actual == 0) & (predicted == 1)).sum())
            fn = int(((actual == 1) & (predicted == 0)).sum())

            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0

            label_rows.append({
                "model": model_name,
                "label": label,
                "support": int(actual.sum()),
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            })

    details = pd.DataFrame(label_rows)

    audio_labels = details[details["model"] == "audio"].set_index("label")
    pca_labels = details[details["model"] == PCA_MODEL].set_index("label")

    label_comparison = pd.DataFrame({
        "support": audio_labels["support"],
        "audio_f1": audio_labels["f1"],
        "pca_f1": pca_labels["f1"],
        "f1_difference": pca_labels["f1"] - audio_labels["f1"],
        "audio_precision": audio_labels["precision"],
        "pca_precision": pca_labels["precision"],
        "audio_recall": audio_labels["recall"],
        "pca_recall": pca_labels["recall"],
    }).reset_index()

    # Compare scores from matching evaluation folds.
    baseline_folds = pd.read_csv(
        PROJECT / "results/grouped_cv/fold_results.csv"
    )
    audio_folds = baseline_folds[
        baseline_folds["model"] == "audio"
    ].copy()

    pca_folds = pd.read_csv(OUTPUT / "fold_results.csv")
    pca_folds = pca_folds[pca_folds["model"] == PCA_MODEL].copy()

    for table in (audio_folds, pca_folds):
        assert table["fold"].is_unique
        assert set(table["fold"]) == {1, 2, 3, 4, 5}
        assert np.allclose(table["threshold"], THRESHOLD)
        assert (table["source_overlap"] == 0).all()

    paired = audio_folds.merge(
        pca_folds,
        on="fold",
        suffixes=("_audio", "_pca"),
        validate="one_to_one",
    ).sort_values("fold")

    for column in ("fit_clips", "evaluation_clips"):
        assert (
            paired[f"{column}_audio"] == paired[f"{column}_pca"]
        ).all()

    benefit_rows = []

    for metric in ("macro_f1", "micro_f1", "hamming_loss"):
        # Positive benefit always means the PCA model performed better.
        if metric == "hamming_loss":
            benefit = paired[f"{metric}_audio"] - paired[f"{metric}_pca"]
        else:
            benefit = paired[f"{metric}_pca"] - paired[f"{metric}_audio"]

        paired[f"{metric}_pca_benefit"] = benefit

        benefit_rows.append({
            "metric": metric,
            "audio_mean": paired[f"{metric}_audio"].mean(),
            "pca_mean": paired[f"{metric}_pca"].mean(),
            "mean_pca_benefit": benefit.mean(),
            "benefit_fold_std": benefit.std(ddof=1),
            "folds_pca_better": int((benefit > 1e-12).sum()),
            "folds": len(paired),
        })

    benefit_summary = pd.DataFrame(benefit_rows)

    details.to_csv(OUTPUT / "audio_pca_label_details.csv", index=False)
    label_comparison.to_csv(
        OUTPUT / "audio_pca_label_comparison.csv", index=False
    )
    paired.to_csv(OUTPUT / "audio_pca_paired_folds.csv", index=False)
    benefit_summary.to_csv(
        OUTPUT / "audio_pca_paired_summary.csv", index=False
    )

    print("Per-label out-of-fold comparison:")
    print(
        label_comparison.sort_values("f1_difference", ascending=False)
        .round(4).to_string(index=False)
    )

    print("\nPaired-fold comparison:")
    print(benefit_summary.round(4).to_string(index=False))

    print("\nPositive benefit = PCA model performed better.")
    print("Per-label scores pool all training out-of-fold predictions.")
    print("Fold consistency is descriptive, not a significance test.")
    print(f"\nSaved in: {OUTPUT}")


if __name__ == "__main__":
    run_stage(
        __file__, run,
        requires=['dataset/model_inputs/train_y.csv', 'results/grouped_cv/audio_oof_probabilities.csv', 'results/grouped_cv/fold_results.csv', 'results/embedding_pca_grouped_cv/audio_embedding_pca32_lr_oof_probabilities.csv', 'results/embedding_pca_grouped_cv/fold_results.csv'],
        produces=['results/embedding_pca_grouped_cv/audio_pca_*.csv'],
        external=(),
        originals=False,
    )
