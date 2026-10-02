"""Notebook-derived standalone stage. Source cells: [59]."""
from script_runtime import run_stage


def run(PROJECT, original_videos=None):
    from pathlib import Path

    import numpy as np
    import pandas as pd


    CV_DIR = PROJECT / "results" / "grouped_cv"
    OUTPUT = PROJECT / "results" / "ablation"
    OUTPUT.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(CV_DIR / "fold_results.csv")

    if results.duplicated(["model", "fold"]).any():
        raise ValueError("Duplicate model-fold rows found.")

    required_models = {"audio", "visual", "combined"}

    if not required_models.issubset(set(results["model"])):
        raise ValueError("Required model results are missing.")

    if not results["source_overlap"].eq(0).all():
        raise ValueError("Source overlap detected.")

    if not np.allclose(results["threshold"], 0.5):
        raise ValueError("Expected the fixed 0.5 threshold.")

    comparisons = [
        ("Without visual", "audio"),
        ("Without audio", "visual"),
    ]

    metrics = ["macro_f1", "micro_f1", "hamming_loss"]
    fold_rows = []
    summary_rows = []

    for experiment, reduced_model in comparisons:
        full = results.loc[
            results["model"] == "combined"
        ].set_index("fold")

        reduced = results.loc[
            results["model"] == reduced_model
        ].set_index("fold")

        if set(full.index) != set(reduced.index):
            raise ValueError("Evaluation folds do not match.")

        reduced = reduced.loc[full.index]

        if not np.array_equal(
            full["evaluation_clips"].to_numpy(),
            reduced["evaluation_clips"].to_numpy(),
        ):
            raise ValueError("Evaluation clip counts do not match.")

        for metric in metrics:
            # Positive benefit always means the full model performed better.
            if metric == "hamming_loss":
                benefit = reduced[metric] - full[metric]
            else:
                benefit = full[metric] - reduced[metric]

            for fold in full.index:
                fold_rows.append({
                    "experiment": experiment,
                    "metric": metric,
                    "fold": int(fold),
                    "full_model_score": float(full.loc[fold, metric]),
                    "reduced_model_score": float(reduced.loc[fold, metric]),
                    "full_model_benefit": float(benefit.loc[fold]),
                })

            summary_rows.append({
                "experiment": experiment,
                "metric": metric,
                "full_model_mean": full[metric].mean(),
                "reduced_model_mean": reduced[metric].mean(),
                "mean_full_model_benefit": benefit.mean(),
                "benefit_fold_std": benefit.std(ddof=1),
                "folds_full_model_better": int((benefit > 0).sum()),
                "folds": len(benefit),
            })

    fold_report = pd.DataFrame(fold_rows)
    summary = pd.DataFrame(summary_rows)

    fold_report.to_csv(
        OUTPUT / "paired_fold_ablation.csv", index=False
    )
    summary.to_csv(
        OUTPUT / "ablation_summary.csv", index=False
    )

    # Document the interpretation and experiment boundaries.
    notes = """# Audio-Visual Baseline Ablation

    The combined, audio-only, and visual-only Logistic Regression models
    were fitted separately using identical source-group cross-validation
    folds from the training split.

    All models used a fixed prediction threshold of 0.5.
    Scalers were fitted inside each training fold.

    Without visual = retrained audio-only model.
    Without audio = retrained visual-only model.

    A positive full-model benefit means:
    - Higher Macro F1 or Micro F1 for the combined model.
    - Lower Hamming loss for the combined model.

    These are retrained modality ablations, not zeroing inputs in an
    already-fitted model. Fold variation is descriptive and is not a
    significance test. Structured behavioral inputs are not included.
    """

    (OUTPUT / "ABLATION_NOTES.md").write_text(
        notes, encoding="utf-8"
    )

    print("Paired-fold ablation summary:")
    print(summary.round(4).to_string(index=False))
    print("\nPositive benefit = combined model performed better.")
    print("Saved in:", OUTPUT)


if __name__ == "__main__":
    run_stage(
        __file__, run,
        requires=['results/grouped_cv/fold_results.csv'],
        produces=['results/ablation/*'],
        external=(),
        originals=False,
    )
