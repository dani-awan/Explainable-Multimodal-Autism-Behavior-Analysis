"""Standalone E6 stage derived from notebook cell 123."""
from stage_runtime import stage_main

SPEC = {'name': '05_compare_folds', 'source_cell': 123, 'fields': {'base_run': 'results/asd_feat/e6_grouped_cv/20261007T202805_649935Z'}, 'requires': [('base_run', 'fold_results.csv'), ('base_run', 'experiment_config.json')], 'packages': ['numpy', 'pandas'], 'writes': True}

def run(args):
    from pathlib import Path
    import json

    import numpy as np
    import pandas as pd

    ROOT = args.project
    RUN = args.base_run

    OUTPUT = args.output
    OUTPUT.mkdir(parents=True, exist_ok=False)

    config = json.loads((RUN / "experiment_config.json").read_text(encoding="utf-8"))
    results = pd.read_csv(RUN / "fold_results.csv")

    if results[["model", "fold"]].duplicated().any():
        raise ValueError("Duplicate model/fold results.")

    comparisons = [
        ("three_branch_pca32_lr", "behavior_lr"),
        ("three_branch_pca32_lr", "rgb_pca32_lr"),
        ("three_branch_late_fusion", "behavior_lr"),
        ("three_branch_late_fusion", "rgb_pca32_lr"),
        ("three_branch_pca32_lr", "three_branch_late_fusion"),
    ]

    metrics = [
        "asd_f1", "macro_f1", "precision", "recall",
        "balanced_accuracy", "roc_auc", "average_precision",
    ]

    summary_rows = []
    difference_rows = []

    # Pair results from the same saved evaluation folds.
    for candidate, reference in comparisons:
        candidate_rows = (
            results[results["model"].eq(candidate)]
            .set_index("fold").sort_index()
        )
        reference_rows = (
            results[results["model"].eq(reference)]
            .set_index("fold").sort_index()
        )

        if (
            candidate_rows.index.tolist() != [1, 2, 3, 4, 5]
            or reference_rows.index.tolist() != [1, 2, 3, 4, 5]
        ):
            raise ValueError("Expected exactly five matching folds.")

        for metric in metrics:
            candidate_values = candidate_rows[metric].to_numpy(dtype=float)
            reference_values = reference_rows[metric].to_numpy(dtype=float)

            if not (
                np.isfinite(candidate_values).all()
                and np.isfinite(reference_values).all()
            ):
                raise ValueError(f"Non-finite scores: {metric}")

            differences = candidate_values - reference_values
            tied = np.isclose(differences, 0, rtol=0, atol=1e-12)

            summary_rows.append({
                "candidate": candidate,
                "reference": reference,
                "metric": metric,
                "candidate_mean": float(candidate_values.mean()),
                "reference_mean": float(reference_values.mean()),
                "mean_candidate_benefit": float(differences.mean()),
                "benefit_fold_std": float(differences.std(ddof=1)),
                "folds_candidate_better": int(
                    ((differences > 0) & ~tied).sum()
                ),
                "folds_tied": int(tied.sum()),
                "folds": 5,
            })

            for fold, difference in zip(range(1, 6), differences):
                difference_rows.append({
                    "candidate": candidate,
                    "reference": reference,
                    "metric": metric,
                    "fold": fold,
                    "candidate_minus_reference": float(difference),
                })

    summary = pd.DataFrame(summary_rows)
    differences = pd.DataFrame(difference_rows)

    summary.to_csv(OUTPUT / "paired_fold_comparison.csv", index=False)
    differences.to_csv(OUTPUT / "paired_fold_differences.csv", index=False)

    notes = {
        "comparison_unit": "Matched development evaluation folds",
        "recordings": config["recordings"],
        "threshold": config["threshold"],
        "positive_benefit": "Candidate score exceeds reference score",
        "interpretation": "Descriptive fold consistency; not a significance test",
        "scope": "No retraining, threshold tuning or official test evaluation",
        "limitations": [
            "Five folds have overlapping fitting sets.",
            "Three-branch models include human-coded behavioral features.",
            "Comparing three branches with RGB alone adds audio and behavior together; "
            "it does not isolate either branch's contribution.",
        ],
    }

    (OUTPUT / "paired_comparison_notes.json").write_text(
        json.dumps(notes, indent=2), encoding="utf-8"
    )

    print("Paired-fold comparison:")
    print(summary.round(4).to_string(index=False))
    print("\nPositive benefit = candidate performed better.")
    print("Fold consistency is descriptive, not a significance test.")
    print(f"Saved in: {OUTPUT}")
    print("No models or predictions were changed.")

if __name__ == "__main__":
    stage_main(__file__, run, SPEC)
