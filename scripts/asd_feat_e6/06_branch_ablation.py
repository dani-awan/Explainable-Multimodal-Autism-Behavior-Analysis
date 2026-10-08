"""Standalone E6 stage derived from notebook cell 124."""
from stage_runtime import stage_main

SPEC = {'name': '06_branch_ablation', 'source_cell': 124, 'fields': {'input_dir': 'dataset/external/asd_feat/derived/e6_inputs', 'base_run': 'results/asd_feat/e6_grouped_cv/20261007T202805_649935Z'}, 'requires': [('input_dir', 'development_combined_X.csv'), ('input_dir', 'development_metadata.csv'), ('input_dir', 'development_y.csv'), ('input_dir', 'input_schema.json'), ('base_run', 'experiment_config.json'), ('base_run', 'fold_results.csv'), ('base_run', 'oof_predictions.csv'), ('base_run', 'cv_summary.csv'), ('base_run', 'pooled_oof_metrics.csv'), ('base_run', 'three_branch_pca32_lr_fold1.joblib'), ('base_run', 'three_branch_pca32_lr_fold2.joblib'), ('base_run', 'three_branch_pca32_lr_fold3.joblib'), ('base_run', 'three_branch_pca32_lr_fold4.joblib'), ('base_run', 'three_branch_pca32_lr_fold5.joblib')], 'packages': ['numpy', 'pandas', 'sklearn', 'joblib', 'threadpoolctl', 'scipy'], 'writes': True}

def run(args):
    from pathlib import Path
    from datetime import datetime, timezone
    import hashlib
    import json
    import warnings

    import joblib
    import numpy as np
    import pandas as pd

    from sklearn.base import clone
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from threadpoolctl import threadpool_limits

    ROOT = args.project
    INPUT = args.input_dir
    BASE = args.base_run

    config = json.loads(
        (BASE / "experiment_config.json").read_text(encoding="utf-8")
    )
    THRESHOLD = float(config["threshold"])

    # Require exactly the same input artifacts as the completed E6 comparison.
    for name, expected_hash in config["input_sha256"].items():
        path = INPUT / name
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"Input changed since the base experiment: {name}")

    features = pd.read_csv(INPUT / "development_combined_X.csv")
    metadata = pd.read_csv(INPUT / "development_metadata.csv")
    saved_oof = pd.read_csv(BASE / "oof_predictions.csv")
    base_results = pd.read_csv(BASE / "fold_results.csv")

    ids = features["filename"].tolist()

    for name, table in (
        ("features", features),
        ("metadata", metadata),
        ("saved OOF", saved_oof),
    ):
        if table["filename"].isna().any() or table["filename"].duplicated().any():
            raise ValueError(f"{name}: invalid recording IDs.")
        if set(table["filename"]) != set(ids):
            raise ValueError(f"{name}: recording coverage differs.")

    metadata = metadata.set_index("filename").loc[ids].reset_index()
    saved_oof = saved_oof.set_index("filename").loc[ids].reset_index()

    groups = config["feature_groups"]
    columns = groups["audio"] + groups["rgb"] + groups["behavior"]

    if features.columns.tolist() != ["filename", *columns]:
        raise ValueError("Unexpected feature schema.")

    X = features[columns]
    y = metadata["target"].to_numpy(dtype=int)
    folds = metadata["evaluation_fold"].to_numpy()

    if not metadata["target"].isin([0, 1]).all():
        raise ValueError("Targets must be binary.")

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite input features.")

    for column in ["sub_id", "target", "evaluation_fold"]:
        if not metadata[column].equals(saved_oof[column]):
            raise ValueError(f"Base OOF metadata disagreement: {column}")

    if metadata["sub_id"].isna().any():
        raise ValueError("Missing participant IDs.")

    if not np.isin(folds, [1, 2, 3, 4, 5]).all():
        raise ValueError("Unexpected folds.")

    if metadata.groupby("sub_id")["evaluation_fold"].nunique().gt(1).any():
        raise ValueError("Inconsistent participant fold assignments.")

    full_results = base_results[
        base_results["model"].eq("three_branch_pca32_lr")
    ].set_index("fold").sort_index()

    if full_results.index.tolist() != [1, 2, 3, 4, 5]:
        raise ValueError("Expected five unique full-model fold results.")

    experiments = {
        "without_audio": ["rgb", "behavior"],
        "without_rgb": ["audio", "behavior"],
        "without_behavior": ["audio", "rgb"],
    }

    metric_names = [
        "asd_f1", "macro_f1", "precision", "recall",
        "balanced_accuracy", "roc_auc", "average_precision",
    ]


    def calculate_metrics(actual, probabilities):
        predictions = (probabilities >= THRESHOLD).astype(int)
        return {
            "asd_f1": f1_score(actual, predictions, zero_division=0),
            "macro_f1": f1_score(
                actual, predictions, average="macro", zero_division=0
            ),
            "precision": precision_score(actual, predictions, zero_division=0),
            "recall": recall_score(actual, predictions, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(actual, predictions),
            "roc_auc": roc_auc_score(actual, probabilities),
            "average_precision": average_precision_score(actual, probabilities),
        }


    # Validate fold separation and class support before fitting.
    for fold in range(1, 6):
        fit_mask = folds != fold
        evaluation_mask = folds == fold

        overlap = (
            set(metadata.loc[fit_mask, "sub_id"])
            & set(metadata.loc[evaluation_mask, "sub_id"])
        )
        if overlap:
            raise ValueError(f"Fold {fold}: participant overlap.")

        if (
            np.unique(y[fit_mask]).size != 2
            or np.unique(y[evaluation_mask]).size != 2
        ):
            raise ValueError(f"Fold {fold}: missing class.")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    OUTPUT = args.output
    OUTPUT.mkdir(parents=True, exist_ok=False)

    (OUTPUT / "experiment_config.json").write_text(
        json.dumps({
            "base_run": BASE.relative_to(ROOT).as_posix(),
            "base_config": config,
            "experiments": experiments,
            "scope": "One-branch removal from the fixed early-fusion pipeline",
            "test_evaluated": False,
            "threshold_tuning": False,
        }, indent=2),
        encoding="utf-8",
    )

    oof_probabilities = {
        name: np.full(len(X), np.nan) for name in experiments
    }
    fold_rows = []
    reproduction_rows = []

    print(f"Development recordings: {len(X)}")
    print(f"Fixed threshold: {THRESHOLD}")
    print(f"Output: {OUTPUT}")

    for fold in range(1, 6):
        fit_mask = folds != fold
        evaluation_mask = folds == fold

        # Load this project's saved full model and reproduce its fold predictions.
        bundle = joblib.load(
            BASE / f"three_branch_pca32_lr_fold{fold}.joblib"
        )
        full_model = bundle["model"]

        if bundle["input_columns"] != columns:
            raise ValueError("Saved model input order differs.")
        if not np.isclose(bundle["threshold"], THRESHOLD):
            raise ValueError("Saved model threshold differs.")

        with threadpool_limits(limits=2):
            positive_index = list(full_model.classes_).index(1)
            reproduced = full_model.predict_proba(
                X.loc[evaluation_mask]
            )[:, positive_index]

        expected = saved_oof.loc[
            evaluation_mask, "three_branch_pca32_lr_probability"
        ].to_numpy(dtype=float)

        if not np.allclose(reproduced, expected, rtol=1e-7, atol=1e-8):
            raise ValueError(f"Fold {fold}: full-model OOF reproduction failed.")

        full_scores = calculate_metrics(y[evaluation_mask], reproduced)

        for metric in metric_names:
            if not np.isclose(
                full_scores[metric], full_results.loc[fold, metric],
                rtol=1e-7, atol=1e-8,
            ):
                raise ValueError(f"Fold {fold}: base metric differs: {metric}")

        reproduction_rows.append({
            "fold": fold,
            "maximum_probability_difference": float(
                np.max(np.abs(reproduced - expected))
            ),
            "status": "PASS",
        })

        for name, retained_branches in experiments.items():
            print(f"Fold {fold}/5: Training {name}...", flush=True)

            # Clone clears fitted state; retain the original preprocessing settings.
            model = clone(full_model)
            transformers = [
                transformer
                for transformer in model.named_steps["branches"].transformers
                if transformer[0] in retained_branches
            ]

            if {item[0] for item in transformers} != set(retained_branches):
                raise ValueError("Unexpected saved pipeline branch structure.")

            model.set_params(branches__transformers=transformers)

            with threadpool_limits(limits=2):
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    model.fit(X.loc[fit_mask], y[fit_mask])

                positive_index = list(model.classes_).index(1)
                probabilities = model.predict_proba(
                    X.loc[evaluation_mask]
                )[:, positive_index]

            if (
                not np.isfinite(probabilities).all()
                or not ((probabilities >= 0) & (probabilities <= 1)).all()
            ):
                raise ValueError("Invalid ablation probabilities.")

            oof_probabilities[name][evaluation_mask] = probabilities
            fold_rows.append({
                "fold": fold,
                "experiment": name,
                **calculate_metrics(y[evaluation_mask], probabilities),
            })

            joblib.dump({
                "model": model,
                "input_columns": columns,
                "retained_branches": retained_branches,
                "threshold": THRESHOLD,
                "evaluation_fold": fold,
            }, OUTPUT / f"{name}_fold{fold}.joblib")

        pd.DataFrame(fold_rows).to_csv(
            OUTPUT / "fold_results.csv", index=False
        )
        print(f"Fold {fold}/5 complete | Participant overlap: 0", flush=True)

    ablation = pd.DataFrame(fold_rows)
    summary_rows = []

    # Positive benefit means the full model outperformed branch removal.
    for name in experiments:
        reduced = (
            ablation[ablation["experiment"].eq(name)]
            .set_index("fold").sort_index()
        )

        for metric in metric_names:
            full_values = full_results[metric].to_numpy(dtype=float)
            reduced_values = reduced[metric].to_numpy(dtype=float)
            benefit = full_values - reduced_values
            tied = np.isclose(benefit, 0, rtol=0, atol=1e-12)

            summary_rows.append({
                "experiment": name,
                "metric": metric,
                "full_model_mean": float(full_values.mean()),
                "reduced_model_mean": float(reduced_values.mean()),
                "mean_full_model_benefit": float(benefit.mean()),
                "benefit_fold_std": float(benefit.std(ddof=1)),
                "folds_full_model_better": int(((benefit > 0) & ~tied).sum()),
                "folds_tied": int(tied.sum()),
                "folds": 5,
            })

    oof = metadata[
        ["filename", "sub_id", "visit", "target", "evaluation_fold"]
    ].copy()

    for name, probabilities in oof_probabilities.items():
        if not np.isfinite(probabilities).all():
            raise ValueError(f"{name}: incomplete OOF coverage.")

        oof[f"{name}_probability"] = probabilities
        oof[f"{name}_prediction"] = (
            probabilities >= THRESHOLD
        ).astype(int)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT / "paired_ablation_summary.csv", index=False)
    oof.to_csv(OUTPUT / "oof_predictions.csv", index=False)
    pd.DataFrame(reproduction_rows).to_csv(
        OUTPUT / "full_model_reproduction.csv", index=False
    )

    print("\nPaired branch-ablation summary:")
    print(summary.round(4).to_string(index=False))
    print("\nPositive benefit = full model performed better.")
    print("Fold consistency is descriptive, not a significance test.")
    print("Full-model OOF reproduction: PASS")
    print("Ablation OOF coverage: PASS")
    print("Official test evaluation: NOT performed")
    print(f"Saved in: {OUTPUT}")


if __name__ == "__main__":
    stage_main(__file__, run, SPEC)
