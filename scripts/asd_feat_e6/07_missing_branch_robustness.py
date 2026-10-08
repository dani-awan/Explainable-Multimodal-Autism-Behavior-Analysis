"""Standalone E6 stage derived from notebook cell 126."""
from stage_runtime import stage_main

SPEC = {'name': '07_missing_branch_robustness', 'source_cell': 126, 'fields': {'input_dir': 'dataset/external/asd_feat/derived/e6_inputs', 'base_run': 'results/asd_feat/e6_grouped_cv/20261007T202805_649935Z'}, 'requires': [('input_dir', 'development_combined_X.csv'), ('input_dir', 'development_metadata.csv'), ('input_dir', 'development_y.csv'), ('input_dir', 'input_schema.json'), ('base_run', 'experiment_config.json'), ('base_run', 'fold_results.csv'), ('base_run', 'oof_predictions.csv'), ('base_run', 'cv_summary.csv'), ('base_run', 'pooled_oof_metrics.csv'), ('base_run', 'three_branch_pca32_lr_fold1.joblib'), ('base_run', 'three_branch_pca32_lr_fold2.joblib'), ('base_run', 'three_branch_pca32_lr_fold3.joblib'), ('base_run', 'three_branch_pca32_lr_fold4.joblib'), ('base_run', 'three_branch_pca32_lr_fold5.joblib')], 'packages': ['numpy', 'pandas', 'sklearn', 'joblib', 'threadpoolctl', 'scipy'], 'writes': True}

def run(args):
    from pathlib import Path
    from datetime import datetime, timezone
    import hashlib
    import json

    import joblib
    import numpy as np
    import pandas as pd

    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        confusion_matrix,
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

    # Require the same inputs used in the original development experiment.
    for name, expected_hash in config["input_sha256"].items():
        path = INPUT / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Input changed since the base experiment: {name}")

    features = pd.read_csv(INPUT / "development_combined_X.csv")
    metadata = pd.read_csv(INPUT / "development_metadata.csv")
    saved_oof = pd.read_csv(BASE / "oof_predictions.csv")

    ids = features["filename"].tolist()

    for name, table in (
        ("features", features),
        ("metadata", metadata),
        ("saved OOF", saved_oof),
    ):
        if table["filename"].isna().any() or table["filename"].duplicated().any():
            raise ValueError(f"{name}: invalid recording IDs.")
        if set(table["filename"]) != set(ids):
            raise ValueError(f"{name}: recording sets differ.")

    metadata = metadata.set_index("filename").loc[ids].reset_index()
    saved_oof = saved_oof.set_index("filename").loc[ids].reset_index()

    groups = config["feature_groups"]
    columns = groups["audio"] + groups["rgb"] + groups["behavior"]

    if features.columns.tolist() != ["filename", *columns]:
        raise ValueError("Feature schema differs from the saved experiment.")

    X = features[columns]
    y = metadata["target"].to_numpy(dtype=int)
    folds = metadata["evaluation_fold"].to_numpy()
    threshold = float(config["threshold"])

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite input features.")

    if not metadata["target"].isin([0, 1]).all():
        raise ValueError("Targets must be binary.")

    if metadata["sub_id"].isna().any():
        raise ValueError("Missing participant IDs.")

    if not np.isin(folds, [1, 2, 3, 4, 5]).all():
        raise ValueError("Unexpected fold assignments.")

    for column in ["sub_id", "target", "evaluation_fold"]:
        if not metadata[column].equals(saved_oof[column]):
            raise ValueError(f"Saved OOF metadata disagreement: {column}")

    if metadata.groupby("sub_id")["evaluation_fold"].nunique().gt(1).any():
        raise ValueError("A participant appears in multiple evaluation folds.")

    conditions = {
        "all_available": None,
        "audio_mean_imputed": "audio",
        "rgb_mean_imputed": "rgb",
        "behavior_mean_imputed": "behavior",
    }


    def metrics(actual, probabilities):
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            actual, predictions, labels=[0, 1]
        ).ravel()

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
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "predicted_positive_fraction": float(predictions.mean()),
        }


    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    OUTPUT = args.output
    OUTPUT.mkdir(parents=True, exist_ok=False)

    protocol = {
        "base_run": BASE.relative_to(ROOT).as_posix(),
        "model": "Saved three_branch_pca32_lr fold models",
        "threshold": threshold,
        "conditions": conditions,
        "imputation": "Raw feature means computed from each fold's fitting rows only",
        "scope": "Simulated single-branch unavailability on development OOF rows",
        "models_retrained": False,
        "test_evaluated": False,
    }

    (OUTPUT / "experiment_config.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )

    probability_tables = {
        condition: np.full(len(X), np.nan) for condition in conditions
    }
    fold_rows = []
    reproduction_rows = []

    print(f"Development recordings: {len(X)}")
    print(f"Frozen threshold: {threshold}")
    print("Running simulated missing-branch checks...")

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

        bundle = joblib.load(
            BASE / f"three_branch_pca32_lr_fold{fold}.joblib"
        )

        if bundle["input_columns"] != columns:
            raise ValueError("Saved model feature order differs.")
        if not np.isclose(bundle["threshold"], threshold):
            raise ValueError("Saved model threshold differs.")

        model = bundle["model"]
        positive_index = list(model.classes_).index(1)

        # Compute imputation values exclusively from the fitting portion.
        fitting_means = X.loc[fit_mask].mean(axis=0)
        evaluation_features = X.loc[evaluation_mask]

        for condition, missing_branch in conditions.items():
            simulated = evaluation_features.copy()

            if missing_branch is not None:
                branch_columns = groups[missing_branch]
                simulated.loc[:, branch_columns] = np.broadcast_to(
                    fitting_means[branch_columns].to_numpy(dtype=float),
                    (len(simulated), len(branch_columns)),
                )

            with threadpool_limits(limits=2):
                probabilities = model.predict_proba(simulated)[:, positive_index]

            if (
                not np.isfinite(probabilities).all()
                or not ((probabilities >= 0) & (probabilities <= 1)).all()
            ):
                raise ValueError(f"{condition}: invalid probabilities.")

            if condition == "all_available":
                expected = saved_oof.loc[
                    evaluation_mask, "three_branch_pca32_lr_probability"
                ].to_numpy(dtype=float)

                if not np.allclose(
                    probabilities, expected, rtol=1e-7, atol=1e-8
                ):
                    raise ValueError(f"Fold {fold}: baseline reproduction failed.")

                reproduction_rows.append({
                    "fold": fold,
                    "maximum_probability_difference": float(
                        np.max(np.abs(probabilities - expected))
                    ),
                    "status": "PASS",
                })

            probability_tables[condition][evaluation_mask] = probabilities
            fold_rows.append({
                "fold": fold,
                "condition": condition,
                **metrics(y[evaluation_mask], probabilities),
            })

        print(f"Fold {fold}/5 complete | Participant overlap: 0", flush=True)

    fold_results = pd.DataFrame(fold_rows)
    metric_names = [
        "asd_f1", "macro_f1", "precision", "recall",
        "balanced_accuracy", "roc_auc", "average_precision",
    ]

    summary = fold_results.groupby("condition")[metric_names].agg(["mean", "std"])
    summary.columns = [
        f"{metric}_{statistic}" for metric, statistic in summary.columns
    ]
    summary = summary.reset_index()

    baseline = summary.set_index("condition").loc["all_available"]
    for metric in metric_names:
        summary[f"{metric}_mean_change"] = (
            summary[f"{metric}_mean"] - baseline[f"{metric}_mean"]
        )

    oof = metadata[
        ["filename", "sub_id", "visit", "target", "evaluation_fold"]
    ].copy()
    pooled_rows = []

    for condition, probabilities in probability_tables.items():
        if not np.isfinite(probabilities).all():
            raise ValueError(f"{condition}: incomplete OOF coverage.")

        oof[f"{condition}_probability"] = probabilities
        oof[f"{condition}_prediction"] = (
            probabilities >= threshold
        ).astype(int)

        pooled_rows.append({
            "condition": condition,
            **metrics(y, probabilities),
        })

    fold_results.to_csv(OUTPUT / "fold_results.csv", index=False)
    summary.to_csv(OUTPUT / "cv_robustness_summary.csv", index=False)
    pd.DataFrame(pooled_rows).to_csv(
        OUTPUT / "pooled_oof_metrics.csv", index=False
    )
    oof.to_csv(OUTPUT / "oof_predictions.csv", index=False)
    pd.DataFrame(reproduction_rows).to_csv(
        OUTPUT / "baseline_reproduction.csv", index=False
    )

    print("\nMean fold robustness comparison:")
    print(summary[[
        "condition", "asd_f1_mean", "precision_mean", "recall_mean",
        "roc_auc_mean", "average_precision_mean",
        "asd_f1_mean_change", "roc_auc_mean_change",
    ]].round(4).to_string(index=False))

    print("\nPooled recording-level metrics:")
    print(pd.DataFrame(pooled_rows).round(4).to_string(index=False))
    print("\nBaseline reproduction and OOF coverage: PASS")
    print("Simulation does not establish real-world missing-input reliability.")
    print("No models retrained or official test evaluated.")
    print(f"Saved in: {OUTPUT}")

if __name__ == "__main__":
    stage_main(__file__, run, SPEC)
