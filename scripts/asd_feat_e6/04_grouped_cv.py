"""Standalone E6 stage derived from notebook cell 122."""
from stage_runtime import stage_main

SPEC = {'name': '04_grouped_cv', 'source_cell': 122, 'fields': {'input_dir': 'dataset/external/asd_feat/derived/e6_inputs'}, 'requires': [('input_dir', 'development_combined_X.csv'), ('input_dir', 'development_metadata.csv'), ('input_dir', 'development_y.csv'), ('input_dir', 'input_schema.json')], 'packages': ['numpy', 'pandas', 'sklearn', 'joblib', 'threadpoolctl', 'scipy'], 'writes': True}

def run(args):
    from pathlib import Path
    from datetime import datetime, timezone
    import hashlib
    import json
    import warnings

    import joblib
    import numpy as np
    import pandas as pd
    import sklearn

    from sklearn.base import clone
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA
    from sklearn.dummy import DummyClassifier
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    ROOT = args.project
    INPUT = args.input_dir

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    OUTPUT = args.output
    OUTPUT.mkdir(parents=True, exist_ok=False)

    THRESHOLD = 0.5
    PCA_COMPONENTS = 32
    SEED = 42

    schema_path = INPUT / "input_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    groups = schema["feature_groups"]

    audio_columns = groups["audio"]
    rgb_columns = groups["rgb"]
    behavior_columns = groups["behavior"]
    all_columns = audio_columns + rgb_columns + behavior_columns

    X_path = INPUT / "development_combined_X.csv"
    metadata_path = INPUT / "development_metadata.csv"
    targets_path = INPUT / "development_y.csv"

    features = pd.read_csv(X_path)
    metadata = pd.read_csv(metadata_path)
    targets = pd.read_csv(targets_path)

    for name, table in (
        ("features", features),
        ("metadata", metadata),
        ("targets", targets),
    ):
        if table["filename"].isna().any():
            raise ValueError(f"{name}: missing filenames.")
        if table["filename"].duplicated().any():
            raise ValueError(f"{name}: duplicate filenames.")

    ids = features["filename"].tolist()

    for table in [metadata, targets]:
        if set(table["filename"]) != set(ids):
            raise ValueError("Input recording sets differ.")

    metadata = metadata.set_index("filename").loc[ids].reset_index()
    targets = targets.set_index("filename").loc[ids].reset_index()

    if features.columns.tolist() != ["filename", *all_columns]:
        raise ValueError("Combined feature schema or order differs.")

    expected_hash = schema["output_feature_sha256"][X_path.name]
    actual_hash = hashlib.sha256(X_path.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError("Combined features changed after input validation.")

    X = features[all_columns]
    y = targets["target"].to_numpy(dtype=int)
    folds = metadata["evaluation_fold"].to_numpy()

    if not targets["target"].isin([0, 1]).all():
        raise ValueError("Targets must be binary.")

    if not np.array_equal(y, metadata["target"].to_numpy()):
        raise ValueError("Target tables disagree.")

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Input features contain non-finite values.")

    if metadata["sub_id"].isna().any():
        raise ValueError("Missing participant IDs.")

    if not np.isin(folds, [1, 2, 3, 4, 5]).all():
        raise ValueError("Unexpected fold assignments.")

    if metadata.groupby("sub_id")["evaluation_fold"].nunique().gt(1).any():
        raise ValueError("A participant appears in multiple evaluation folds.")


    def pca_branch():
        # Fit preprocessing only when the enclosing model is fitted.
        return Pipeline([
            ("input_scaler", StandardScaler()),
            ("pca", PCA(
                n_components=PCA_COMPONENTS,
                svd_solver="randomized",
                random_state=SEED,
            )),
        ])


    def classifier():
        return LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=3000,
            random_state=SEED,
        )


    def make_model(branches):
        transformers = []

        if "audio" in branches:
            transformers.append(("audio", pca_branch(), audio_columns))
        if "rgb" in branches:
            transformers.append(("rgb", pca_branch(), rgb_columns))
        if "behavior" in branches:
            transformers.append((
                "behavior", StandardScaler(), behavior_columns
            ))

        return Pipeline([
            ("branches", ColumnTransformer(
                transformers,
                remainder="drop",
                n_jobs=1,
            )),
            ("output_scaler", StandardScaler()),
            ("classifier", classifier()),
        ])


    templates = {
        "behavior_lr": make_model(["behavior"]),
        "audio_pca32_lr": make_model(["audio"]),
        "rgb_pca32_lr": make_model(["rgb"]),
        "three_branch_pca32_lr": make_model(["audio", "rgb", "behavior"]),
        "dummy_prior": DummyClassifier(strategy="prior"),
    }

    model_names = [*templates, "three_branch_late_fusion"]


    def metrics(actual, probabilities):
        predictions = (probabilities >= THRESHOLD).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            actual, predictions, labels=[0, 1]
        ).ravel()

        return {
            "macro_f1": f1_score(
                actual, predictions, average="macro", zero_division=0
            ),
            "asd_f1": f1_score(actual, predictions, zero_division=0),
            "precision": precision_score(actual, predictions, zero_division=0),
            "recall": recall_score(actual, predictions, zero_division=0),
            "specificity": float(tn / (tn + fp)),
            "balanced_accuracy": balanced_accuracy_score(actual, predictions),
            "accuracy": accuracy_score(actual, predictions),
            "roc_auc": roc_auc_score(actual, probabilities),
            "average_precision": average_precision_score(actual, probabilities),
            "mcc": matthews_corrcoef(actual, predictions),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        }


    # Check all folds before fitting any model.
    fold_support = []

    for fold in range(1, 6):
        fit_mask = folds != fold
        evaluation_mask = folds == fold

        fit_participants = set(metadata.loc[fit_mask, "sub_id"])
        evaluation_participants = set(metadata.loc[evaluation_mask, "sub_id"])
        overlap = fit_participants & evaluation_participants

        if overlap:
            raise ValueError(f"Fold {fold}: participant overlap.")

        if (
            np.unique(y[fit_mask]).size != 2
            or np.unique(y[evaluation_mask]).size != 2
        ):
            raise ValueError(f"Fold {fold}: missing class.")

        if int(fit_mask.sum()) <= PCA_COMPONENTS:
            raise ValueError("Insufficient fitting rows for PCA.")

        fold_support.append({
            "fold": fold,
            "fit_recordings": int(fit_mask.sum()),
            "evaluation_recordings": int(evaluation_mask.sum()),
            "fit_participants": len(fit_participants),
            "evaluation_participants": len(evaluation_participants),
            "evaluation_positive_recordings": int(y[evaluation_mask].sum()),
            "participant_overlap": len(overlap),
        })

    config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "recordings": len(X),
        "participants": int(metadata["sub_id"].nunique()),
        "feature_groups": groups,
        "fold_source": schema["fold_source"],
        "threshold": THRESHOLD,
        "pca_components_per_audio_or_rgb_branch": PCA_COMPONENTS,
        "pca_solver": "randomized",
        "random_state": SEED,
        "logistic_settings": {
            "C": 1.0,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "max_iter": 3000,
        },
        "late_fusion": "Equal mean of audio, RGB and behavioral probabilities",
        "selection_or_tuning": "None; fixed exploratory development comparison",
        "target": schema["target"],
        "behavior_source": schema["behavior_source"],
        "sklearn_version": sklearn.__version__,
        "input_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [X_path, metadata_path, targets_path, schema_path]
        },
        "test_evaluated": False,
    }

    (OUTPUT / "experiment_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    pd.DataFrame(fold_support).to_csv(
        OUTPUT / "fold_support.csv", index=False
    )

    oof_probabilities = {
        name: np.full(len(X), np.nan) for name in model_names
    }
    fold_results = []

    print(f"Development recordings: {len(X)}")
    print(f"Participants: {metadata['sub_id'].nunique()}")
    print(f"Fixed threshold: {THRESHOLD}")
    print(f"Output: {OUTPUT}")

    for fold in range(1, 6):
        fit_mask = folds != fold
        evaluation_mask = folds == fold
        fold_probabilities = {}

        for name, template in templates.items():
            print(f"Fold {fold}/5: Training {name}...", flush=True)
            model = clone(template)

            # Keep preprocessing inside the fitted pipeline and limit CPU threads.
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
                or not np.logical_and(
                    probabilities >= 0, probabilities <= 1
                ).all()
            ):
                raise ValueError(f"{name}: invalid probabilities.")

            fold_probabilities[name] = probabilities
            oof_probabilities[name][evaluation_mask] = probabilities

            fold_results.append({
                "fold": fold,
                "model": name,
                **metrics(y[evaluation_mask], probabilities),
            })

            joblib.dump({
                "model": model,
                "input_columns": all_columns,
                "feature_groups": groups,
                "threshold": THRESHOLD,
                "evaluation_fold": fold,
                "target": schema["target"],
            }, OUTPUT / f"{name}_fold{fold}.joblib")

        # Use fixed equal weights without learning from evaluation outcomes.
        fusion = np.mean([
            fold_probabilities["audio_pca32_lr"],
            fold_probabilities["rgb_pca32_lr"],
            fold_probabilities["behavior_lr"],
        ], axis=0)

        oof_probabilities["three_branch_late_fusion"][evaluation_mask] = fusion
        fold_results.append({
            "fold": fold,
            "model": "three_branch_late_fusion",
            **metrics(y[evaluation_mask], fusion),
        })

        pd.DataFrame(fold_results).to_csv(
            OUTPUT / "fold_results.csv", index=False
        )
        print(f"Fold {fold}/5 complete | Participant overlap: 0", flush=True)

    # Verify complete out-of-fold coverage before reporting results.
    for name, probabilities in oof_probabilities.items():
        if not np.isfinite(probabilities).all():
            raise ValueError(f"{name}: incomplete OOF coverage.")

    oof = metadata[
        ["filename", "sub_id", "visit", "target", "evaluation_fold"]
    ].copy()

    for name, probabilities in oof_probabilities.items():
        oof[f"{name}_probability"] = probabilities
        oof[f"{name}_prediction"] = (
            probabilities >= THRESHOLD
        ).astype(int)

    oof.to_csv(OUTPUT / "oof_predictions.csv", index=False)

    fold_table = pd.DataFrame(fold_results)
    summary_metrics = [
        "macro_f1", "asd_f1", "balanced_accuracy",
        "roc_auc", "average_precision", "precision", "recall",
    ]

    summary = fold_table.groupby("model")[summary_metrics].agg(["mean", "std"])
    summary.columns = [
        f"{metric}_{statistic}" for metric, statistic in summary.columns
    ]
    summary = summary.reset_index()
    summary.to_csv(OUTPUT / "cv_summary.csv", index=False)

    pooled = pd.DataFrame([
        {"model": name, **metrics(y, probabilities)}
        for name, probabilities in oof_probabilities.items()
    ])
    pooled.to_csv(OUTPUT / "pooled_oof_metrics.csv", index=False)

    print("\nCross-validation summary:")
    print(summary.round(4).to_string(index=False))
    print("\nPooled recording-level out-of-fold metrics:")
    print(pooled.round(4).to_string(index=False))
    print("\nOut-of-fold coverage: PASS")
    print("Official test evaluation: NOT performed")
    print(f"Saved in: {OUTPUT}")

if __name__ == "__main__":
    stage_main(__file__, run, SPEC)
