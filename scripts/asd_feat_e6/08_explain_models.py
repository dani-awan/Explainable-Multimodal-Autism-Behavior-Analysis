"""Standalone E6 stage derived from notebook cell 129."""
from stage_runtime import stage_main

SPEC = {'name': '08_explain_models', 'source_cell': 129, 'fields': {'input_dir': 'dataset/external/asd_feat/derived/e6_inputs', 'base_run': 'results/asd_feat/e6_grouped_cv/20261007T202805_649935Z'}, 'requires': [('input_dir', 'development_combined_X.csv'), ('input_dir', 'development_metadata.csv'), ('input_dir', 'development_y.csv'), ('input_dir', 'input_schema.json'), ('base_run', 'experiment_config.json'), ('base_run', 'fold_results.csv'), ('base_run', 'oof_predictions.csv'), ('base_run', 'cv_summary.csv'), ('base_run', 'pooled_oof_metrics.csv'), ('base_run', 'three_branch_pca32_lr_fold1.joblib'), ('base_run', 'three_branch_pca32_lr_fold2.joblib'), ('base_run', 'three_branch_pca32_lr_fold3.joblib'), ('base_run', 'three_branch_pca32_lr_fold4.joblib'), ('base_run', 'three_branch_pca32_lr_fold5.joblib')], 'packages': ['numpy', 'pandas', 'sklearn', 'joblib', 'threadpoolctl', 'scipy', 'matplotlib'], 'writes': True}

def run(args):
    from pathlib import Path
    from datetime import datetime, timezone
    import hashlib
    import json

    import joblib
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.special import expit

    ROOT = args.project
    INPUT_DIR = args.input_dir
    RUN_DIR = args.base_run

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    OUTPUT_DIR = args.output

    # Load existing development inputs and out-of-fold predictions.
    X_table = pd.read_csv(INPUT_DIR / "development_combined_X.csv")
    metadata = pd.read_csv(INPUT_DIR / "development_metadata.csv")
    oof = pd.read_csv(RUN_DIR / "oof_predictions.csv")

    MODEL_NAME = "three_branch_pca32_lr"
    PROBABILITY_COLUMN = f"{MODEL_NAME}_probability"

    for name, table in (
        ("features", X_table),
        ("metadata", metadata),
        ("OOF predictions", oof),
    ):
        if "filename" not in table:
            raise ValueError(f"{name}: filename column missing.")
        if table["filename"].isna().any():
            raise ValueError(f"{name}: missing recording IDs.")
        if table["filename"].duplicated().any():
            raise ValueError(f"{name}: duplicate recording IDs.")

    ids = X_table["filename"].tolist()

    if set(ids) != set(metadata["filename"]):
        raise ValueError("Feature and metadata recording IDs differ.")

    if set(ids) != set(oof["filename"]):
        raise ValueError("Feature and OOF recording IDs differ.")

    metadata = metadata.set_index("filename").loc[ids].reset_index()
    oof = oof.set_index("filename").loc[ids].reset_index()

    for column in ["sub_id", "evaluation_fold", "target"]:
        if column not in metadata or column not in oof:
            raise ValueError(f"Required metadata missing: {column}")
        if not np.array_equal(
            metadata[column].to_numpy(),
            oof[column].to_numpy(),
        ):
            raise ValueError(f"Metadata disagreement: {column}")

    if PROBABILITY_COLUMN not in oof:
        raise ValueError(f"OOF probabilities missing: {PROBABILITY_COLUMN}")

    feature_columns = X_table.columns.drop("filename").tolist()
    X = X_table[feature_columns]

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Input features contain missing or infinite values.")

    if not metadata["target"].isin([0, 1]).all():
        raise ValueError("Targets must be binary.")

    folds = sorted(metadata["evaluation_fold"].unique().tolist())

    if folds != [1, 2, 3, 4, 5]:
        raise ValueError(f"Unexpected evaluation folds: {folds}")

    # Explain one predetermined recording, without selecting by model score.
    example_index = int(
        np.flatnonzero(metadata["evaluation_fold"].to_numpy() == 1)[0]
    )

    branch_tables = []
    validation_rows = []
    model_hashes = {}
    importance_sum = np.zeros(len(feature_columns), dtype=float)
    explained_count = 0
    example_features = None
    example_branches = None
    example_summary = None

    for fold in folds:
        matches = list(
            RUN_DIR.rglob(f"{MODEL_NAME}_fold{fold}.joblib")
        )
        if len(matches) != 1:
            raise ValueError(
                f"Fold {fold}: expected one saved model, found {len(matches)}."
            )

        model_path = matches[0]

        # Load only this project's trusted model artifacts.
        bundle = joblib.load(model_path)
        model = bundle["model"]

        if bundle["input_columns"] != feature_columns:
            raise ValueError(f"Fold {fold}: input feature order differs.")

        if int(bundle["evaluation_fold"]) != int(fold):
            raise ValueError(f"Fold {fold}: saved fold identifier differs.")

        if not np.isclose(float(bundle["threshold"]), 0.5):
            raise ValueError(f"Fold {fold}: unexpected threshold.")

        groups = bundle["feature_groups"]
        grouped_columns = [
            column
            for branch in ["audio", "rgb", "behavior"]
            for column in groups[branch]
        ]

        if (
            len(grouped_columns) != len(feature_columns)
            or len(set(grouped_columns)) != len(feature_columns)
            or set(grouped_columns) != set(feature_columns)
        ):
            raise ValueError(f"Fold {fold}: feature groups are invalid.")

        evaluation_mask = (
            metadata["evaluation_fold"].to_numpy() == fold
        )
        fit_mask = ~evaluation_mask

        fit_participants = set(metadata.loc[fit_mask, "sub_id"])
        evaluation_participants = set(
            metadata.loc[evaluation_mask, "sub_id"]
        )
        if fit_participants & evaluation_participants:
            raise ValueError(f"Fold {fold}: participant overlap found.")

        X_eval = X.loc[evaluation_mask]
        classifier = model.named_steps["classifier"]

        if list(classifier.classes_) != [0, 1]:
            raise ValueError("Expected a binary classifier with classes [0, 1].")

        # Use the training-fold mean as the explanation reference.
        reference = X.loc[fit_mask].mean().to_numpy(dtype=float)
        reference_table = pd.DataFrame(
            [reference], columns=feature_columns
        )

        preprocessing = model[:-1]
        reference_transformed = np.asarray(
            preprocessing.transform(reference_table), dtype=float
        )[0]

        coefficients = np.asarray(classifier.coef_, dtype=float)[0]
        reference_logit = float(
            np.asarray(model.decision_function(reference_table)).reshape(-1)[0]
        )

        # Back-project the affine scaler/PCA pipeline in small batches.
        # This calculates coefficients without fitting any transformer.
        raw_weights = np.empty(len(feature_columns), dtype=float)
        batch_size = 64

        for start in range(0, len(feature_columns), batch_size):
            stop = min(start + batch_size, len(feature_columns))
            positions = np.arange(start, stop)

            probes = np.repeat(
                reference[None, :], len(positions), axis=0
            )
            probes[np.arange(len(positions)), positions] += 1.0

            transformed = np.asarray(
                preprocessing.transform(
                    pd.DataFrame(probes, columns=feature_columns)
                ),
                dtype=float,
            )

            raw_weights[positions] = (
                transformed - reference_transformed
            ) @ coefficients

        contributions = (
            X_eval.to_numpy(dtype=float) - reference
        ) * raw_weights

        actual_logits = np.asarray(
            model.decision_function(X_eval), dtype=float
        ).reshape(-1)

        reconstructed_logits = (
            reference_logit + contributions.sum(axis=1)
        )

        if not np.allclose(
            reconstructed_logits, actual_logits,
            rtol=1e-7, atol=1e-8,
        ):
            raise ValueError(f"Fold {fold}: logit reconstruction failed.")

        probabilities = model.predict_proba(X_eval)[:, 1]
        expected_probabilities = oof.loc[
            evaluation_mask, PROBABILITY_COLUMN
        ].to_numpy(dtype=float)

        if not np.allclose(
            probabilities, expected_probabilities,
            rtol=1e-7, atol=1e-8,
        ):
            raise ValueError(f"Fold {fold}: saved OOF reproduction failed.")

        if not np.allclose(
            expit(reconstructed_logits), probabilities,
            rtol=1e-7, atol=1e-8,
        ):
            raise ValueError(f"Fold {fold}: probability reconstruction failed.")

        branch_table = metadata.loc[
            evaluation_mask,
            ["filename", "sub_id", "evaluation_fold", "target"],
        ].copy()

        branch_table["model_probability"] = probabilities
        branch_table["prediction"] = (probabilities >= 0.5).astype(int)
        branch_table["reference_logit"] = reference_logit
        branch_table["reference_probability"] = expit(reference_logit)

        for branch in ["audio", "rgb", "behavior"]:
            positions = [
                feature_columns.index(column)
                for column in groups[branch]
            ]
            branch_table[f"{branch}_logit_contribution"] = (
                contributions[:, positions].sum(axis=1)
            )

        branch_table["reconstructed_logit"] = reconstructed_logits
        branch_tables.append(branch_table)

        importance_sum += np.abs(contributions).sum(axis=0)
        explained_count += len(X_eval)

        if example_index in X_eval.index:
            local_position = X_eval.index.tolist().index(example_index)

            example_features = pd.DataFrame({
                "feature": feature_columns,
                "value": X_eval.iloc[local_position].to_numpy(dtype=float),
                "training_fold_reference": reference,
                "raw_coefficient": raw_weights,
                "logit_contribution": contributions[local_position],
            })

            branch_lookup = {
                column: branch
                for branch in ["audio", "rgb", "behavior"]
                for column in groups[branch]
            }
            example_features["branch"] = (
                example_features["feature"].map(branch_lookup)
            )

            example_branches = pd.DataFrame({
                "branch": ["audio", "rgb", "behavior"],
                "logit_contribution": [
                    float(branch_table.iloc[local_position][
                        f"{branch}_logit_contribution"
                    ])
                    for branch in ["audio", "rgb", "behavior"]
                ],
            })

            example_summary = {
                "filename": ids[example_index],
                "evaluation_fold": int(fold),
                "target": int(metadata.loc[example_index, "target"]),
                "probability": float(probabilities[local_position]),
                "prediction": int(probabilities[local_position] >= 0.5),
                "threshold": 0.5,
                "reference_logit": reference_logit,
                "reference_probability": float(expit(reference_logit)),
            }

        model_hashes[model_path.relative_to(ROOT).as_posix()] = (
            hashlib.sha256(model_path.read_bytes()).hexdigest()
        )

        validation_rows.append({
            "fold": int(fold),
            "recordings": len(X_eval),
            "participant_overlap": 0,
            "maximum_logit_difference": float(
                np.max(np.abs(reconstructed_logits - actual_logits))
            ),
            "maximum_oof_probability_difference": float(
                np.max(np.abs(probabilities - expected_probabilities))
            ),
            "status": "PASS",
        })

        print(
            f"Fold {fold}/5: PASS | {len(X_eval)} recordings | "
            "Logit reconstruction and OOF reproduction"
        )

    # Check complete coverage before saving the explanations.
    all_branches = pd.concat(branch_tables, ignore_index=True)

    if (
        explained_count != len(X)
        or all_branches["filename"].duplicated().any()
        or set(all_branches["filename"]) != set(ids)
    ):
        raise ValueError("Explanation coverage failed.")

    if example_features is None:
        raise ValueError("The predetermined example was not explained.")

    all_branches = (
        all_branches.set_index("filename").loc[ids].reset_index()
    )

    global_features = pd.DataFrame({
        "feature": feature_columns,
        "mean_absolute_logit_contribution": importance_sum / explained_count,
    })

    global_features["branch"] = global_features["feature"].map(
        {
            column: branch
            for branch in ["audio", "rgb", "behavior"]
            for column in groups[branch]
        }
    )
    global_features = global_features.sort_values(
        "mean_absolute_logit_contribution", ascending=False
    )

    global_branches = pd.DataFrame([
        {
            "branch": branch,
            "mean_absolute_net_logit_contribution": float(
                all_branches[f"{branch}_logit_contribution"].abs().mean()
            ),
        }
        for branch in ["audio", "rgb", "behavior"]
    ])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    all_branches.to_csv(
        OUTPUT_DIR / "oof_branch_contributions.csv", index=False
    )
    global_features.to_csv(
        OUTPUT_DIR / "global_feature_contributions.csv", index=False
    )
    global_branches.to_csv(
        OUTPUT_DIR / "global_branch_contributions.csv", index=False
    )
    example_features.to_csv(
        OUTPUT_DIR / "example_feature_contributions.csv", index=False
    )
    example_branches.to_csv(
        OUTPUT_DIR / "example_branch_contributions.csv", index=False
    )
    pd.DataFrame(validation_rows).to_csv(
        OUTPUT_DIR / "reconstruction_checks.csv", index=False
    )

    # Plot signed branch contributions for the predetermined example.
    fig, ax = plt.subplots(figsize=(8, 4))
    values = example_branches["logit_contribution"]
    ax.barh(
        example_branches["branch"],
        values,
        color=["#2563eb" if value >= 0 else "#dc2626" for value in values],
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Logit contribution relative to training-fold mean")
    ax.set_title("Example: branch contributions")
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "example_branch_contributions.png",
        dpi=180,
    )
    plt.close(fig)

    # Plot the largest original-feature contributions for the example.
    top = example_features.assign(
        magnitude=example_features["logit_contribution"].abs()
    ).nlargest(15, "magnitude").sort_values("logit_contribution")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        top["feature"],
        top["logit_contribution"],
        color=[
            "#2563eb" if value >= 0 else "#dc2626"
            for value in top["logit_contribution"]
        ],
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Logit contribution relative to training-fold mean")
    ax.set_title("Example: top 15 feature contributions")
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "example_feature_contributions.png",
        dpi=180,
    )
    plt.close(fig)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "model": MODEL_NAME,
        "recordings": explained_count,
        "features": len(feature_columns),
        "folds": 5,
        "reference": "Each model's training-fold raw feature means",
        "method": "Exact affine-pipeline logit decomposition",
        "example": example_summary,
        "model_sha256": model_hashes,
        "limitations": [
            "Contributions explain model scores, not clinical causes.",
            "Contributions depend on the chosen reference.",
            "RGB dimensions are latent features, not named behaviors.",
            "Behavioral inputs are human-annotated.",
            "Branch sums can contain cancellation between features.",
            "Fold models differ; pooled attribution summaries are descriptive.",
        ],
        "scope": "Development OOF explanations only; no fitting or test evaluation",
    }

    (OUTPUT_DIR / "explanation_evidence.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nExample recording:", example_summary["filename"])
    print("Model score:", round(example_summary["probability"], 4))
    print("Frozen fold threshold:", example_summary["threshold"])
    print("\nExample branch contributions:")
    print(example_branches.round(4).to_string(index=False))
    print(f"\nExplained recordings: {explained_count}/{len(X)}")
    print("Reconstruction and OOF coverage: PASS")
    print("Saved:", OUTPUT_DIR)
    print("No models retrained or official test evaluated.")

if __name__ == "__main__":
    stage_main(__file__, run, SPEC)
