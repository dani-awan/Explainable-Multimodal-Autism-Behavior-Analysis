"""Notebook-derived standalone stage. Source cells: [71]."""
from script_runtime import run_stage


def run(PROJECT, original_videos=None):
    from pathlib import Path
    import json
    import warnings

    import numpy as np
    import pandas as pd

    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, hamming_loss
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler


    OUTPUT = PROJECT / "results" / "embedding_pca_grouped_cv"
    OUTPUT.mkdir(parents=True, exist_ok=True)

    THRESHOLD = 0.5
    PCA_COMPONENTS = 32
    MODEL_NAME = "audio_embedding_pca32_lr"


    def load_indexed(path):
        table = pd.read_csv(path)
        if table["Video_ID"].isna().any() or not table["Video_ID"].is_unique:
            raise ValueError(f"Missing or duplicate IDs: {path.name}")
        return table.set_index("Video_ID")


    # Load training inputs and the existing fold assignments.
    targets = load_indexed(PROJECT / "dataset/model_inputs/train_y.csv")
    audio = load_indexed(PROJECT / "dataset/model_inputs/train_audio_X.csv")
    embeddings = load_indexed(
        PROJECT / "dataset/features/visual_embeddings_train/"
        "train_visual_embeddings.csv"
    )
    assignments = load_indexed(
        PROJECT / "results/grouped_cv/fold_assignments.csv"
    )

    ids = targets.index
    for name, table in {
        "audio": audio,
        "embeddings": embeddings,
        "fold assignments": assignments,
    }.items():
        if set(table.index) != set(ids):
            raise ValueError(f"Training ID coverage mismatch: {name}")

    audio = audio.loc[ids]
    embeddings = embeddings.loc[ids]
    assignments = assignments.loc[ids]

    assert audio.shape[1] == 44
    assert embeddings.shape[1] == 1024
    assert targets.shape[1] == 10
    assert not set(audio.columns).intersection(embeddings.columns)

    X = pd.concat([audio, embeddings], axis=1)
    y = targets.to_numpy(dtype=int)
    labels = targets.columns.tolist()

    assert np.isfinite(X.to_numpy(dtype=float)).all()
    assert np.isin(y, [0, 1]).all()

    groups = np.array([video_id.rsplit("_", 2)[0] for video_id in ids])
    assert np.array_equal(
        groups, assignments["Original_ID"].astype(str).to_numpy()
    )

    fold_numbers = assignments["Evaluation_fold"].to_numpy()
    assert set(fold_numbers) == {1, 2, 3, 4, 5}

    rows = []
    oof = np.full(y.shape, np.nan)

    print(f"Training clips: {len(ids)}")
    print(f"Source-video groups: {len(set(groups))}")
    print(f"Features after PCA: {44 + PCA_COMPONENTS}")
    print(f"Fixed threshold: {THRESHOLD}")

    for fold in range(1, 6):
        fit_mask = fold_numbers != fold
        evaluation_mask = fold_numbers == fold

        overlap = set(groups[fit_mask]) & set(groups[evaluation_mask])
        assert not overlap

        # Fit embedding scaling and PCA only on the fold's training data.
        preprocessing = ColumnTransformer([
            ("audio", "passthrough", audio.columns.tolist()),
            (
                "embedding_pca",
                Pipeline([
                    ("scale", StandardScaler()),
                    ("pca", PCA(
                        n_components=PCA_COMPONENTS,
                        svd_solver="full",
                        whiten=False,
                    )),
                ]),
                embeddings.columns.tolist(),
            ),
        ])

        # Keep the original classifier settings for this comparison.
        model = Pipeline([
            ("preprocessing", preprocessing),
            ("scale", StandardScaler()),
            ("classifier", OneVsRestClassifier(
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=3000,
                    random_state=42,
                ),
                n_jobs=1,
            )),
        ])

        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(X.loc[fit_mask], y[fit_mask])

        probabilities = model.predict_proba(X.loc[evaluation_mask])
        assert probabilities.shape == y[evaluation_mask].shape
        assert np.isfinite(probabilities).all()
        assert ((probabilities >= 0) & (probabilities <= 1)).all()

        predictions = (probabilities >= THRESHOLD).astype(int)
        oof[evaluation_mask] = probabilities

        fitted_pca = (
            model.named_steps["preprocessing"]
            .named_transformers_["embedding_pca"]
            .named_steps["pca"]
        )

        rows.append({
            "model": MODEL_NAME,
            "fold": fold,
            "fit_clips": int(fit_mask.sum()),
            "evaluation_clips": int(evaluation_mask.sum()),
            "source_overlap": len(overlap),
            "threshold": THRESHOLD,
            "macro_f1": f1_score(
                y[evaluation_mask], predictions,
                average="macro", zero_division=0,
            ),
            "micro_f1": f1_score(
                y[evaluation_mask], predictions,
                average="micro", zero_division=0,
            ),
            "hamming_loss": hamming_loss(
                y[evaluation_mask], predictions
            ),
            "pca_explained_variance": float(
                fitted_pca.explained_variance_ratio_.sum()
            ),
        })

        pd.DataFrame(rows).to_csv(OUTPUT / "fold_results.csv", index=False)
        print(f"Fold {fold}/5 complete | Source overlap: 0")

    assert np.isfinite(oof).all()

    oof_table = pd.DataFrame(oof, columns=labels)
    oof_table.insert(0, "Video_ID", ids.to_numpy())
    oof_table.to_csv(
        OUTPUT / f"{MODEL_NAME}_oof_probabilities.csv", index=False
    )

    # Compare fold means with the previously completed experiments.
    previous = pd.concat([
        pd.read_csv(PROJECT / "results/grouped_cv/fold_results.csv"),
        pd.read_csv(PROJECT / "results/embedding_grouped_cv/fold_results.csv"),
    ], ignore_index=True)

    assert np.allclose(previous["threshold"], THRESHOLD)
    assert not previous.duplicated(["model", "fold"]).any()

    all_results = pd.concat([previous, pd.DataFrame(rows)], ignore_index=True)

    comparison = all_results.groupby("model").agg(
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        micro_f1_mean=("micro_f1", "mean"),
        micro_f1_std=("micro_f1", "std"),
        hamming_loss_mean=("hamming_loss", "mean"),
        hamming_loss_std=("hamming_loss", "std"),
    ).reset_index()

    comparison.to_csv(OUTPUT / "cv_comparison.csv", index=False)

    config = {
        "model": MODEL_NAME,
        "data": "Training split only",
        "folds": "Existing five source-group fold assignments",
        "threshold": THRESHOLD,
        "pca_components": PCA_COMPONENTS,
        "audio_features": 44,
        "classifier_C": 1.0,
        "class_weight": "balanced",
        "preprocessing": "Fitted separately within each training fold",
        "interpretation": "Exploratory follow-up after reviewing prior CV results",
    }
    (OUTPUT / "cv_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print("\nCV comparison:")
    print(comparison.round(4).to_string(index=False))
    print("\nOut-of-fold coverage: PASS")
    print(f"Saved in: {OUTPUT}")


if __name__ == "__main__":
    run_stage(
        __file__, run,
        requires=['dataset/model_inputs/train_y.csv', 'dataset/model_inputs/train_audio_X.csv', 'dataset/features/visual_embeddings_train/train_visual_embeddings.csv', 'results/grouped_cv/fold_assignments.csv', 'results/grouped_cv/fold_results.csv', 'results/embedding_grouped_cv/fold_results.csv'],
        produces=['results/embedding_pca_grouped_cv/*.csv', 'results/embedding_pca_grouped_cv/cv_config.json'],
        external=(),
        originals=False,
    )
