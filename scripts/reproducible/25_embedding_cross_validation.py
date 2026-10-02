"""Notebook-derived standalone stage. Source cells: [69]."""
from script_runtime import run_stage


def run(PROJECT, original_videos=None):
    from pathlib import Path
    import json
    import warnings

    import numpy as np
    import pandas as pd

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, hamming_loss
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler


    INPUTS = PROJECT / "dataset" / "model_inputs"
    BASE_CV = PROJECT / "results" / "grouped_cv"
    EMBEDDINGS = (
        PROJECT / "dataset" / "features" / "visual_embeddings_train"
    )
    OUTPUT = PROJECT / "results" / "embedding_grouped_cv"
    OUTPUT.mkdir(parents=True, exist_ok=True)

    THRESHOLD = 0.5


    def load_indexed(path):
        """Load a table and validate clip IDs."""
        table = pd.read_csv(path)

        if table["Video_ID"].isna().any():
            raise ValueError(f"Missing IDs: {path.name}")

        if table["Video_ID"].duplicated().any():
            raise ValueError(f"Duplicate IDs: {path.name}")

        return table.set_index("Video_ID")


    targets = load_indexed(INPUTS / "train_y.csv")
    audio = load_indexed(INPUTS / "train_audio_X.csv")
    embeddings = load_indexed(
        EMBEDDINGS / "train_visual_embeddings.csv"
    )
    assignments = load_indexed(BASE_CV / "fold_assignments.csv")
    extraction_report = load_indexed(
        EMBEDDINGS / "train_embedding_report.csv"
    )

    # Require complete training coverage before fitting.
    for name, table in [
        ("audio", audio),
        ("embeddings", embeddings),
        ("fold assignments", assignments),
        ("extraction report", extraction_report),
    ]:
        if set(table.index) != set(targets.index):
            raise ValueError(f"{name}: Training clip IDs differ.")

    audio = audio.loc[targets.index]
    embeddings = embeddings.loc[targets.index]
    assignments = assignments.loc[targets.index]
    extraction_report = extraction_report.loc[targets.index]

    if len(targets) != 312 or embeddings.shape[1] != 1024:
        raise ValueError("Unexpected training size or embedding dimensions.")

    if not extraction_report["status"].eq("PASS").all():
        raise ValueError("Some embedding extractions failed.")

    if not extraction_report["frames"].eq(5).all():
        raise ValueError("Expected five sampled frames per clip.")

    if not targets.isin([0, 1]).all().all():
        raise ValueError("Targets must be binary.")

    labels = targets.columns.tolist()
    y = targets.to_numpy(dtype=int)

    audio_values = audio.to_numpy(dtype=float)
    embedding_values = embeddings.to_numpy(dtype=float)

    features = {
        "embedding_lr": embedding_values,
        "audio_embedding_lr": np.concatenate(
            [audio_values, embedding_values], axis=1
        ),
    }

    for name, values in features.items():
        if not np.isfinite(values).all():
            raise ValueError(f"{name}: Non-finite feature values.")

    groups = (
        targets.index.to_series()
        .str.rsplit("_", n=2).str[0]
        .to_numpy()
    )

    if not np.array_equal(
        groups, assignments["Original_ID"].to_numpy()
    ):
        raise ValueError("Source IDs do not match the saved folds.")

    fold_numbers = assignments["Evaluation_fold"].to_numpy()

    if set(fold_numbers) != {1, 2, 3, 4, 5}:
        raise ValueError("Expected five saved evaluation folds.")

    rows = []
    oof = {
        name: np.full(y.shape, np.nan)
        for name in features
    }

    print("Training clips:", len(targets))
    print("Source-video groups:", len(np.unique(groups)))
    print("Fixed threshold:", THRESHOLD)

    for fold in range(1, 6):
        fit_indices = np.flatnonzero(fold_numbers != fold)
        eval_indices = np.flatnonzero(fold_numbers == fold)

        overlap = set(groups[fit_indices]) & set(groups[eval_indices])

        if overlap:
            raise ValueError("Source-video leakage detected.")

        for name, values in features.items():
            print(f"Fold {fold}/5: Training {name}...", flush=True)

            # Fit a fresh scaler and regularized LR within each fold.
            model = make_pipeline(
                StandardScaler(),
                OneVsRestClassifier(
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        solver="lbfgs",
                        max_iter=3000,
                        random_state=42,
                    ),
                    n_jobs=1,
                ),
            )

            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(values[fit_indices], y[fit_indices])

            probabilities = model.predict_proba(values[eval_indices])

            if probabilities.shape != y[eval_indices].shape:
                raise ValueError("Unexpected probability shape.")

            if not np.isfinite(probabilities).all():
                raise ValueError("Invalid probabilities.")

            predicted = (probabilities >= THRESHOLD).astype(int)
            oof[name][eval_indices] = probabilities

            rows.append({
                "model": name,
                "fold": fold,
                "features": values.shape[1],
                "fit_clips": len(fit_indices),
                "evaluation_clips": len(eval_indices),
                "source_overlap": len(overlap),
                "threshold": THRESHOLD,
                "macro_f1": f1_score(
                    y[eval_indices], predicted,
                    average="macro", zero_division=0,
                ),
                "micro_f1": f1_score(
                    y[eval_indices], predicted,
                    average="micro", zero_division=0,
                ),
                "hamming_loss": hamming_loss(
                    y[eval_indices], predicted
                ),
            })

            pd.DataFrame(rows).to_csv(
                OUTPUT / "fold_results.csv", index=False
            )

        print(f"Fold {fold}/5 complete | Source overlap: 0")

    new_results = pd.DataFrame(rows)

    # Combine summaries with the earlier fixed-threshold training CV.
    previous = pd.read_csv(BASE_CV / "fold_results.csv")

    if not previous["threshold"].eq(THRESHOLD).all():
        raise ValueError("Previous CV used a different threshold.")

    for fold in range(1, 6):
        expected_count = int((fold_numbers == fold).sum())
        old_fold = previous.loc[previous["fold"].eq(fold)]

        if old_fold.empty or not old_fold["evaluation_clips"].eq(
            expected_count
        ).all():
            raise ValueError("Previous evaluation fold sizes differ.")

    all_results = pd.concat([previous, new_results], ignore_index=True)

    summary = all_results.groupby("model").agg(
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        micro_f1_mean=("micro_f1", "mean"),
        micro_f1_std=("micro_f1", "std"),
        hamming_loss_mean=("hamming_loss", "mean"),
        hamming_loss_std=("hamming_loss", "std"),
    ).reset_index()

    summary.to_csv(
        OUTPUT / "baseline_embedding_cv_comparison.csv",
        index=False,
    )

    # Save one out-of-fold prediction score per training clip and target.
    for name, probabilities in oof.items():
        if not np.isfinite(probabilities).all():
            raise ValueError(f"{name}: Incomplete out-of-fold coverage.")

        table = pd.DataFrame(probabilities, columns=labels)
        table.insert(0, "Video_ID", targets.index)
        table.to_csv(
            OUTPUT / f"{name}_oof_probabilities.csv",
            index=False,
        )

    config = {
        "data": "training split only",
        "folds": "same saved five source-group folds as baseline CV",
        "threshold": THRESHOLD,
        "lr_C": 1.0,
        "class_weight": "balanced",
        "scaling": "fit separately inside each training fold",
        "visual_embedding_features": 1024,
        "audio_embedding_features": int(features["audio_embedding_lr"].shape[1]),
        "pooling": "mean of five sampled frame embeddings",
        "purpose": "additional feature comparison; no validation/test evaluation",
    }

    (OUTPUT / "cv_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print("\nBaseline and embedding CV comparison:")
    print(summary.round(4).to_string(index=False))
    print("\nOut-of-fold coverage: PASS")
    print("Saved in:", OUTPUT)


if __name__ == "__main__":
    run_stage(
        __file__, run,
        requires=['dataset/model_inputs/train_y.csv', 'dataset/model_inputs/train_audio_X.csv', 'results/grouped_cv/fold_assignments.csv', 'results/grouped_cv/fold_results.csv', 'dataset/features/visual_embeddings_train/train_visual_embeddings.csv', 'dataset/features/visual_embeddings_train/train_embedding_report.csv'],
        produces=['results/embedding_grouped_cv/*.csv', 'results/embedding_grouped_cv/cv_config.json'],
        external=(),
        originals=False,
    )
