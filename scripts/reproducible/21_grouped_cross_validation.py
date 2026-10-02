"""Notebook-derived standalone stage. Source cells: [58]."""
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
    from sklearn.model_selection import GroupKFold
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler


    INPUTS = PROJECT / "dataset" / "model_inputs"
    OUTPUT = PROJECT / "results" / "grouped_cv"
    OUTPUT.mkdir(parents=True, exist_ok=True)

    THRESHOLD = 0.5
    SEED = 42
    FOLDS = 5


    def load_table(path):
        """Load a table and validate its clip IDs."""
        table = pd.read_csv(path)

        if table["Video_ID"].isna().any():
            raise ValueError(f"Missing IDs in {path.name}")

        if table["Video_ID"].duplicated().any():
            raise ValueError(f"Duplicate IDs in {path.name}")

        return table.set_index("Video_ID")


    targets = load_table(INPUTS / "train_y.csv")
    labels = targets.columns.tolist()

    if not targets.isin([0, 1]).all().all():
        raise ValueError("Targets must be binary.")

    y = targets.to_numpy(dtype=int)
    feature_tables = {}

    for modality in ["audio", "visual", "combined"]:
        table = load_table(INPUTS / f"train_{modality}_X.csv")

        if set(table.index) != set(targets.index):
            raise ValueError(f"{modality}: Clip IDs differ.")

        table = table.loc[targets.index]
        values = table.to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(f"{modality}: Invalid feature values.")

        feature_tables[modality] = values

    parts = targets.index.to_series().str.rsplit("_", n=2)

    if not parts.map(len).eq(3).all():
        raise ValueError("Unexpected clip ID format.")

    groups = parts.str[0].to_numpy()

    if len(np.unique(groups)) < FOLDS:
        raise ValueError("Not enough source-video groups.")

    # Use identical source-group folds for every modality.
    splitter = GroupKFold(
        n_splits=FOLDS,
        shuffle=True,
        random_state=SEED,
    )
    # Reuse saved assignments when present instead of regenerating them.
    assignment_path = OUTPUT / "fold_assignments.csv"
    if assignment_path.is_file():
        saved = load_table(assignment_path)
        if set(saved.index) != set(targets.index):
            raise ValueError("Saved fold IDs differ from the training data.")
        saved = saved.loc[targets.index]
        if not np.array_equal(saved["Original_ID"].to_numpy(), groups):
            raise ValueError("Saved source groups differ.")
        membership = saved["Evaluation_fold"].to_numpy()
        if set(membership) != set(range(1, FOLDS + 1)):
            raise ValueError("Saved fold numbers differ.")
        splits = [(np.flatnonzero(membership != fold), np.flatnonzero(membership == fold))
                  for fold in range(1, FOLDS + 1)]
    else:
        splits = list(splitter.split(feature_tables["combined"], y, groups))

    fold_assignment = np.zeros(len(targets), dtype=int)
    rows = []
    support_rows = []

    model_names = ["audio", "visual", "combined", "late_fusion"]
    oof_probabilities = {
        name: np.full(y.shape, np.nan, dtype=float)
        for name in model_names
    }

    print("Training clips:", len(targets))
    print("Original-video groups:", len(np.unique(groups)))
    print("Fixed threshold:", THRESHOLD)

    for fold, (fit_indices, eval_indices) in enumerate(splits, start=1):
        overlap = set(groups[fit_indices]) & set(groups[eval_indices])

        if overlap:
            raise ValueError("Source-video leakage detected.")

        fold_assignment[eval_indices] = fold
        fold_probabilities = {}

        # Record label support to expose sparse or single-class folds.
        for label_index, label in enumerate(labels):
            support_rows.append({
                "fold": fold,
                "label": label,
                "fit_positive": int(y[fit_indices, label_index].sum()),
                "evaluation_positive": int(y[eval_indices, label_index].sum()),
                "fit_single_class": bool(
                    len(np.unique(y[fit_indices, label_index])) < 2
                ),
            })

        for modality in ["audio", "visual", "combined"]:
            # Fit a fresh scaler and classifier within this fold only.
            model = make_pipeline(
                StandardScaler(),
                OneVsRestClassifier(
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        solver="lbfgs",
                        max_iter=3000,
                        random_state=SEED,
                    ),
                    n_jobs=1,
                ),
            )

            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(
                    feature_tables[modality][fit_indices],
                    y[fit_indices],
                )

            probabilities = model.predict_proba(
                feature_tables[modality][eval_indices]
            )

            if probabilities.shape != y[eval_indices].shape:
                raise ValueError("Unexpected probability shape.")

            if not np.isfinite(probabilities).all():
                raise ValueError("Invalid probabilities.")

            fold_probabilities[modality] = probabilities

        # Use fixed equal weights without tuning on evaluation folds.
        fold_probabilities["late_fusion"] = (
            0.5 * fold_probabilities["audio"]
            + 0.5 * fold_probabilities["visual"]
        )

        for name, probabilities in fold_probabilities.items():
            predicted = (probabilities >= THRESHOLD).astype(int)
            oof_probabilities[name][eval_indices] = probabilities

            rows.append({
                "model": name,
                "fold": fold,
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

        print(f"Fold {fold}/{FOLDS} complete | Source overlap: 0")

    fold_results = pd.DataFrame(rows)
    fold_results.to_csv(OUTPUT / "fold_results.csv", index=False)

    summary = fold_results.groupby("model").agg(
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        micro_f1_mean=("micro_f1", "mean"),
        micro_f1_std=("micro_f1", "std"),
        hamming_loss_mean=("hamming_loss", "mean"),
        hamming_loss_std=("hamming_loss", "std"),
    ).reset_index()

    summary.to_csv(OUTPUT / "cv_summary.csv", index=False)
    pd.DataFrame(support_rows).to_csv(
        OUTPUT / "fold_label_support.csv", index=False
    )

    pd.DataFrame({
        "Video_ID": targets.index,
        "Original_ID": groups,
        "Evaluation_fold": fold_assignment,
    }).to_csv(OUTPUT / "fold_assignments.csv", index=False)

    # Save one out-of-fold probability row for every training clip.
    for name, probabilities in oof_probabilities.items():
        if not np.isfinite(probabilities).all():
            raise ValueError(f"{name}: Incomplete out-of-fold coverage.")

        table = pd.DataFrame(probabilities, columns=labels)
        table.insert(0, "Video_ID", targets.index)
        table.to_csv(
            OUTPUT / f"{name}_oof_probabilities.csv", index=False
        )

    config = {
        "data": "training split only",
        "folds": FOLDS,
        "grouping": "original video ID",
        "seed": SEED,
        "fixed_threshold": THRESHOLD,
        "late_fusion_audio_weight": 0.5,
        "scaling": "fitted separately inside each training fold",
        "macro_f1": "all 10 targets; zero_division=0",
        "purpose": "additional baseline robustness comparison",
    }

    (OUTPUT / "cv_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print("\nCross-validation summary:")
    print(summary.round(4).to_string(index=False))
    print("\nSaved in:", OUTPUT)
    print("Out-of-fold coverage: PASS")


if __name__ == "__main__":
    run_stage(
        __file__, run,
        requires=['dataset/model_inputs/train_y.csv', 'dataset/model_inputs/train_audio_X.csv', 'dataset/model_inputs/train_visual_X.csv', 'dataset/model_inputs/train_combined_X.csv'],
        produces=['results/grouped_cv/*.csv', 'results/grouped_cv/cv_config.json'],
        external=(),
        originals=False,
    )
