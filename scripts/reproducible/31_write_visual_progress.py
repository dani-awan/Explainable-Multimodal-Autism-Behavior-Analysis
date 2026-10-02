"""Notebook-derived standalone stage. Source cells: [73]."""
from script_runtime import run_stage


def run(PROJECT, original_videos=None):
    from pathlib import Path
    import json
    import pandas as pd


    DOCS = PROJECT / "docs"
    DOCS.mkdir(parents=True, exist_ok=True)

    RESULTS = PROJECT / "results"

    # Load the completed experiment reports.
    comparison = pd.read_csv(
        RESULTS / "embedding_pca_grouped_cv/cv_comparison.csv"
    )
    paired = pd.read_csv(
        RESULTS / "embedding_pca_grouped_cv/audio_pca_paired_summary.csv"
    )
    labels = pd.read_csv(
        RESULTS / "embedding_pca_grouped_cv/audio_pca_label_comparison.csv"
    )
    diagnostics = pd.read_csv(
        RESULTS / "embedding_grouped_cv/oof_prediction_diagnostics.csv"
    )

    # Format Markdown tables without requiring an additional package.
    def markdown_table(table):
        table = table.copy()

        for column in table.select_dtypes(include="number").columns:
            table[column] = table[column].map(
                lambda value: f"{value:.4f}"
                if pd.notna(value) else ""
            )

        def escape(value):
            return str(value).replace("|", r"\|").replace("\n", " ")

        header = "| " + " | ".join(map(escape, table.columns)) + " |"
        separator = "| " + " | ".join(["---"] * len(table.columns)) + " |"
        rows = [
            "| " + " | ".join(map(escape, row)) + " |"
            for row in table.itertuples(index=False, name=None)
        ]

        return "\n".join([header, separator, *rows])


    # Build a separate report without replacing the baseline progress document.
    report = f"""# Visual Feature Experiments

    ## Dataset and evaluation

    - Prepared clips: 527.
    - Training / validation / test clips: 312 / 108 / 107.
    - These experiments used the training split only.
    - Training source-video groups: 185.
    - Evaluation used the same five saved source-group folds.
    - Source overlap between fitting and evaluation portions was zero.
    - The prediction threshold remained fixed at 0.5.
    - Source-video grouping does not establish participant independence.

    ## Pose feasibility

    The Pose Landmarker Lite audit processed 150 sampled frames.
    There were 112 single-pose frames. At least one complete confident
    arm was available in 60 frames, and both arms in 29 frames.

    The Full model audit found both confident arms in 22 of 150 frames.
    It did not improve complete-arm coverage in this audit.

    These are coverage measurements, not pose accuracy measurements.
    Pose features were not added to the main classifier.

    ## Pretrained visual embeddings

    - Extractor: MobileNetV3 Small ImageEmbedder pretrained on ImageNet.
    - Sampling: five uniformly spaced frames per clip.
    - Clip representation: mean of the five frame embeddings.
    - Embedding dimension: 1024.
    - Training extraction: 312 clips passed, zero failed.
    - Mean pooling produces a static clip representation.
    - These embeddings are visual features, not an independent third modality.
    - Backgrounds, overlays, and scene appearance can affect the features.

    ## Raw embedding diagnostics

    {markdown_table(diagnostics)}

    The all-negative reference Hamming loss was approximately 0.1615.
    Low Hamming loss alone did not establish useful behavior detection.
    Raw embedding models predicted fewer positive labels and missed
    more actual positive labels than the audio-only model.

    ## PCA experiment

    The 1024 embedding features were reduced to 32 PCA components.
    With 44 audio features, the classifier received 76 features.

    Embedding scaling, PCA, and final feature scaling were fitted
    separately within each fold's fitting portion. Logistic regression
    settings and the prediction threshold were unchanged.

    ## Grouped cross-validation comparison

    {markdown_table(comparison)}

    These values are means and sample standard deviations across folds.

    ## Paired-fold comparison with audio-only

    {markdown_table(paired)}

    Positive benefit means the PCA model performed better.
    Fold consistency is descriptive and is not a significance test.

    ## Per-label comparison

    {markdown_table(labels.sort_values("f1_difference", ascending=False))}

    Per-label scores pool the training out-of-fold predictions.
    They are not the same calculation as mean per-fold Macro F1.

    ## Interpretation

    - PCA improved F1 compared with the raw audio-plus-embedding model.
    - PCA improved Micro F1 over audio-only in four of five folds.
    - PCA reduced Hamming loss over audio-only in all five folds.
    - PCA improved Macro F1 over audio-only in only one of five folds.
    - Self-Spinning and Upper Limb Stereotypies showed higher pooled F1.
    - Object Lining-Up had zero pooled F1 and recall with the PCA model.
    - Background had only seven positive training examples.

    The PCA model remains an experimental comparison.
    The previously frozen primary baseline was not replaced.

    ## Limits and experiment status

    The PCA experiment followed inspection of earlier CV results.
    Its scores are exploratory; repeated use of these folds can influence
    model development decisions.

    The original test results had already been inspected in earlier work.
    These new training experiments do not create a new independent test set.

    Only the available 527-clip subset was prepared. Class imbalance,
    limited source groups, incomplete pose coverage, and static frame
    pooling remain relevant limitations.

    ## Saved evidence

    - results/grouped_cv/
    - results/embedding_grouped_cv/
    - results/embedding_pca_grouped_cv/
    - results/pose_quality_audit/
    - results/pose_quality_audit_full/
    - dataset/features/visual_embeddings_train/
    """

    report_path = DOCS / "VISUAL_EXPERIMENT_PROGRESS.md"
    report_path.write_text(report, encoding="utf-8")

    # Save a concise machine-readable experiment record.
    record = {
        "training_clips": 312,
        "source_video_groups": 185,
        "evaluation_folds": 5,
        "threshold": 0.5,
        "embedding_dimension": 1024,
        "frames_per_clip": 5,
        "pooling": "mean",
        "pca_components": 32,
        "audio_pca_feature_count": 76,
        "primary_baseline_changed": False,
        "status": "Exploratory training-only comparison",
        "cv_comparison": comparison.to_dict(orient="records"),
        "paired_fold_comparison": paired.to_dict(orient="records"),
    }

    record_path = DOCS / "visual_experiment_summary.json"
    record_path.write_text(
        json.dumps(record, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(f"Saved: {report_path}")
    print(f"Saved: {record_path}")
    print("\nVisual experiment documentation complete.")


if __name__ == "__main__":
    run_stage(
        __file__, run,
        requires=['results/embedding_pca_grouped_cv/cv_comparison.csv', 'results/embedding_pca_grouped_cv/audio_pca_paired_summary.csv', 'results/embedding_pca_grouped_cv/audio_pca_label_comparison.csv', 'results/embedding_grouped_cv/oof_prediction_diagnostics.csv'],
        produces=['docs/VISUAL_EXPERIMENT_PROGRESS.md', 'docs/visual_experiment_summary.json'],
        external=(),
        originals=False,
    )
