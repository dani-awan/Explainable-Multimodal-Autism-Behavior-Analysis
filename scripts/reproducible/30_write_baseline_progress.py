"""Notebook-derived standalone stage. Source cells: [52]."""
from script_runtime import run_stage


def run(PROJECT, original_videos=None):
    from pathlib import Path
    import json
    import platform
    import importlib.metadata
    import pandas as pd


    DOC_DIR = PROJECT / "docs"
    RESULT_DIR = PROJECT / "results" / "baselines"
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    # Read the frozen evaluation results
    results = pd.read_csv(
        RESULT_DIR / "frozen_baseline_test_comparison.csv"
    )
    with (RESULT_DIR / "frozen_baseline_config.json").open(
        encoding="utf-8"
    ) as file:
        config = json.load(file)

    # Format the results without additional dependencies
    columns = [
        "model", "threshold", "macro_f1", "micro_f1",
        "precision", "recall", "hamming_loss", "exact_match_accuracy"
    ]

    table = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |"
    ]

    for _, row in results.iterrows():
        values = [
            str(row[column]) if column == "model"
            else f"{float(row[column]):.4f}"
            for column in columns
        ]
        table.append("| " + " | ".join(values) + " |")

    # Record package versions without personal filesystem paths
    versions = {"python": platform.python_version()}
    for package in [
        "numpy", "pandas", "librosa", "soundfile",
        "opencv-python", "scikit-learn", "matplotlib", "joblib"
    ]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not found"

    report = f"""# Baseline Experiment Progress

    ## Project
    Explainable Multimodal AI for Autism-Related Behavioral Analysis.

    ## Completed work
    - Cleaned and checked 928 AV-ASD annotations.
    - Prepared and verified 527 audio-video clips.
    - Preserved splits: 312 training, 108 validation, 107 test clips.
    - Verified zero clip and source-video overlap across splits.
    - Extracted 44 audio and 14 basic visual features per clip.
    - Confirmed 7 clips have entirely zero audio samples.
    - Recovered 6 initially failed visual extractions.
    - Created aligned feature and target tables with 58 input features.
    - Trained audio, visual, and combined Logistic Regression baselines.
    - Trained a combined Random Forest baseline.
    - Evaluated equal-weight audio-visual probability late fusion.
    - Selected thresholds using validation Macro F1.
    - Completed held-out test evaluation with frozen configurations.
    - Generated exact linear contribution explanations and charts.

    ## Evaluation protocol
    Primary baseline: `{config["primary_model"]}`.

    Models were fitted on the training split only. Logistic Regression
    scalers were fitted on training features only. Thresholds were selected
    on validation data before test evaluation. Test results were not used
    to change the primary model or thresholds.

    ## Test results
    {chr(10).join(table)}

    ## Explainability
    The primary Logistic Regression model was explained using signed
    feature contributions to logits relative to training feature means.
    The reconstruction checks passed.

    These contributions are not SHAP values and do not establish causality.
    Model probability scores are not accuracy percentages.

    ## Limitations
    - This is an available-media subset of the original annotations.
    - Features describe basic sound and visual appearance; they do not
      directly encode gaze, pose, speech meaning, or specific behaviors.
    - Temporal-change features can reflect camera movement or scene cuts.
    - Five short clips required first/last-frame sampling.
    - Seven silent clips were retained; the test split contains none.
    - Background has only 7 training positives and 1 validation positive.
    - False positives remain frequent and performance is limited.
    - Repeated validation comparisons may overestimate selected scores.
    - Source-video separation does not establish participant separation.
    - Grouped cross-validation and uncertainty estimates remain pending.

    ## Remaining work
    - Review per-label test errors and estimate uncertainty.
    - Assess robustness and explanation stability.
    - Build a research demonstration dashboard.
    - Consolidate reusable scripts and documentation.

    This project is intended for behavioral research, not clinical diagnosis.
    """

    report_path = DOC_DIR / "BASELINE_PROGRESS.md"
    report_path.write_text(report, encoding="utf-8")

    version_path = DOC_DIR / "experiment_versions.json"
    version_path.write_text(
        json.dumps(versions, indent=2), encoding="utf-8"
    )

    print("Saved:", report_path)
    print("Saved:", version_path)


if __name__ == "__main__":
    run_stage(
        __file__, run,
        requires=['results/baselines/frozen_baseline_test_comparison.csv', 'results/baselines/frozen_baseline_config.json'],
        produces=['docs/BASELINE_PROGRESS.md', 'docs/experiment_versions.json'],
        external=(),
        originals=False,
    )
