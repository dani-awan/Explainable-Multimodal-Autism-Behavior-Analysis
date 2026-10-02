"""Notebook-derived standalone stage. Source cells: [78]."""
from script_runtime import run_stage


def run(PROJECT, original_videos=None):
    from pathlib import Path
    RESULTS = PROJECT / "results/modality_robustness"
    DOCS = PROJECT / "docs"
    DOCS.mkdir(parents=True, exist_ok=True)
    required = [RESULTS / "validation_missing_modality_comparison.csv"]
    # Build the report directly from the saved results.
    import pandas as pd

    table = pd.read_csv(required[0])

    def markdown_table(frame):
        formatted = frame.copy()
        for column in formatted.select_dtypes(include="number").columns:
            formatted[column] = formatted[column].map(
                lambda value: f"{value:.4f}"
            )
        header = "| " + " | ".join(formatted.columns) + " |"
        separator = "| " + " | ".join(["---"] * len(formatted.columns)) + " |"
        rows = [
            "| " + " | ".join(map(str, row)) + " |"
            for row in formatted.itertuples(index=False, name=None)
        ]
        return "\n".join([header, separator, *rows])

    report_text = f"""# Missing-Modality Robustness

    ## Method

    - Model: frozen combined logistic regression.
    - Data: 108 validation clips.
    - Prediction threshold: 0.30.
    - Model retraining: none.
    - Missing features were replaced with the scaler's saved training means.
    - Conditions: both available, audio replaced, visual replaced.

    ## Results

    {markdown_table(table)}

    ## Interpretation

    Both replacement conditions reduced Macro F1 and Micro F1.

    Replacing audio reduced false positives from 336 to 270, while
    false negatives increased from 64 to 83. Its lower Hamming loss
    therefore does not establish an overall improvement.

    Replacing visual features increased false negatives from 64 to 73.

    The frozen model is sensitive to both feature groups under these
    replacement conditions.

    ## Limits

    This is a synthetic sensitivity check, not a retrained ablation.
    Training-mean replacement does not reproduce every form of real
    missing audio or missing visual data.

    The validation split had already been used for model development.
    These results are descriptive, not a new independent evaluation.

    The primary baseline and its frozen threshold remain unchanged.
    """

    report = DOCS / "MODALITY_ROBUSTNESS.md"
    report.write_text(report_text, encoding="utf-8")

    # Link the robustness report from the visual progress document.
    progress = DOCS / "VISUAL_EXPERIMENT_PROGRESS.md"
    if progress.is_file():
        content = progress.read_text(encoding="utf-8")
        link = (
            "\n\n## Additional robustness experiment\n\n"
            "See [Missing-Modality Robustness](MODALITY_ROBUSTNESS.md) "
            "for the frozen-model validation sensitivity check.\n"
        )
        if "(MODALITY_ROBUSTNESS.md)" not in content:
            progress.write_text(content + link, encoding="utf-8")

    print(f"Saved: {report}")


if __name__ == "__main__":
    run_stage(
        __file__, run,
        requires=['results/modality_robustness/validation_missing_modality_comparison.csv'],
        produces=['docs/MODALITY_ROBUSTNESS.md', 'docs/VISUAL_EXPERIMENT_PROGRESS.md'],
        external=(),
        originals=False,
    )
