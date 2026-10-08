"""Standalone E6 stage derived from notebook cell 125."""
from stage_runtime import stage_main

SPEC = {'name': '09_write_development_report', 'source_cell': 125, 'fields': {'base_run': 'results/asd_feat/e6_grouped_cv/20261007T202805_649935Z', 'ablation_run': 'results/asd_feat/e6_ablation/20261007T203536_456317Z'}, 'requires': [('base_run', 'experiment_config.json'), ('base_run', 'fold_results.csv'), ('base_run', 'oof_predictions.csv'), ('base_run', 'cv_summary.csv'), ('base_run', 'pooled_oof_metrics.csv'), ('ablation_run', 'paired_ablation_summary.csv'), ('ablation_run', 'full_model_reproduction.csv')], 'packages': ['pandas'], 'writes': True}

def run(args):
    from pathlib import Path
    from datetime import datetime, timezone
    import hashlib
    import json

    import pandas as pd

    ROOT = args.project
    BASE = args.base_run
    ABLATION = args.ablation_run
    DOCS = args.output
    DOCS.mkdir(parents=True, exist_ok=True)

    config = json.loads(
        (BASE / "experiment_config.json").read_text(encoding="utf-8")
    )
    comparison = pd.read_csv(BASE / "cv_summary.csv")
    pooled = pd.read_csv(BASE / "pooled_oof_metrics.csv")
    ablation = pd.read_csv(ABLATION / "paired_ablation_summary.csv")
    reproduction = pd.read_csv(ABLATION / "full_model_reproduction.csv")

    if not reproduction["status"].eq("PASS").all():
        raise ValueError("Full-model reproduction was not successful.")

    if comparison["model"].duplicated().any():
        raise ValueError("Duplicate comparison models.")

    if ablation[["experiment", "metric"]].duplicated().any():
        raise ValueError("Duplicate ablation summary rows.")


    def markdown_table(table):
        # Generate Markdown without requiring the optional tabulate package.
        columns = table.columns.tolist()
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
        ]

        for row in table.itertuples(index=False, name=None):
            values = []
            for value in row:
                if isinstance(value, float):
                    text = f"{value:.4f}"
                else:
                    text = str(value)
                values.append(text.replace("|", "\\|").replace("\n", " "))
            lines.append("| " + " | ".join(values) + " |")

        return "\n".join(lines)


    cv_columns = [
        "model", "asd_f1_mean", "asd_f1_std",
        "precision_mean", "recall_mean",
        "roc_auc_mean", "average_precision_mean",
    ]

    ablation_columns = [
        "experiment", "metric", "full_model_mean", "reduced_model_mean",
        "mean_full_model_benefit", "folds_full_model_better", "folds",
    ]

    report = f"""# ASD-FEAT E6 Development Experiments

    ## Scope

    This exploratory comparison uses {config['recordings']} development recordings
    from {config['participants']} participants. The target is current-visit
    `asd_longitudinal`; this experiment does not establish future diagnosis prediction.

    Three feature branches were aligned by recording filename:
    224 audio summaries, 1024 pooled I3D RGB features, and 12 human-coded behavioral
    features. The behavioral branch requires human annotations. The three branches
    do not represent three independent sensor modalities.

    Coverage details are recorded in the input schema and source coverage audit.
    This report describes the selected run; it does not assume full release coverage.

    ## Evaluation Protocol

    - Reused the existing five participant-group development folds.
    - Participant overlap between fitting and evaluation portions was zero.
    - Used the same {config["recordings"]} recordings for all E6 models and ablations.
    - Fitted scaling and branch-specific PCA inside each fitting fold only.
    - Used 32 PCA components per audio or RGB branch.
    - Used balanced logistic regression with C=1.0 and a fixed threshold of 0.5.
    - Used equal probability weights for late fusion.
    - Did not tune thresholds or evaluate the official test set in E6.

    ## Initial Model Comparison

    {markdown_table(comparison[cv_columns])}

    Scores above are unweighted means across evaluation folds. They differ from
    metrics calculated after pooling all out-of-fold predictions.

    ## Pooled Recording-Level Metrics

    {markdown_table(pooled)}

    ## Branch Ablation

    Positive full-model benefit means the three-branch early-fusion model
    outperformed the reduced model.

    {markdown_table(ablation[ablation_columns])}

    ## Findings

    The ablation table reports changes after refitting each reduced model.
    Positive benefit favors the full model, and negative benefit favors the
    reduced model. Interpretation applies to this representation and configuration;
    it does not establish that a branch is generally uninformative.

    The full three-branch model's saved out-of-fold probabilities were reproduced
    before ablation. Original models and predictions were preserved.

    ## Limitations

    - Precision remains low, and false positives remain substantial.
    - Repeated recordings from one participant are not independent participants.
    - Fold fitting sets overlap; fold consistency is descriptive and is not a
      statistical significance test.
    - Results are exploratory development evidence, not independent clinical
      validation.
    - Human-coded behavioral inputs limit claims of fully automated inference.
    - Branch removal measures performance changes after refitting; it does not
      establish causal contributions.
    - The official ASD-FEAT test set was previously evaluated for E1 and contains
      positive recordings from only one participant. Future E6 test results must
      disclose that history and limitation.
    - No final E6 model was frozen for official test evaluation by this report.

    ## Evidence Locations

    - Initial comparison: `{BASE.relative_to(ROOT).as_posix()}`
    - Ablation: `{ABLATION.relative_to(ROOT).as_posix()}`
    - Input provenance: the selected base run's `experiment_config.json`
    """

    report_path = DOCS / "ASD_FEAT_E6_DEVELOPMENT_RESULTS.md"
    report_path.write_text(report, encoding="utf-8")

    source_paths = [
        BASE / "experiment_config.json",
        BASE / "cv_summary.csv",
        BASE / "pooled_oof_metrics.csv",
        ABLATION / "paired_ablation_summary.csv",
        ABLATION / "full_model_reproduction.csv",
    ]

    evidence = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_run": BASE.relative_to(ROOT).as_posix(),
        "ablation_run": ABLATION.relative_to(ROOT).as_posix(),
        "recordings": config["recordings"],
        "participants": config["participants"],
        "threshold": config["threshold"],
        "source_sha256": {
            path.relative_to(ROOT).as_posix():
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_paths
        },
        "initial_cv_summary": comparison.to_dict(orient="records"),
        "paired_ablation": ablation.to_dict(orient="records"),
        "scope": "Development documentation only; no model selection or retraining",
        "e6_official_test_evaluated": False,
    }

    evidence_path = DOCS / "ASD_FEAT_E6_DEVELOPMENT_EVIDENCE.json"
    evidence_path.write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )

    print(f"Saved: {report_path}")
    print(f"Saved: {evidence_path}")
    print("E6 findings documented. No models or predictions changed.")

if __name__ == "__main__":
    stage_main(__file__, run, SPEC)
