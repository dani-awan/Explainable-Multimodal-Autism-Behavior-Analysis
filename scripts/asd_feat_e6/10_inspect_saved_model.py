"""Standalone E6 stage derived from notebook cell 128."""
from stage_runtime import stage_main

SPEC = {'name': '10_inspect_saved_model', 'source_cell': 128, 'fields': {'base_run': 'results/asd_feat/e6_grouped_cv/20261007T202805_649935Z'}, 'requires': [('base_run', 'three_branch_pca32_lr_fold1.joblib')], 'packages': ['joblib', 'sklearn'], 'writes': False}

def run(args):
    from pathlib import Path
    import joblib

    ROOT = args.project
    RUN_DIR = args.base_run

    matches = list(
        RUN_DIR.rglob("three_branch_pca32_lr_fold1.joblib")
    )

    if len(matches) != 1:
        available = [
            path.relative_to(RUN_DIR).as_posix()
            for path in RUN_DIR.rglob("*.joblib")
        ]
        raise ValueError(
            f"Expected one fold-1 model; found {len(matches)}. "
            f"Available models: {available}"
        )

    # Load only this project's trusted saved model.
    bundle = joblib.load(matches[0])

    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError("Unexpected model bundle structure.")

    model = bundle["model"]

    print("Model file:", matches[0].name)
    print("Bundle keys:", list(bundle))
    print("Pipeline:", model)
    print("Threshold:", bundle.get("threshold"))
    print("Input columns:", len(bundle.get("input_columns", [])))
    print("No training or predictions were changed.")

if __name__ == "__main__":
    stage_main(__file__, run, SPEC)
