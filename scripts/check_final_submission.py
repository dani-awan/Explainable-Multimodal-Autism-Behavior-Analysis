"""Read-only final project checks; no model loading, fitting or file writes."""
import argparse
import hashlib
import importlib.util
from pathlib import Path
import sys

EXPECTED_HASH = "2c8e9d9c126e9a7a8134eb9792550254de3747847b96aeaca1e396bae57327d4"
MODEL = (
    "results/asd_feat/e6_reproduction/12_freeze_rgb_behavior_model/"
    "20261008T192951_063340Z/rgb_behavior_pca32_lr_frozen.joblib"
)
REQUIRED = [
    "README.md", "dashboard/app.py", "dashboard/pages/7_E6_Frozen_Test.py",
    "dashboard/pages/8_Research_Analysis.py",
    "docs/ASD_FEAT_E6_FROZEN_RESULTS.md", "docs/ASD_FEAT_E6_SELECTED_MODEL_CARD.md",
    "docs/RESEARCH_ANALYSIS_UI_VALIDATION.md", "docs/RESEARCH_INPUT_VALIDATION.json",
    "docs/E6_STANDALONE_WINDOWS_VALIDATION.md", "docs/MODEL_CARD.md",
    "docs/TRAINING_REPRODUCTION.md", "docs/DATASET_CARD.md",
    "docs/FINAL_RESEARCH_REPORT.pdf", "docs/FINAL_RESEARCH_REPORT.docx",
    "docs/FINAL_RESEARCH_REPORT.md", "docs/FINAL_DEMO_GUIDE.md",
    "docs/FINAL_SUBMISSION_CHECKLIST.md",
    "presentations/FINAL_RESEARCH_PRESENTATION.pptx", MODEL,
    "results/asd_feat/e6_reproduction/15_evaluate_frozen_rgb_behavior/"
    "20261008T194502_516518Z/frozen_test_comparison.csv",
    "results/asd_feat/e6_reproduction/15_evaluate_frozen_rgb_behavior/"
    "20261008T194502_516518Z/frozen_test_predictions.csv",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.project.resolve()
    failures = []
    print(f"Project: {root}")
    print(f"Python: {sys.executable}")
    for name in REQUIRED:
        path = root / name
        if path.is_file() and path.stat().st_size > 0:
            print(f"PASS file: {name}")
        else:
            failures.append(f"Missing or empty file: {name}")
    for name in ["dashboard/app.py", "dashboard/pages/7_E6_Frozen_Test.py", "dashboard/pages/8_Research_Analysis.py"]:
        path = root / name
        if not path.is_file():
            continue
        try:
            compile(path.read_text(encoding="utf-8-sig"), str(path), "exec")
            print(f"PASS syntax: {name}")
        except (SyntaxError, UnicodeError, OSError) as error:
            failures.append(f"Syntax/read error in {name}: {error}")
    for package in ["numpy", "pandas", "sklearn", "joblib", "streamlit"]:
        if importlib.util.find_spec(package) is None:
            failures.append(f"Package not available in this Python: {package}")
        else:
            print(f"PASS package presence: {package}")
    model_path = root / MODEL
    if model_path.is_file():
        digest = hashlib.sha256()
        with model_path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() == EXPECTED_HASH:
            print("PASS frozen model SHA-256")
        else:
            failures.append("Frozen model hash differs from the supplied final evidence. Do not replace or retrain it automatically.")
    print("\nInspection only. No models loaded, results recomputed or files changed.")
    if failures:
        for error in failures:
            print(f"FAIL: {error}")
        print(f"Final local checks: {len(failures)} finding(s). Review these before submission.")
        return 1
    print("Final local presence, syntax, package-presence and artifact-hash checks: PASS")
    print("Experiment validity, licensing and GitHub merge status are separate reviews.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
