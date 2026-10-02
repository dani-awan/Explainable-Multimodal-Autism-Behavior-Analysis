"""Notebook-derived standalone stage. Source cells: [48]."""
from script_runtime import run_stage


def run(PROJECT, original_videos=None):
    from pathlib import Path
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        f1_score, precision_score, recall_score, hamming_loss
    )


    INPUT_DIR = PROJECT / "dataset" / "model_inputs"
    MODEL_DIR = PROJECT / "models" / "baselines"
    RESULT_DIR = PROJECT / "results" / "baselines"

    # Load the trained unimodal models
    audio_bundle = joblib.load(
        MODEL_DIR / "audio_logistic_regression.joblib"
    )
    visual_bundle = joblib.load(
        MODEL_DIR / "visual_logistic_regression.joblib"
    )

    target_cols = audio_bundle["target_columns"]
    assert target_cols == visual_bundle["target_columns"]

    labels = pd.read_csv(INPUT_DIR / "val_y.csv")
    audio_probs = pd.read_csv(
        RESULT_DIR / "audio_validation_probabilities.csv"
    )
    visual_probs = pd.read_csv(
        RESULT_DIR / "visual_validation_probabilities.csv"
    )

    # Confirm that both probability tables describe the same clips
    assert audio_probs["Video_ID"].is_unique
    assert visual_probs["Video_ID"].is_unique
    assert audio_probs["Video_ID"].tolist() == labels["Video_ID"].tolist()
    assert visual_probs["Video_ID"].tolist() == labels["Video_ID"].tolist()

    y = labels[target_cols].to_numpy(dtype=int)
    pa = audio_probs[target_cols].to_numpy(dtype=float)
    pv = visual_probs[target_cols].to_numpy(dtype=float)

    assert np.isfinite(pa).all() and np.isfinite(pv).all()
    assert ((pa >= 0) & (pa <= 1)).all()
    assert ((pv >= 0) & (pv <= 1)).all()

    # Use fixed equal weights for this late-fusion baseline
    fused = 0.5 * pa + 0.5 * pv

    rows = []
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        pred = (fused >= threshold).astype(int)

        rows.append({
            "threshold": threshold,
            "macro_f1": f1_score(
                y, pred, average="macro", zero_division=0
            ),
            "micro_f1": f1_score(
                y, pred, average="micro", zero_division=0
            ),
            "precision": precision_score(
                y, pred, average="micro", zero_division=0
            ),
            "recall": recall_score(
                y, pred, average="micro", zero_division=0
            ),
            "hamming_loss": hamming_loss(y, pred)
        })

    # Apply the same selection rule as the earlier baselines
    best = max(
        rows,
        key=lambda row: (
            row["macro_f1"],
            row["micro_f1"],
            -abs(row["threshold"] - 0.5)
        )
    )

    # Save both component models and the fusion configuration
    joblib.dump(
        {
            "type": "late_fusion",
            "audio_bundle": audio_bundle,
            "visual_bundle": visual_bundle,
            "audio_weight": 0.5,
            "visual_weight": 0.5,
            "target_columns": target_cols,
            "threshold": best["threshold"],
            "threshold_selection": "Validation Macro F1"
        },
        MODEL_DIR / "late_fusion_logistic_tuned.joblib"
    )

    pd.DataFrame(rows).to_csv(
        RESULT_DIR / "late_fusion_threshold_search.csv", index=False
    )

    probability_table = pd.DataFrame(fused, columns=target_cols)
    probability_table.insert(
        0, "Video_ID", labels["Video_ID"].to_numpy()
    )
    probability_table.to_csv(
        RESULT_DIR / "late_fusion_validation_probabilities.csv",
        index=False
    )

    print("Late fusion validation comparison:")
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    print(
        f"\nSelected threshold: {best['threshold']} | "
        f"Macro F1: {best['macro_f1']:.4f} | "
        f"Micro F1: {best['micro_f1']:.4f}"
    )
    print("Late fusion model and reports saved.")


if __name__ == "__main__":
    run_stage(
        __file__, run,
        requires=['dataset/model_inputs/val_y.csv', 'models/baselines/audio_logistic_regression.joblib', 'models/baselines/visual_logistic_regression.joblib', 'results/baselines/audio_validation_probabilities.csv', 'results/baselines/visual_validation_probabilities.csv'],
        produces=['models/baselines/late_fusion_logistic_tuned.joblib', 'results/baselines/late_fusion_*.csv'],
        external=(),
        originals=False,
    )
