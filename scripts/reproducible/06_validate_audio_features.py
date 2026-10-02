"""Notebook-derived standalone stage. Source cells: [37]."""
from script_runtime import run_stage


def run(PROJECT, original_videos=None):
    from pathlib import Path
    import numpy as np
    import pandas as pd

    # Define the project folders

    CSV_DIR = PROJECT / "dataset" / "csvs"
    FEATURE_DIR = PROJECT / "dataset" / "features"

    # Load the saved features and extraction report
    features = pd.read_csv(FEATURE_DIR / "audio_features_all.csv")
    ready = pd.read_csv(CSV_DIR / "dataset_ready.csv")
    report = pd.read_csv(FEATURE_DIR / "audio_extraction_report.csv")

    # Define the expected 44 feature columns
    feature_cols = [
        f"mfcc_{i}_{stat}"
        for i in range(1, 21)
        for stat in ["mean", "std"]
    ] + ["rms_mean", "rms_std", "zcr_mean", "zcr_std"]

    # Check IDs, feature values, and extraction status
    assert set(features.columns) == {"Video_ID", *feature_cols}
    assert features["Video_ID"].notna().all()
    assert features["Video_ID"].is_unique
    assert set(features["Video_ID"]) == set(ready["Video_ID"])
    assert np.isfinite(features[feature_cols].to_numpy(dtype=float)).all()

    assert report["Video_ID"].is_unique
    assert set(report["Video_ID"]) == set(ready["Video_ID"])
    assert report["status"].eq("PASS").all()

    print("All features:", features.shape)
    print("Missing or infinite feature values: 0")
    print("Duplicate IDs: 0")
    print("Ready CSV coverage: PASS")

    # Check each split's row order and feature values
    for split in ["train", "val", "test"]:
        labels = pd.read_csv(CSV_DIR / f"{split}_ready.csv")
        split_features = pd.read_csv(
            FEATURE_DIR / f"{split}_audio_features.csv"
        )

        assert split_features["Video_ID"].is_unique
        assert split_features["Video_ID"].tolist() == labels["Video_ID"].tolist()

        expected = features.set_index("Video_ID").loc[
            labels["Video_ID"], feature_cols
        ]
        actual = split_features.set_index("Video_ID")[feature_cols]

        assert np.allclose(
            actual.to_numpy(dtype=float),
            expected.to_numpy(dtype=float)
        )

        print(f"{split}: {len(split_features)} clips | PASS")

    # Display clips whose extracted audio contains only zeros
    silent_mask = (
        report["all_zero_audio"].astype(str).str.lower().eq("true")
    )
    silent = report.loc[
        silent_mask, ["Video_ID", "duration_seconds"]
    ]

    print("\nAll-zero audio clips:", len(silent))
    print(silent.to_string(index=False))
    print("\nAudio feature validation complete.")


if __name__ == "__main__":
    run_stage(
        __file__, run,
        requires=['dataset/features/audio_features_all.csv', 'dataset/features/audio_extraction_report.csv', 'dataset/csvs/dataset_ready.csv', 'dataset/csvs/train_ready.csv', 'dataset/csvs/val_ready.csv', 'dataset/csvs/test_ready.csv', 'dataset/features/train_audio_features.csv', 'dataset/features/val_audio_features.csv', 'dataset/features/test_audio_features.csv'],
        produces=[],
        external=(),
        originals=False,
    )
