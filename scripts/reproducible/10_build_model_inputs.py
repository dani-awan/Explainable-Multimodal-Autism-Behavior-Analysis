"""Notebook-derived standalone stage. Source cells: [43]."""
from script_runtime import run_stage


def run(PROJECT, original_videos=None):
    from pathlib import Path
    import numpy as np
    import pandas as pd

    # Define the project folders

    CSV_DIR = PROJECT / "dataset" / "csvs"
    FEATURE_DIR = PROJECT / "dataset" / "features"
    MODEL_DIR = PROJECT / "dataset" / "model_inputs"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load the complete audio and visual feature tables
    audio = pd.read_csv(FEATURE_DIR / "audio_features_all.csv")
    visual = pd.read_csv(FEATURE_DIR / "visual_features_all.csv")

    audio_cols = audio.columns.drop("Video_ID").tolist()
    visual_cols = visual.columns.drop("Video_ID").tolist()

    assert len(audio_cols) == 44
    assert len(visual_cols) == 14
    assert not set(audio_cols).intersection(visual_cols)
    assert audio["Video_ID"].is_unique
    assert visual["Video_ID"].is_unique
    assert set(audio["Video_ID"]) == set(visual["Video_ID"])

    # Join features by clip identity
    combined = audio.merge(
        visual,
        on="Video_ID",
        how="inner",
        validate="one_to_one"
    )
    feature_cols = audio_cols + visual_cols

    assert np.isfinite(
        combined[feature_cols].to_numpy(dtype=float)
    ).all()

    # Save aligned features and targets for each original split
    clip_sets = {}
    source_sets = {}

    for split in ["train", "val", "test"]:
        labels = pd.read_csv(CSV_DIR / f"{split}_ready.csv")
        target_cols = labels.columns.drop("Video_ID").tolist()

        assert len(target_cols) == 10
        assert labels["Video_ID"].is_unique
        assert labels[target_cols].isin([0, 1]).all().all()

        aligned = labels[["Video_ID"]].merge(
            combined,
            on="Video_ID",
            how="left",
            validate="one_to_one",
            sort=False
        )

        assert aligned["Video_ID"].tolist() == labels["Video_ID"].tolist()
        assert np.isfinite(
            aligned[feature_cols].to_numpy(dtype=float)
        ).all()

        # Keep identifiers for alignment, but exclude them during training
        aligned[["Video_ID", *audio_cols]].to_csv(
            MODEL_DIR / f"{split}_audio_X.csv", index=False
        )
        aligned[["Video_ID", *visual_cols]].to_csv(
            MODEL_DIR / f"{split}_visual_X.csv", index=False
        )
        aligned[["Video_ID", *feature_cols]].to_csv(
            MODEL_DIR / f"{split}_combined_X.csv", index=False
        )
        labels[["Video_ID", *target_cols]].to_csv(
            MODEL_DIR / f"{split}_y.csv", index=False
        )

        clip_sets[split] = set(labels["Video_ID"])
        source_sets[split] = set(
            labels["Video_ID"].str.rsplit("_", n=2).str[0]
        )

        print(
            f"{split}: {len(aligned)} clips | "
            f"{len(feature_cols)} features | {len(target_cols)} targets"
        )

    # Verify that split membership remains separate
    for left, right in [
        ("train", "val"), ("train", "test"), ("val", "test")
    ]:
        clip_overlap = len(clip_sets[left] & clip_sets[right])
        source_overlap = len(source_sets[left] & source_sets[right])

        assert clip_overlap == 0
        assert source_overlap == 0

        print(
            f"{left}/{right}: clip overlap={clip_overlap}, "
            f"source overlap={source_overlap}"
        )

    print("\nModel input tables saved:", MODEL_DIR)
    print("Alignment and leakage checks: PASS")


if __name__ == "__main__":
    run_stage(
        __file__, run,
        requires=['dataset/features/audio_features_all.csv', 'dataset/features/visual_features_all.csv', 'dataset/csvs/train_ready.csv', 'dataset/csvs/val_ready.csv', 'dataset/csvs/test_ready.csv'],
        produces=['dataset/model_inputs/*.csv'],
        external=(),
        originals=False,
    )
