"""Notebook-derived standalone stage. Source cells: [32]."""
from script_runtime import run_stage


def run(PROJECT, original_videos=None):
    from pathlib import Path
    from itertools import combinations
    import pandas as pd

    # Set project paths.
    project = PROJECT
    csv_folder = project / "dataset" / "csvs"
    clip_folder = project / "dataset" / "clips"

    # Load the source validation report.
    report = pd.read_csv(csv_folder / "video_check_report.csv")
    valid_ids = set(
        report.loc[report["status"] == "metadata_ok", "Video_ID"]
        .astype(str)
        .str.strip()
    )

    # Confirm every expected clip exists and is non-empty.
    missing_clips = [
        video_id
        for video_id in sorted(valid_ids)
        if not (clip_folder / f"{video_id}.mp4").is_file()
        or (clip_folder / f"{video_id}.mp4").stat().st_size == 0
    ]

    if missing_clips:
        raise ValueError(
            f"{len(missing_clips)} expected clips are missing or empty: "
            f"{missing_clips[:10]}"
        )

    # Keep the original train, validation, and test assignments.
    ready = {}

    for split in ("train", "val", "test"):
        df = pd.read_csv(csv_folder / f"{split}_clean.csv")
        df["Video_ID"] = df["Video_ID"].astype(str).str.strip()

        if df["Video_ID"].duplicated().any():
            raise ValueError(f"Duplicate Video_ID found in {split}.")

        ready[split] = df.loc[df["Video_ID"].isin(valid_ids)].copy()

    # Check clip and source-video separation before saving.
    for first, second in combinations(ready, 2):
        first_ids = set(ready[first]["Video_ID"])
        second_ids = set(ready[second]["Video_ID"])

        first_sources = {
            video_id.rsplit("_", 2)[0] for video_id in first_ids
        }
        second_sources = {
            video_id.rsplit("_", 2)[0] for video_id in second_ids
        }

        clip_overlap = first_ids & second_ids
        source_overlap = first_sources & second_sources

        if clip_overlap or source_overlap:
            raise ValueError(f"Overlap detected between {first} and {second}.")

        print(f"{first}/{second}: clip overlap=0, source overlap=0")

    # Confirm the splits cover all expected verified clips.
    combined = pd.concat(ready.values(), ignore_index=True)

    if set(combined["Video_ID"]) != valid_ids:
        raise ValueError("Split coverage does not match the valid clip IDs.")

    # Save separate CSVs without overwriting the cleaned originals.
    for split, df in ready.items():
        output = csv_folder / f"{split}_ready.csv"
        df.to_csv(output, index=False)
        print(f"{split}: {len(df)} clips | Saved: {output.name}")

    combined.to_csv(csv_folder / "dataset_ready.csv", index=False)
    print(f"\nTotal ready clips: {len(combined)}")
    print("Ready CSV files saved successfully.")


if __name__ == "__main__":
    run_stage(
        __file__, run,
        requires=['dataset/csvs/video_check_report.csv', 'dataset/csvs/train_clean.csv', 'dataset/csvs/val_clean.csv', 'dataset/csvs/test_clean.csv', 'dataset/clips'],
        produces=['dataset/csvs/*_ready.csv'],
        external=(),
        originals=False,
    )
