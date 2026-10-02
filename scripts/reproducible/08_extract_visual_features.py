"""Notebook-derived standalone stage. Source cells: [40, 41]."""
from script_runtime import run_stage


def run(PROJECT, original_videos=None):
    from pathlib import Path
    import cv2
    import numpy as np
    import pandas as pd

    # Define the project folders

    CSV_DIR = PROJECT / "dataset" / "csvs"
    CLIP_DIR = PROJECT / "dataset" / "clips"
    FEATURE_DIR = PROJECT / "dataset" / "features"
    PREVIEW_DIR = PROJECT / "dataset" / "frame_previews"

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)


    def extract_visual_features(video_id):
        """Extract basic appearance and temporal-change features."""
        clip_path = CLIP_DIR / f"{video_id}.mp4"
        cap = cv2.VideoCapture(str(clip_path))

        frame_values = []
        changes = []
        previous_gray = None

        try:
            if not cap.isOpened():
                raise RuntimeError("Cannot open the video.")

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

            if not np.isfinite([fps, frame_count]).all():
                raise ValueError("Invalid video metadata.")
            if fps <= 0 or frame_count <= 0:
                raise ValueError("Invalid FPS or frame count.")

            # Select approximately one frame per second
            duration = frame_count / fps
            targets = np.unique(
                np.rint(np.arange(0, duration, 1.0) * fps).astype(int)
            )
            targets = targets[targets < int(frame_count)]

            for index, frame_number in enumerate(targets):
                if not cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number)):
                    raise RuntimeError("Frame seeking is unsupported.")

                success, frame = cap.read()
                if not success or frame is None:
                    raise RuntimeError(
                        f"Cannot decode frame {frame_number}."
                    )

                # Resize while preserving the original aspect ratio
                height, width = frame.shape[:2]
                scale = 224 / max(height, width)
                resized = cv2.resize(
                    frame,
                    (max(1, round(width * scale)),
                     max(1, round(height * scale))),
                    interpolation=cv2.INTER_AREA
                )

                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                gray_float = gray.astype(np.float32) / 255.0
                edges = cv2.Canny(gray, 100, 200)

                # Measure RGB averages, brightness, contrast, and edges
                rgb_mean = rgb.mean(axis=(0, 1)) / 255.0
                frame_values.append([
                    *rgb_mean,
                    float(gray_float.mean()),
                    float(gray_float.std()),
                    float((edges > 0).mean())
                ])

                # Measure appearance change between sampled frames
                if previous_gray is not None:
                    changes.append(
                        float(np.abs(gray_float - previous_gray).mean())
                    )
                previous_gray = gray_float

                # Save up to three preview images per clip
                if index < 3:
                    preview_path = PREVIEW_DIR / (
                        f"{video_id}_frame_{index:02d}.jpg"
                    )
                    if not cv2.imwrite(str(preview_path), resized):
                        raise RuntimeError("Cannot save the frame preview.")

            if len(frame_values) < 2:
                raise ValueError("At least two sampled frames are required.")

            values = np.asarray(frame_values)
            names = [
                "red", "green", "blue",
                "brightness", "contrast", "edge_density"
            ]

            features = {}
            for column, name in enumerate(names):
                features[f"visual_{name}_mean"] = float(
                    values[:, column].mean()
                )
                features[f"visual_{name}_std"] = float(
                    values[:, column].std()
                )

            features["visual_temporal_change_mean"] = float(
                np.mean(changes)
            )
            features["visual_temporal_change_std"] = float(
                np.std(changes)
            )

            if not np.isfinite(list(features.values())).all():
                raise ValueError("Invalid feature values.")

            return features, len(frame_values)

        finally:
            cap.release()



    import numpy as np
    import pandas as pd

    # Confirm that the extraction function is available

    # Load the original prepared splits
    split_tables = {
        split: pd.read_csv(CSV_DIR / f"{split}_ready.csv")
        for split in ["train", "val", "test"]
    }
    all_clips = pd.concat(split_tables.values(), ignore_index=True)

    assert all_clips["Video_ID"].notna().all()
    assert all_clips["Video_ID"].is_unique

    # Define the expected feature columns
    base_names = [
        "red", "green", "blue",
        "brightness", "contrast", "edge_density"
    ]
    visual_cols = [
        f"visual_{name}_{stat}"
        for name in base_names
        for stat in ["mean", "std"]
    ] + [
        "visual_temporal_change_mean",
        "visual_temporal_change_std"
    ]

    visual_rows = []
    visual_report = []

    print("Total clips:", len(all_clips))

    for number, video_id in enumerate(
        all_clips["Video_ID"].astype(str), start=1
    ):
        try:
            values, sampled_frames = extract_visual_features(video_id)
            visual_rows.append({"Video_ID": video_id, **values})
            visual_report.append({
                "Video_ID": video_id,
                "status": "PASS",
                "sampled_frames": sampled_frames,
                "error": ""
            })

        except Exception as error:
            visual_report.append({
                "Video_ID": video_id,
                "status": "FAIL",
                "error": str(error)
            })
            print(f"FAIL: {video_id} | {error}")

        # Save progress after every clip
        checkpoint = FEATURE_DIR / "visual_features_checkpoint.csv"
        temporary = FEATURE_DIR / "visual_features_checkpoint.tmp.csv"

        pd.DataFrame(
            visual_rows, columns=["Video_ID", *visual_cols]
        ).to_csv(temporary, index=False)
        temporary.replace(checkpoint)

        pd.DataFrame(visual_report).to_csv(
            FEATURE_DIR / "visual_extraction_report.csv", index=False
        )

        if number % 25 == 0 or number == len(all_clips):
            print(
                f"Processed {number}/{len(all_clips)} | "
                f"Passed: {len(visual_rows)} | "
                f"Failed: {number - len(visual_rows)}"
            )

    # Save the complete successful feature table
    visual_df = pd.DataFrame(
        visual_rows, columns=["Video_ID", *visual_cols]
    )
    visual_df.to_csv(
        FEATURE_DIR / "visual_features_all.csv", index=False
    )

    # Save separate feature files in the original split order
    for split, labels in split_tables.items():
        split_features = labels[["Video_ID"]].merge(
            visual_df,
            on="Video_ID",
            how="inner",
            validate="one_to_one",
            sort=False
        )
        split_features.to_csv(
            FEATURE_DIR / f"{split}_visual_features.csv",
            index=False
        )
        print(f"{split}: {len(split_features)}/{len(labels)} clips")

    print("\nPassed:", len(visual_rows))
    print("Failed:", len(all_clips) - len(visual_rows))
    print("Feature table shape:", visual_df.shape)
    print("Saved in:", FEATURE_DIR)

    # Validate full coverage when no clips failed
    if len(visual_rows) == len(all_clips):
        assert visual_df["Video_ID"].is_unique
        assert set(visual_df["Video_ID"]) == set(all_clips["Video_ID"])
        assert np.isfinite(
            visual_df[visual_cols].to_numpy(dtype=float)
        ).all()
        print("Full coverage and finite feature checks: PASS")
    else:
        print("Review failed clips before model training.")


if __name__ == "__main__":
    run_stage(
        __file__, run,
        requires=['dataset/csvs/train_ready.csv', 'dataset/csvs/val_ready.csv', 'dataset/csvs/test_ready.csv', 'dataset/clips'],
        produces=['dataset/features/visual_features_*.csv', 'dataset/features/visual_extraction_report.csv', 'dataset/features/*_visual_features.csv', 'dataset/frame_previews/*.jpg'],
        external=(),
        originals=False,
    )
