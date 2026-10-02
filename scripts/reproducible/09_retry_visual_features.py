"""Notebook-derived standalone stage. Source cells: [42]."""
from script_runtime import run_stage


def run(PROJECT, original_videos=None):
    from pathlib import Path
    import shutil
    import subprocess
    import cv2
    import numpy as np
    import pandas as pd

    # Define the project folders

    CSV_DIR = PROJECT / "dataset" / "csvs"
    CLIP_DIR = PROJECT / "dataset" / "clips"
    FEATURE_DIR = PROJECT / "dataset" / "features"

    FFMPEG = shutil.which("ffmpeg")
    if FFMPEG is None:
        raise RuntimeError("FFmpeg was not found.")

    feature_path = FEATURE_DIR / "visual_features_all.csv"
    report_path = FEATURE_DIR / "visual_extraction_report.csv"

    existing = pd.read_csv(feature_path)
    old_report = pd.read_csv(report_path)
    failed_ids = old_report.loc[
        old_report["status"].eq("FAIL"), "Video_ID"
    ].tolist()

    # Preserve the original results before updating them
    for path in [feature_path, report_path]:
        backup = path.with_name(f"{path.stem}_before_retry.csv")
        if not backup.exists():
            shutil.copy2(path, backup)


    def retry_visual_features(video_id):
        """Decode sequentially with FFmpeg and handle short clips."""
        path = CLIP_DIR / f"{video_id}.mp4"

        # Read the dimensions and nominal frame rate
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                raise RuntimeError("Cannot open the clip.")

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        finally:
            cap.release()

        if not np.isfinite([fps, width, height]).all():
            raise ValueError("Invalid video metadata.")
        if min(fps, width, height) <= 0:
            raise ValueError("Invalid FPS or dimensions.")

        # Match the previous aspect-preserving resize
        scale = 224 / max(width, height)
        out_width = max(1, round(width * scale))
        out_height = max(1, round(height * scale))

        # Decode every frame without seeking or changing the frame rate
        result = subprocess.run(
            [
                FFMPEG, "-nostdin", "-v", "error", "-xerror",
                "-threads", "1", "-i", str(path),
                "-map", "0:v:0", "-an",
                "-vf", f"scale={out_width}:{out_height}:flags=area",
                "-fps_mode", "passthrough",
                "-pix_fmt", "bgr24", "-c:v", "rawvideo",
                "-threads", "1", "-f", "rawvideo", "pipe:1"
            ],
            capture_output=True,
            timeout=300
        )

        if result.returncode != 0:
            raise RuntimeError(
                result.stderr.decode(errors="replace").strip()
            )

        frame_bytes = out_width * out_height * 3
        if not result.stdout or len(result.stdout) % frame_bytes != 0:
            raise ValueError("Empty or incomplete decoded frame data.")

        frames = np.frombuffer(result.stdout, dtype=np.uint8).reshape(
            -1, out_height, out_width, 3
        )
        actual_count = len(frames)

        if actual_count < 2:
            raise ValueError("The clip contains fewer than two decoded frames.")

        # Sample approximately once per second using the decoded frame count
        targets = np.unique(
            np.rint(np.arange(0, actual_count / fps, 1.0) * fps).astype(int)
        )
        targets = targets[targets < actual_count]

        short_fallback = len(targets) < 2
        if short_fallback:
            targets = np.array([0, actual_count - 1])

        frame_values = []
        changes = []
        previous_gray = None

        for frame_number in targets:
            frame = frames[frame_number]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_float = gray.astype(np.float32) / 255.0
            edges = cv2.Canny(gray, 100, 200)

            frame_values.append([
                *(rgb.mean(axis=(0, 1)) / 255.0),
                float(gray_float.mean()),
                float(gray_float.std()),
                float((edges > 0).mean())
            ])

            if previous_gray is not None:
                changes.append(
                    float(np.abs(gray_float - previous_gray).mean())
                )
            previous_gray = gray_float

        values = np.asarray(frame_values)
        names = [
            "red", "green", "blue",
            "brightness", "contrast", "edge_density"
        ]

        features = {}
        for column, name in enumerate(names):
            features[f"visual_{name}_mean"] = float(values[:, column].mean())
            features[f"visual_{name}_std"] = float(values[:, column].std())

        features["visual_temporal_change_mean"] = float(np.mean(changes))
        features["visual_temporal_change_std"] = float(np.std(changes))

        if not np.isfinite(list(features.values())).all():
            raise ValueError("Invalid feature values.")

        return features, {
            "sampled_frames": len(targets),
            "decoded_frames": actual_count,
            "short_clip_fallback": short_fallback,
            "mean_sample_interval_seconds": float(np.diff(targets).mean() / fps)
        }


    # Retry only the failed clips
    repaired_rows = []
    retry_report = []

    for video_id in failed_ids:
        try:
            values, details = retry_visual_features(video_id)
            repaired_rows.append({"Video_ID": video_id, **values})
            retry_report.append({
                "Video_ID": video_id,
                "status": "PASS",
                "error": "",
                "decoder": "FFmpeg",
                **details
            })
            print(
                f"PASS: {video_id} | {details['sampled_frames']} frames | "
                f"Short fallback: {details['short_clip_fallback']}"
            )

        except Exception as error:
            retry_report.append({
                "Video_ID": video_id,
                "status": "FAIL",
                "error": str(error)
            })
            print(f"FAIL: {video_id} | {error}")

    # Merge repaired features with the successful results
    repaired = pd.DataFrame(repaired_rows)
    combined = pd.concat([existing, repaired], ignore_index=True)
    assert combined["Video_ID"].is_unique

    split_tables = {
        split: pd.read_csv(CSV_DIR / f"{split}_ready.csv")
        for split in ["train", "val", "test"]
    }
    all_labels = pd.concat(split_tables.values(), ignore_index=True)

    combined = all_labels[["Video_ID"]].merge(
        combined, on="Video_ID", how="inner",
        validate="one_to_one", sort=False
    )

    retry_df = pd.DataFrame(retry_report, columns=["Video_ID", "status", "error", "decoder", "sampled_frames", "decoded_frames", "short_clip_fallback", "mean_sample_interval_seconds"])
    updated_report = pd.concat(
        [
            old_report.loc[~old_report["Video_ID"].isin(failed_ids)],
            retry_df
        ],
        ignore_index=True
    )

    # Save the updated results and retry details
    combined.to_csv(feature_path, index=False)
    updated_report.to_csv(report_path, index=False)
    retry_df.to_csv(FEATURE_DIR / "visual_retry_report.csv", index=False)

    for split, labels in split_tables.items():
        output = labels[["Video_ID"]].merge(
            combined, on="Video_ID", how="inner",
            validate="one_to_one", sort=False
        )
        output.to_csv(
            FEATURE_DIR / f"{split}_visual_features.csv", index=False
        )
        print(f"{split}: {len(output)}/{len(labels)} clips")

    print("\nTotal passed:", len(combined))
    print("Still failed:", updated_report["status"].eq("FAIL").sum())
    print("Feature table shape:", combined.shape)

    feature_cols = combined.columns.drop("Video_ID")
    assert np.isfinite(combined[feature_cols].to_numpy(dtype=float)).all()

    if set(combined["Video_ID"]) == set(all_labels["Video_ID"]):
        print("Full coverage and finite feature checks: PASS")
    else:
        raise RuntimeError("Some clips still need review.")


if __name__ == "__main__":
    run_stage(
        __file__, run,
        requires=['dataset/features/visual_features_all.csv', 'dataset/features/visual_extraction_report.csv', 'dataset/clips', 'dataset/csvs/train_ready.csv', 'dataset/csvs/val_ready.csv', 'dataset/csvs/test_ready.csv'],
        produces=['dataset/features/visual_features_all.csv', 'dataset/features/visual_extraction_report.csv', 'dataset/features/visual_retry_report.csv', 'dataset/features/*_visual_features.csv'],
        external=(),
        originals=False,
    )
