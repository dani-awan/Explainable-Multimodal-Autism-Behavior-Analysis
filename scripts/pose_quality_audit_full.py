
from pathlib import Path
from io import BytesIO
import hashlib
import json
import shutil
import subprocess
import time
import urllib.request

import mediapipe as mp
import numpy as np
import pandas as pd
import psutil
from PIL import Image, ImageDraw
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

PROJECT = Path(__file__).resolve().parents[1]
CLIPS = PROJECT / "dataset" / "clips"
OUTPUT = PROJECT / "results" / "pose_quality_audit_full"
MODEL_DIR = PROJECT / "models" / "pretrained"
OUTPUT.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL = MODEL_DIR / "pose_landmarker_full.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/"
    "pose_landmarker_full.task"
)

ffmpeg = shutil.which("ffmpeg")
ffprobe = shutil.which("ffprobe")

if not ffmpeg or not ffprobe:
    raise RuntimeError("FFmpeg and FFprobe must be available.")

# Download the official model using a temporary file.
if not MODEL.is_file():
    print("Downloading the official Pose Landmarker Full model...", flush=True)
    temporary = MODEL.with_suffix(".download")
    with urllib.request.urlopen(MODEL_URL, timeout=120) as response:
        with temporary.open("wb") as destination:
            shutil.copyfileobj(response, destination)
    temporary.replace(MODEL)

train = pd.read_csv(PROJECT / "dataset" / "csvs" / "train_ready.csv")
train["Original_ID"] = train["Video_ID"].str.rsplit("_", n=2).str[0]

# Select clips from five different training source videos.
selected = (
    train.drop_duplicates("Original_ID")
    .sample(n=30, random_state=42)
    .reset_index(drop=True)
)

selected[["Video_ID", "Original_ID"]].to_csv(
    OUTPUT / "selected_training_clips.csv", index=False
)

clip_paths = {}
for path in CLIPS.iterdir():
    if path.is_file() and path.suffix.lower() in {".mp4", ".mkv", ".webm"}:
        if path.stem in clip_paths:
            raise ValueError(f"Multiple clip files found: {path.stem}")
        clip_paths[path.stem] = path

options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=str(MODEL),
        delegate=python.BaseOptions.Delegate.CPU,
    ),
    running_mode=vision.RunningMode.IMAGE,
    num_poses=2,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    output_segmentation_masks=False,
)

process = psutil.Process()
clip_rows = []
frame_rows = []
landmark_rows = []

with vision.PoseLandmarker.create_from_options(options) as detector:
    for clip_id in selected["Video_ID"]:
        started = time.perf_counter()
        peak_rss = process.memory_info().rss
        local_rows = []
        preview_saved = False

        try:
            path = clip_paths.get(clip_id)
            if path is None:
                raise FileNotFoundError(f"Clip missing: {clip_id}")

            probe = subprocess.run(
                [
                    ffprobe, "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "json", str(path),
                ],
                capture_output=True, text=True,
                check=True, timeout=30,
            )
            duration = float(json.loads(probe.stdout)["format"]["duration"])

            if not np.isfinite(duration) or duration <= 0:
                raise ValueError("Invalid clip duration.")

            # Sample five points across the clip, avoiding both endpoints.
            timestamps = duration * np.array([0.1, 0.3, 0.5, 0.7, 0.9])

            for frame_number, timestamp in enumerate(timestamps):
                try:
                    decoded = subprocess.run(
                        [
                            ffmpeg, "-hide_banner", "-loglevel", "error",
                            "-nostdin", "-threads", "1",
                            "-ss", f"{timestamp:.6f}",
                            "-i", str(path),
                            "-map", "0:v:0", "-frames:v", "1",
                            "-vf",
                            "scale=512:512:force_original_aspect_ratio=decrease",
                            "-an", "-c:v", "png", "-threads", "1",
                            "-f", "image2pipe", "pipe:1",
                        ],
                        capture_output=True, check=True, timeout=60,
                    )

                    with Image.open(BytesIO(decoded.stdout)) as source:
                        preview = source.convert("RGB")

                    pixels = np.ascontiguousarray(np.asarray(preview))
                    image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=pixels,
                    )

                    detected = detector.detect(image)
                    poses = detected.pose_landmarks
                    reliable_upper_body = False

                    for pose_index, pose in enumerate(poses):
                        reliable_indices = set()

                        for joint_index, landmark in enumerate(pose):
                            coordinates = np.array(
                                [landmark.x, landmark.y, landmark.z],
                                dtype=float,
                            )

                            reliable = bool(
                                np.isfinite(coordinates).all()
                                and landmark.visibility >= 0.7
                                and landmark.presence >= 0.7
                                and 0 <= landmark.x <= 1
                                and 0 <= landmark.y <= 1
                            )

                            if reliable:
                                reliable_indices.add(joint_index)

                            landmark_rows.append({
                                "Video_ID": clip_id,
                                "frame_number": frame_number,
                                "timestamp_seconds": float(timestamp),
                                "pose_index": pose_index,
                                "joint_index": joint_index,
                                "raw_x": landmark.x,
                                "raw_y": landmark.y,
                                "raw_z": landmark.z,
                                "visibility": landmark.visibility,
                                "presence": landmark.presence,
                                "reliable_joint": reliable,
                                "x": landmark.x if reliable else np.nan,
                                "y": landmark.y if reliable else np.nan,
                                "z": landmark.z if reliable else np.nan,
                                "multiple_pose_frame": len(poses) > 1,
                            })

                        # This is a confidence screen, not an accuracy guarantee.
                        if (
                            len(poses) == 1
                            and {11, 12, 13, 14, 15, 16}.issubset(
                                reliable_indices
                            )
                        ):
                            reliable_upper_body = True

                    # Count confident, in-frame landmarks for each detected pose.
                    visible_counts = [
                        sum(
                            landmark.visibility >= 0.5
                            and landmark.presence >= 0.5
                            and 0 <= landmark.x <= 1
                            and 0 <= landmark.y <= 1
                            for landmark in pose
                        )
                        for pose in poses
                    ]

                    row = {
                        "Video_ID": clip_id,
                        "frame_number": frame_number,
                        "timestamp_seconds": float(timestamp),
                        "status": "PASS",
                        "pose_count": len(poses),
                        "max_visible_landmarks": max(visible_counts, default=0),
                        "error": "",
                    }

                    row["single_pose_upper_body_confident"] = (
                        reliable_upper_body
                    )
                    row["ambiguous_multiple_poses"] = len(poses) > 1

                    # Save one local preview per clip for manual inspection.
                    if not preview_saved and poses:
                        drawing = ImageDraw.Draw(preview)
                        colors = ["lime", "orange"]

                        for pose_index, pose in enumerate(poses):
                            for landmark in pose:
                                if (
                                    landmark.visibility >= 0.5
                                    and landmark.presence >= 0.5
                                    and 0 <= landmark.x <= 1
                                    and 0 <= landmark.y <= 1
                                ):
                                    x = landmark.x * preview.width
                                    y = landmark.y * preview.height
                                    drawing.ellipse(
                                        (x - 3, y - 3, x + 3, y + 3),
                                        fill=colors[pose_index],
                                    )

                        preview.save(OUTPUT / f"{clip_id}_pose_preview.jpg")
                        preview_saved = True

                except Exception as error:
                    row = {
                        "Video_ID": clip_id,
                        "frame_number": frame_number,
                        "timestamp_seconds": float(timestamp),
                        "status": "FAIL",
                        "pose_count": 0,
                        "max_visible_landmarks": 0,
                        "error": str(error),
                    }

                local_rows.append(row)
                frame_rows.append(row)
                peak_rss = max(peak_rss, process.memory_info().rss)

            successful = [row for row in local_rows if row["status"] == "PASS"]
            detected_frames = sum(row["pose_count"] > 0 for row in successful)
            multiple_frames = sum(row["pose_count"] > 1 for row in successful)
            coverage = detected_frames / len(successful) if successful else np.nan
            confident_frames = sum(
                row.get("single_pose_upper_body_confident", False)
                for row in successful
            )


            clip_rows.append({
                "Video_ID": clip_id,
                "status": "PASS" if len(successful) == 5 else "REVIEW",
                "duration_seconds": duration,
                "successful_frames": len(successful),
                "frames_with_pose": detected_frames,
                "frames_with_multiple_poses": multiple_frames,
                "pose_detection_fraction": coverage,
                "upper_body_confident_frames": confident_frames,
                "upper_body_confident_fraction": (
                    confident_frames / len(successful)
                    if successful else np.nan
                ),
                "elapsed_seconds": time.perf_counter() - started,
                "python_peak_sampled_rss_mb": peak_rss / 1024**2,
                "error": "",
            })

            print(
                f"{clip_id} | Frames: {len(successful)}/5 | "
                f"Pose: {detected_frames}/{len(successful)} | "
                f"Multiple poses: {multiple_frames}",
                flush=True,
            )

        except Exception as error:
            clip_rows.append({
                "Video_ID": clip_id,
                "status": "FAIL",
                "error": str(error),
            })
            print(f"FAIL: {clip_id} | {error}", flush=True)

        # Save progress after each clip.
        pd.DataFrame(clip_rows).to_csv(
            OUTPUT / "pose_pilot_report.csv", index=False
        )
        pd.DataFrame(frame_rows).to_csv(
            OUTPUT / "pose_pilot_frames.csv", index=False
        )
        pd.DataFrame(landmark_rows).to_csv(
            OUTPUT / "pose_landmarks_audit.csv", index=False
        )


config = {
    "mediapipe_version": mp.__version__,
    "model_url": MODEL_URL,
    "model_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(),
    "sampling": "five points at 10%, 30%, 50%, 70%, 90% of duration",
    "mode": "independent images; no person tracking",
    "maximum_poses": 2,
    "selected_training_clips": 30,
    "joint_visibility_threshold": 0.7,
    "joint_presence_threshold": 0.7,
    "unreliable_coordinates": "NaN; raw coordinates retained for audit",
    "upper_body_screen": (
        "exactly one detected pose and all joints 11-16 pass confidence screen"
    ),
    "identity_limitation": (
        "pose_index is not a tracked identity; intended person not verified"
    ),
    "memory_measure": "sampled Python RSS; excludes FFmpeg subprocess",
    "scope": "training-only feasibility pilot; no classifier training",
}

(OUTPUT / "pose_pilot_config.json").write_text(
    json.dumps(config, indent=2), encoding="utf-8"
)

print("\nPilot summary:")
print(pd.DataFrame(clip_rows).round(3).to_string(index=False))
print("\nSaved in:", OUTPUT)
