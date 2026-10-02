from pathlib import Path
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
from PIL import Image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


PROJECT = Path(__file__).resolve().parents[1]
CLIPS = PROJECT / "dataset" / "clips"
OUTPUT = PROJECT / "results" / "visual_embedding_pilot"
MODEL_DIR = PROJECT / "models" / "pretrained"

OUTPUT.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL = MODEL_DIR / "mobilenet_v3_small_embedder_v1.tflite"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "image_embedder/mobilenet_v3_small/float32/1/"
    "mobilenet_v3_small.tflite"
)

ffmpeg = shutil.which("ffmpeg")
ffprobe = shutil.which("ffprobe")

if not ffmpeg or not ffprobe:
    raise RuntimeError("FFmpeg and FFprobe are required.")

# Download the official versioned model.
if not MODEL.is_file():
    print("Downloading MobileNetV3 Small model...", flush=True)
    temporary = MODEL.with_suffix(".download")

    with urllib.request.urlopen(MODEL_URL, timeout=120) as response:
        with temporary.open("wb") as destination:
            shutil.copyfileobj(response, destination)

    temporary.replace(MODEL)


def get_duration(path):
    """Read the clip duration with FFprobe."""
    result = subprocess.run(
        [
            ffprobe, "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )

    duration = float(json.loads(result.stdout)["format"]["duration"])

    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid clip duration.")

    return duration


def read_frame(path, timestamp):
    """Decode one RGB frame while limiting memory use."""
    result = subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error",
            "-nostdin", "-threads", "1",
            "-ss", f"{timestamp:.6f}",
            "-i", str(path),
            "-map", "0:v:0",
            "-frames:v", "1",
            "-vf", "scale=512:512:force_original_aspect_ratio=decrease",
            "-an", "-c:v", "png", "-threads", "1",
            "-f", "image2pipe", "pipe:1",
        ],
        capture_output=True,
        check=True,
        timeout=60,
    )

    with Image.open(BytesIO(result.stdout)) as image:
        return np.ascontiguousarray(
            np.asarray(image.convert("RGB"))
        )


# Select five different source videos from the training split.
train = pd.read_csv(
    PROJECT / "dataset" / "csvs" / "train_ready.csv"
)

if train["Video_ID"].isna().any():
    raise ValueError("Missing training clip IDs.")

if train["Video_ID"].duplicated().any():
    raise ValueError("Duplicate training clip IDs.")

train["Original_ID"] = (
    train["Video_ID"].str.rsplit("_", n=2).str[0]
)

selected = (
    train.drop_duplicates("Original_ID")
    .sample(n=5, random_state=42)
    .reset_index(drop=True)
)

selected[["Video_ID", "Original_ID"]].to_csv(
    OUTPUT / "selected_training_clips.csv",
    index=False,
)

clip_paths = {}

for path in CLIPS.iterdir():
    if path.is_file() and path.suffix.lower() in {".mp4", ".mkv", ".webm"}:
        if path.stem in clip_paths:
            raise ValueError(f"Multiple files for clip: {path.stem}")
        clip_paths[path.stem] = path

options = vision.ImageEmbedderOptions(
    base_options=python.BaseOptions(
        model_asset_path=str(MODEL),
        delegate=python.BaseOptions.Delegate.CPU,
    ),
    running_mode=vision.RunningMode.IMAGE,
    l2_normalize=False,
    quantize=False,
)

process = psutil.Process()
report_rows = []
feature_rows = []
expected_dimension = None

with vision.ImageEmbedder.create_from_options(options) as embedder:
    for clip_id in selected["Video_ID"]:
        started = time.perf_counter()
        peak_rss = process.memory_info().rss
        vectors = []

        try:
            path = clip_paths.get(clip_id)

            if path is None:
                raise FileNotFoundError(f"Missing clip: {clip_id}")

            duration = get_duration(path)
            timestamps = duration * np.array(
                [0.1, 0.3, 0.5, 0.7, 0.9]
            )

            for timestamp in timestamps:
                pixels = read_frame(path, timestamp)

                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=pixels,
                )
                result = embedder.embed(image)

                if len(result.embeddings) != 1:
                    raise ValueError("Expected one embedding head.")

                vector = np.asarray(
                    result.embeddings[0].embedding,
                    dtype=np.float32,
                )

                if vector.ndim != 1 or vector.size == 0:
                    raise ValueError("Invalid embedding shape.")

                if not np.isfinite(vector).all():
                    raise ValueError("Non-finite embedding values.")

                if np.linalg.norm(vector) == 0:
                    raise ValueError("All-zero embedding.")

                if expected_dimension is None:
                    expected_dimension = int(vector.size)

                if vector.size != expected_dimension:
                    raise ValueError("Embedding dimensions differ.")

                vectors.append(vector)
                peak_rss = max(
                    peak_rss,
                    process.memory_info().rss,
                )

            matrix = np.stack(vectors)

            # Save ordered frame vectors for future sequence experiments.
            np.savez_compressed(
                OUTPUT / f"{clip_id}_frame_embeddings.npz",
                embeddings=matrix,
                timestamps_seconds=timestamps,
            )

            # Mean pooling creates a static clip representation.
            pooled = matrix.mean(axis=0)
            feature_row = {"Video_ID": clip_id}

            for index, value in enumerate(pooled):
                feature_row[f"visual_emb_{index:04d}"] = float(value)

            feature_rows.append(feature_row)

            report_rows.append({
                "Video_ID": clip_id,
                "status": "PASS",
                "duration_seconds": duration,
                "frames": len(vectors),
                "embedding_dimension": expected_dimension,
                "elapsed_seconds": time.perf_counter() - started,
                "python_peak_sampled_rss_mb": peak_rss / 1024**2,
                "error": "",
            })

            print(
                f"PASS: {clip_id} | {len(vectors)} frames | "
                f"{expected_dimension} features | "
                f"Python RAM: {peak_rss / 1024**2:.1f} MB",
                flush=True,
            )

        except Exception as error:
            report_rows.append({
                "Video_ID": clip_id,
                "status": "FAIL",
                "frames": len(vectors),
                "elapsed_seconds": time.perf_counter() - started,
                "error": str(error),
            })

            print(
                f"FAIL: {clip_id} | {error}",
                flush=True,
            )

        # Save progress after every clip.
        pd.DataFrame(report_rows).to_csv(
            OUTPUT / "embedding_pilot_report.csv",
            index=False,
        )

        if feature_rows:
            pd.DataFrame(feature_rows).to_csv(
                OUTPUT / "visual_embedding_pilot_features.csv",
                index=False,
            )

config = {
    "model_url": MODEL_URL,
    "model_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(),
    "mediapipe_version": mp.__version__,
    "sampling": "five frames at 10%, 30%, 50%, 70%, 90% of duration",
    "pooling": "mean of frame embeddings",
    "embedding_dimension": expected_dimension,
    "quantize": False,
    "l2_normalize_option": False,
    "scope": "training-only feasibility pilot",
    "memory_measure": "sampled Python RSS; excludes FFmpeg subprocess",
    "limitations": (
        "Whole-frame appearance includes backgrounds and text overlays. "
        "Embeddings are not gaze, pose, or behavior-specific measurements."
    ),
}

(OUTPUT / "embedding_pilot_config.json").write_text(
    json.dumps(config, indent=2),
    encoding="utf-8",
)

report = pd.DataFrame(report_rows)

print("\nPassed:", int(report["status"].eq("PASS").sum()))
print("Failed:", int(report["status"].eq("FAIL").sum()))
print("Feature table shape:", pd.DataFrame(feature_rows).shape)
print("\nPilot summary:")
print(report.round(3).to_string(index=False))
print("\nSaved in:", OUTPUT)