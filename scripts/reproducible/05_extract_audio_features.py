"""Notebook-derived standalone stage. Source cells: [35, 36]."""
from script_runtime import run_stage


def run(PROJECT, original_videos=None):
    from pathlib import Path
    import shutil
    import subprocess
    import numpy as np
    import pandas as pd
    import librosa
    import soundfile as sf

    # Define the project folders

    CSV_DIR = PROJECT / "dataset" / "csvs"
    CLIP_DIR = PROJECT / "dataset" / "clips"
    AUDIO_DIR = PROJECT / "dataset" / "audio"
    FEATURE_DIR = PROJECT / "dataset" / "features"

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    FFMPEG = shutil.which("ffmpeg")
    if FFMPEG is None:
        raise RuntimeError("FFmpeg was not found.")

    # Load all prepared splits without changing their membership
    split_tables = {
        split: pd.read_csv(CSV_DIR / f"{split}_ready.csv")
        for split in ["train", "val", "test"]
    }

    sample = pd.concat(split_tables.values(), ignore_index=True)

    if sample["Video_ID"].duplicated().any():
        raise ValueError("Duplicate Video_ID values found across splits.")

    print("Total clips to process:", len(sample))

    features = []
    report = []

    for video_id in sample["Video_ID"].astype(str):
        clip_path = CLIP_DIR / f"{video_id}.mp4"
        audio_path = AUDIO_DIR / f"{video_id}.wav"
        temp_path = AUDIO_DIR / f"{video_id}.tmp.wav"

        try:
            if not clip_path.is_file():
                raise FileNotFoundError(f"Missing clip: {clip_path.name}")

            # Extract the first audio stream as 16 kHz mono PCM
            command = [
                FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error",
                "-y", "-threads", "1",
                "-i", str(clip_path),
                "-map", "0:a:0", "-vn",
                "-ac", "1", "-ar", "16000",
                "-c:a", "pcm_s16le",
                str(temp_path)
            ]

            result = subprocess.run(
                command, capture_output=True, text=True, timeout=300
            )

            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip())

            # Validate the extracted audio
            y, sr = sf.read(temp_path, dtype="float32")

            if sr != 16000 or y.ndim != 1 or len(y) < 512:
                raise ValueError("Unexpected sample rate, channels, or audio length.")

            if not np.isfinite(y).all():
                raise ValueError("Audio contains invalid values.")

            # Calculate frame-level audio features
            mfcc = librosa.feature.mfcc(
                y=y, sr=sr, n_mfcc=20,
                n_fft=512, hop_length=160, n_mels=40
            )
            rms = librosa.feature.rms(
                y=y, frame_length=512, hop_length=160
            )
            zcr = librosa.feature.zero_crossing_rate(
                y, frame_length=512, hop_length=160
            )

            # Summarize each feature over the full clip
            values = {}
            for index in range(20):
                values[f"mfcc_{index + 1}_mean"] = float(mfcc[index].mean())
                values[f"mfcc_{index + 1}_std"] = float(mfcc[index].std())

            for name, array in [("rms", rms), ("zcr", zcr)]:
                values[f"{name}_mean"] = float(array.mean())
                values[f"{name}_std"] = float(array.std())

            if not np.isfinite(list(values.values())).all():
                raise ValueError("Features contain invalid values.")

            # Save only validated audio and feature rows
            temp_path.replace(audio_path)
            features.append({"Video_ID": video_id, **values})

            duration = len(y) / sr
            silent = bool(np.max(np.abs(y)) == 0)

            report.append({
                "Video_ID": video_id,
                "status": "PASS",
                "duration_seconds": duration,
                "all_zero_audio": silent,
                "error": ""
            })

            print(
                f"PASS: {video_id} | {duration:.2f}s | "
                f"{len(values)} features | All-zero audio: {silent}"
            )

        except Exception as error:
            report.append({
                "Video_ID": video_id,
                "status": "FAIL",
                "error": str(error)
            })
            print(f"FAIL: {video_id} | {error}")

        finally:
            if temp_path.exists():
                temp_path.unlink()


    from pathlib import Path
    import pandas as pd


    CSV_DIR = PROJECT / "dataset" / "csvs"
    FEATURE_DIR = PROJECT / "dataset" / "features"
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    # Confirm that the extraction results are still in memory

    feature_df = pd.DataFrame(features)
    report_df = pd.DataFrame(report)

    # Validate the results before saving
    ready = pd.read_csv(CSV_DIR / "dataset_ready.csv")

    assert feature_df["Video_ID"].is_unique
    assert set(feature_df["Video_ID"]) == set(ready["Video_ID"])
    assert report_df["Video_ID"].is_unique
    assert set(report_df["Video_ID"]) == set(ready["Video_ID"])
    assert report_df["status"].eq("PASS").all()

    # Save the complete results
    feature_df.to_csv(
        FEATURE_DIR / "audio_features_all.csv", index=False
    )
    report_df.to_csv(
        FEATURE_DIR / "audio_extraction_report.csv", index=False
    )

    # Save each split in its original row order
    for split in ["train", "val", "test"]:
        labels = pd.read_csv(CSV_DIR / f"{split}_ready.csv")

        split_features = labels[["Video_ID"]].merge(
            feature_df,
            on="Video_ID",
            how="left",
            validate="one_to_one",
            sort=False
        )

        output = FEATURE_DIR / f"{split}_audio_features.csv"
        split_features.to_csv(output, index=False)
        print(f"Saved: {output.name} | {len(split_features)} clips")

    print("\nTotal feature table:", feature_df.shape)
    print("All final CSV files saved successfully.")


if __name__ == "__main__":
    run_stage(
        __file__, run,
        requires=['dataset/csvs/dataset_ready.csv', 'dataset/csvs/train_ready.csv', 'dataset/csvs/val_ready.csv', 'dataset/csvs/test_ready.csv', 'dataset/clips'],
        produces=['dataset/features/audio_features_all.csv', 'dataset/features/audio_extraction_report.csv', 'dataset/features/*_audio_features.csv', 'dataset/audio/*.wav'],
        external=('ffmpeg',),
        originals=False,
    )
