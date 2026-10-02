"""Notebook-derived standalone stage. Source cells: [38]."""
from script_runtime import run_stage


def run(PROJECT, original_videos=None):
    from pathlib import Path
    import shutil
    import subprocess
    import numpy as np
    import pandas as pd


    CLIP_DIR = PROJECT / "dataset" / "clips"
    FEATURE_DIR = PROJECT / "dataset" / "features"

    FFMPEG = shutil.which("ffmpeg")
    if FFMPEG is None:
        raise RuntimeError("FFmpeg was not found.")

    # Select the seven clips flagged as silent
    report = pd.read_csv(FEATURE_DIR / "audio_extraction_report.csv")
    mask = report["all_zero_audio"].astype(str).str.lower().eq("true")
    silent_ids = report.loc[mask, "Video_ID"]

    checks = []

    for video_id in silent_ids:
        try:
            # Decode all channels without mixing them into mono
            result = subprocess.run(
                [
                    FFMPEG, "-nostdin", "-v", "error",
                    "-i", str(CLIP_DIR / f"{video_id}.mp4"),
                    "-map", "0:a:0", "-vn",
                    "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1"
                ],
                capture_output=True,
                timeout=120
            )

            if result.returncode != 0:
                raise RuntimeError(
                    result.stderr.decode(errors="replace").strip()
                )

            samples = np.frombuffer(result.stdout, dtype="<f4")

            if samples.size == 0 or not np.isfinite(samples).all():
                raise ValueError("Empty or invalid decoded audio.")

            peak = float(np.max(np.abs(samples)))
            status = "CONFIRMED_ZERO" if peak == 0 else "NONZERO_SOURCE"

            checks.append({
                "Video_ID": video_id,
                "status": status,
                "peak_amplitude": peak,
                "error": ""
            })

            print(f"{video_id}: {status} | Peak: {peak:.8f}")

        except Exception as error:
            checks.append({
                "Video_ID": video_id,
                "status": "CHECK_FAILED",
                "error": str(error)
            })
            print(f"{video_id}: CHECK_FAILED | {error}")

    # Save the independent silence check
    pd.DataFrame(checks).to_csv(
        FEATURE_DIR / "silent_audio_check.csv",
        index=False
    )

    print("\nSilent audio check saved.")


if __name__ == "__main__":
    run_stage(
        __file__, run,
        requires=['dataset/features/audio_extraction_report.csv', 'dataset/clips'],
        produces=['dataset/features/silent_audio_check.csv'],
        external=('ffmpeg',),
        originals=False,
    )
