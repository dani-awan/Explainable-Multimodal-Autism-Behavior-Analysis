"""Notebook-derived standalone stage. Source cells: [31]."""
from script_runtime import run_stage


def run(PROJECT, original_videos=None):
    from pathlib import Path
    import json
    import shutil
    import subprocess
    import pandas as pd

    project = PROJECT
    output_folder = project / "dataset/clips"
    output_folder.mkdir(parents=True, exist_ok=True)

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("FFmpeg and FFprobe must be available.")

    report = pd.read_csv(project / "dataset/csvs/video_check_report.csv")

    # Process all clips that passed metadata validation.
    clips = report[report["status"] == "metadata_ok"]
    results = []

    for row in clips.itertuples(index=False):
        output = output_folder / f"{row.Video_ID}.mp4"
        temporary = output_folder / f"{row.Video_ID}.partial.mp4"
        expected_duration = float(row.end - row.start)

        try:
            # Create an accurately trimmed clip using one encoding thread.
            if not output.exists():
                command = [
                    ffmpeg, "-hide_banner", "-loglevel", "error",
                    "-nostdin", "-y", "-xerror",
                    "-ss", str(row.start),
                    "-i", str(row.path),
                    "-t", str(expected_duration),
                    "-map", "0:v:0",
                    "-map", "0:a:0",
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-crf", "23",
                    "-threads", "1",
                    "-c:a", "aac",
                    "-b:a", "128k",
                    "-movflags", "+faststart",
                    str(temporary)
                ]
                subprocess.run(
                    command, capture_output=True, text=True,
                    check=True, timeout=1800
                )

            candidate = output if output.exists() else temporary

            # Verify the output duration and required audio/video streams.
            probe = subprocess.run(
                [
                    ffprobe, "-v", "error",
                    "-show_entries", "format=duration:stream=codec_type",
                    "-of", "json", str(candidate)
                ],
                capture_output=True, text=True,
                check=True, timeout=30
            )
            info = json.loads(probe.stdout)
            actual_duration = float(info["format"]["duration"])
            streams = {s["codec_type"] for s in info["streams"]}

            if not {"video", "audio"}.issubset(streams):
                raise ValueError("Output is missing video or audio.")

            if abs(actual_duration - expected_duration) > 0.3:
                raise ValueError(
                    f"Duration mismatch: expected {expected_duration:.2f}s, "
                    f"received {actual_duration:.2f}s"
                )

            # Rename the temporary file only after validation succeeds.
            if candidate == temporary:
                temporary.replace(output)

            results.append((row.Video_ID, "PASS"))
            print(f"PASS: {row.Video_ID} | {actual_duration:.2f}s")

        except Exception as error:
            details = (
                error.stderr
                if isinstance(error, subprocess.CalledProcessError)
                else str(error)
            )
            results.append((row.Video_ID, "FAIL"))
            print(f"FAIL: {row.Video_ID} | {details}")

    print("\nPassed:", sum(status == "PASS" for _, status in results))
    print("Failed:", sum(status == "FAIL" for _, status in results))
    print("Clip folder:", output_folder)

    pd.DataFrame(results, columns=["Video_ID", "status"]).to_csv(project / "dataset/csvs/clip_extraction_report.csv", index=False)
    if any(status != "PASS" for _, status in results):
        raise RuntimeError("Some clips failed extraction or validation.")


if __name__ == "__main__":
    run_stage(
        __file__, run,
        requires=['dataset/csvs/video_check_report.csv'],
        produces=['dataset/csvs/clip_extraction_report.csv'],
        external=('ffmpeg', 'ffprobe'),
        originals=False,
    )
