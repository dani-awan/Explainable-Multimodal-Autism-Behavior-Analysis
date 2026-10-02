"""Notebook-derived standalone stage. Source cells: [29]."""
from script_runtime import run_stage


def run(PROJECT, original_videos=None):
    from pathlib import Path
    import json
    import re
    import shutil
    import subprocess
    import pandas as pd

    project = PROJECT
    video_folder = original_videos
    ffprobe = shutil.which("ffprobe")

    if ffprobe is None:
        raise RuntimeError("FFprobe was not found.")

    df = pd.read_csv(project / "dataset/csvs/dataset_clean.csv")

    # Find completed video files and exclude temporary format files.
    video_files = {}
    for p in video_folder.iterdir():
        if (
            p.is_file()
            and p.suffix.lower() in {".mp4", ".mkv", ".webm"}
            and p.stat().st_size > 0
            and not re.search(r"\.f\d+$", p.stem)
        ):
            video_files.setdefault(p.stem, []).append(p)

    metadata = {}

    # Check the duration and streams of each available video.
    for number, (source_id, paths) in enumerate(video_files.items(), 1):
        candidates = []

        for path in paths:
            try:
                result = subprocess.run(
                    [
                        ffprobe, "-v", "error",
                        "-show_entries", "format=duration:stream=codec_type",
                        "-of", "json", str(path)
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=True
                )

                info = json.loads(result.stdout)
                duration = float(info.get("format", {}).get("duration", 0))
                streams = {
                    stream.get("codec_type")
                    for stream in info.get("streams", [])
                }

                if duration > 0 and "video" in streams:
                    candidates.append({
                        "path": str(path),
                        "duration": duration,
                        "audio": "audio" in streams
                    })

            except (subprocess.SubprocessError, ValueError, TypeError):
                pass

        # Prefer a video with audio when multiple formats are available.
        metadata[source_id] = (
            max(candidates, key=lambda item: (item["audio"], item["duration"]))
            if candidates else None
        )

        if number % 25 == 0:
            print(f"Checked {number}/{len(video_files)} video IDs")

    records = []

    # Validate clip timestamps against the original video duration.
    for video_id in df["Video_ID"].astype(str):
        record = {"Video_ID": video_id, "status": ""}

        try:
            source_id, start, end = video_id.rsplit("_", 2)
            start, end = float(start), float(end)
            record.update(Source_ID=source_id, start=start, end=end)

            if start < 0 or end <= start:
                record["status"] = "invalid_clip_times"

            elif source_id not in metadata:
                record["status"] = "missing_video"

            elif metadata[source_id] is None:
                record["status"] = "metadata_check_failed"

            else:
                meta = metadata[source_id]
                record.update(meta)

                if end > meta["duration"]:
                    record["status"] = "clip_exceeds_video_duration"

                elif not meta["audio"]:
                    record["status"] = "video_without_audio"

                else:
                    record["status"] = "metadata_ok"

        except ValueError:
            record["status"] = "invalid_Video_ID"

        records.append(record)

    # Save the validation report without modifying the original videos.
    report = pd.DataFrame(records)
    output = project / "dataset/csvs/video_check_report.csv"
    report.to_csv(output, index=False)

    print("\nClip check results:")
    print(report["status"].value_counts().to_string())
    print("\nSaved:", output)


if __name__ == "__main__":
    run_stage(
        __file__, run,
        requires=['dataset/csvs/dataset_clean.csv'],
        produces=['dataset/csvs/video_check_report.csv'],
        external=('ffprobe',),
        originals=True,
    )
