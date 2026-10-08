"""Standalone E6 stage derived from notebook cell 116."""
from stage_runtime import stage_main

SPEC = {'name': '01_extract_audio', 'source_cell': 116, 'fields': {}, 'requires': [('project', 'dataset/external/asd_feat/audio_development/full_audio_development_manifest.csv'), ('project', 'dataset/external/asd_feat/audio_features.part01.tar.gz'), ('project', 'dataset/external/asd_feat/audio_features.part02.tar.gz'), ('project', 'dataset/external/asd_feat/audio_features.part03.tar.gz'), ('project', 'dataset/external/asd_feat/audio_features.part04.tar.gz'), ('project', 'dataset/external/asd_feat/audio_features.part05.tar.gz'), ('project', 'dataset/external/asd_feat/audio_features.part06.tar.gz'), ('project', 'dataset/external/asd_feat/audio_features.part07.tar.gz'), ('project', 'dataset/external/asd_feat/audio_features.part08.tar.gz'), ('project', 'dataset/external/asd_feat/audio_features.part09.tar.gz'), ('project', 'dataset/external/asd_feat/audio_features.part10.tar.gz'), ('project', 'dataset/external/asd_feat/audio_features.part11.tar.gz'), ('project', 'dataset/external/asd_feat/audio_features.part12.tar.gz'), ('project', 'dataset/external/asd_feat/audio_features.part13.tar.gz'), ('project', 'dataset/external/asd_feat/audio_features.part14.tar.gz'), ('project', 'dataset/external/asd_feat/audio_features.part15.tar.gz'), ('project', 'dataset/external/asd_feat/audio_features.part16.tar.gz'), ('project', 'dataset/external/asd_feat/audio_features.part17.tar.gz'), ('project', 'dataset/external/asd_feat/audio_features.part18.tar.gz'), ('project', 'dataset/external/asd_feat/audio_features.part19.tar.gz')], 'packages': ['numpy', 'pandas'], 'writes': True}

def run(args):
    from pathlib import Path, PurePosixPath
    import io
    import json
    import tarfile

    import numpy as np
    import pandas as pd

    ROOT = args.project
    DATA = ROOT / "dataset/external/asd_feat"
    MANIFEST_DIR = DATA / "audio_development"
    OUTPUT = args.output
    CHECKPOINTS = OUTPUT / "archive_outputs"

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(
        MANIFEST_DIR / "full_audio_development_manifest.csv"
    )

    required = {
        "filename", "recording_stem", "archive",
        "archive_member", "sub_id", "evaluation_fold", "target"
    }

    if not required.issubset(manifest.columns):
        raise ValueError("Required manifest columns are missing.")

    if manifest[list(required)].isna().any().any():
        raise ValueError("Manifest contains missing required values.")

    for column in ["filename", "recording_stem"]:
        if manifest[column].duplicated().any():
            raise ValueError(f"Duplicate manifest values: {column}")

    if not manifest["target"].isin([0, 1]).all():
        raise ValueError("Targets must be binary.")

    if manifest.groupby("sub_id")["evaluation_fold"].nunique().gt(1).any():
        raise ValueError("Participant fold assignments are inconsistent.")

    FEATURE_COLUMNS = [
        f"mel_{index:03d}_mean" for index in range(112)
    ] + [
        f"mel_{index:03d}_std" for index in range(112)
    ]

    MAX_FILE_BYTES = 64 * 1024 * 1024
    all_features = []
    all_reports = []
    processed = 0

    print(f"Selected development recordings: {len(manifest)}")
    print("Extracting 224 features per recording...")

    # Process archives sequentially and save each archive's results.
    for number in range(1, 20):
        archive_name = f"audio_features.part{number:02d}.tar.gz"
        archive_path = DATA / archive_name

        selected = manifest[manifest["archive"].eq(archive_name)]
        lookup = selected.set_index("archive_member")["filename"].to_dict()
        seen = set()
        feature_rows = []
        report_rows = []

        print(f"\nArchive {number}/19: {archive_name}", flush=True)

        with tarfile.open(archive_path, mode="r|gz") as archive:
            for member in archive:
                if not member.isfile() or member.name not in lookup:
                    continue

                if member.name in seen:
                    raise ValueError(f"Duplicate archive member: {member.name}")
                seen.add(member.name)

                filename = lookup[member.name]
                array = None
                mel = None

                row = {
                    "filename": filename,
                    "archive": archive_name,
                    "status": "FAIL",
                    "shape": "",
                    "error": "",
                }

                try:
                    if PurePosixPath(member.name).suffix.lower() != ".npy":
                        raise ValueError("Expected an NPY file.")

                    if member.size > MAX_FILE_BYTES:
                        raise ValueError("Array exceeds the 64 MiB limit.")

                    source = archive.extractfile(member)
                    if source is None:
                        raise ValueError("Could not read archive member.")

                    with source:
                        payload = source.read(MAX_FILE_BYTES + 1)

                    if len(payload) != member.size:
                        raise ValueError("Archive member size mismatch.")

                    array = np.load(io.BytesIO(payload), allow_pickle=False)
                    del payload

                    row["shape"] = str(tuple(array.shape))
                    row["dtype"] = str(array.dtype)

                    if array.ndim != 2 or array.dtype.kind not in "fiu":
                        raise ValueError("Expected a 2D numeric array.")

                    bin_axes = [
                        axis for axis, size in enumerate(array.shape)
                        if size == 112
                    ]

                    if len(bin_axes) != 1:
                        raise ValueError(
                            f"Cannot identify a unique 112-bin axis: {array.shape}"
                        )

                    mel = array if bin_axes[0] == 0 else array.T

                    if mel.shape[1] < 1:
                        raise ValueError("No time steps found.")

                    if not np.isfinite(mel).all():
                        raise ValueError("Array contains non-finite values.")

                    # Summarize each mel bin over time without fitting a transform.
                    values = np.concatenate([
                        np.mean(mel, axis=1, dtype=np.float64),
                        np.std(mel, axis=1, dtype=np.float64, ddof=0),
                    ])

                    if not np.isfinite(values).all():
                        raise ValueError("Extracted features are not finite.")

                    feature_rows.append({
                        "filename": filename,
                        **dict(zip(FEATURE_COLUMNS, values.tolist())),
                    })

                    row.update({
                        "status": "PASS",
                        "time_steps": int(mel.shape[1]),
                        "all_zero_array": bool(np.all(mel == 0)),
                    })

                except Exception as error:
                    row["error"] = f"{type(error).__name__}: {error}"
                    print(f"FAIL: {filename} | {row['error']}", flush=True)

                finally:
                    array = None
                    mel = None

                report_rows.append(row)
                processed += 1

                if processed % 50 == 0:
                    print(
                        f"Processed {processed}/{len(manifest)}",
                        flush=True,
                    )

        for member_name in sorted(set(lookup) - seen):
            report_rows.append({
                "filename": lookup[member_name],
                "archive": archive_name,
                "status": "FAIL",
                "error": "Selected archive member not found.",
            })

        archive_features = pd.DataFrame(
            feature_rows, columns=["filename", *FEATURE_COLUMNS]
        )
        archive_report = pd.DataFrame(
            report_rows,
            columns=[
                "filename", "archive", "status", "shape", "error",
                "dtype", "time_steps", "all_zero_array"
            ],
        )

        archive_features.to_csv(
            CHECKPOINTS / f"part{number:02d}_features.csv", index=False
        )
        archive_report.to_csv(
            CHECKPOINTS / f"part{number:02d}_report.csv", index=False
        )

        all_features.append(archive_features)
        all_reports.append(archive_report)

        passed = int(archive_report["status"].eq("PASS").sum())
        failed = int(archive_report["status"].eq("FAIL").sum())
        print(f"Archive saved | Passed: {passed} | Failed: {failed}", flush=True)

    features = pd.concat(all_features, ignore_index=True)
    report = pd.concat(all_reports, ignore_index=True)

    if report["filename"].duplicated().any():
        raise ValueError("Duplicate extraction report IDs.")

    if set(report["filename"]) != set(manifest["filename"]):
        raise ValueError("Extraction report coverage differs from the manifest.")

    if features["filename"].duplicated().any():
        raise ValueError("Duplicate feature IDs.")

    passed_ids = set(report.loc[report["status"].eq("PASS"), "filename"])
    if set(features["filename"]) != passed_ids:
        raise ValueError("Features and successful report IDs differ.")

    if not np.isfinite(features[FEATURE_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Final feature table contains non-finite values.")

    # Align the final feature table and successful metadata by filename.
    successful = manifest[
        manifest["filename"].isin(passed_ids)
    ].sort_values("filename").reset_index(drop=True)

    features = successful[["filename"]].merge(
        features,
        on="filename",
        how="left",
        validate="one_to_one",
    )

    features.to_csv(OUTPUT / "development_audio_features.csv", index=False)
    report.to_csv(OUTPUT / "audio_extraction_report.csv", index=False)
    successful.to_csv(OUTPUT / "successful_development_manifest.csv", index=False)

    summary = {
        "selected_development_recordings": len(manifest),
        "passed": len(features),
        "failed": int(report["status"].eq("FAIL").sum()),
        "features_per_recording": len(FEATURE_COLUMNS),
        "participants": int(successful["sub_id"].nunique()),
        "positive_recordings": int(successful["target"].sum()),
        "positive_participants": int(
            successful.loc[successful["target"].eq(1), "sub_id"].nunique()
        ),
        "aggregation": "Per-bin temporal mean and population standard deviation",
        "scope": "Development-only audio extraction; no scaling, PCA or training",
    }

    (OUTPUT / "extraction_evidence.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"\nPassed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Feature table shape: {features.shape}")
    print(f"Participants: {summary['participants']}")
    print(f"Positive recordings: {summary['positive_recordings']}")
    print(f"Positive participants: {summary['positive_participants']}")
    print(f"Saved in: {OUTPUT}")
    print("Development audio extraction complete. No model training performed.")

if __name__ == "__main__":
    stage_main(__file__, run, SPEC)
