"""Standalone E6 stage derived from notebook cell 119."""
from stage_runtime import stage_main

SPEC = {'name': '02_extract_rgb', 'source_cell': 119, 'fields': {}, 'requires': [('project', 'dataset/external/asd_feat/i3d_rgb.tar.gz'), ('project', 'dataset/external/asd_feat/rgb_pilot/audio_rgb_development_manifest.csv'), ('project', 'dataset/external/asd_feat/audio_development/full_audio_development_manifest.csv')], 'packages': ['numpy', 'pandas'], 'writes': True}

def run(args):
    from pathlib import Path
    import io
    import json
    import tarfile

    import numpy as np
    import pandas as pd

    ROOT = args.project
    DATA = ROOT / "dataset/external/asd_feat"
    ARCHIVE = DATA / "i3d_rgb.tar.gz"
    OUTPUT = args.output

    OUTPUT.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(
        DATA / "rgb_pilot/audio_rgb_development_manifest.csv"
    )


    # Recover visit metadata by verified recording ID, never by row position.
    reference = pd.read_csv(DATA / "audio_development/full_audio_development_manifest.csv")
    if reference["filename"].isna().any() or reference["filename"].duplicated().any():
        raise ValueError("Audio reference has missing or duplicate recording IDs.")
    if manifest["filename"].isna().any() or manifest["filename"].duplicated().any():
        raise ValueError("RGB manifest has missing or duplicate recording IDs.")
    reference = reference.set_index("filename")
    if not set(manifest["filename"]).issubset(reference.index):
        raise ValueError("RGB recording IDs are missing from the audio reference.")
    reference = reference.loc[manifest["filename"]].reset_index()
    for column in ["sub_id", "evaluation_fold", "target"]:
        if not manifest[column].reset_index(drop=True).equals(reference[column]):
            raise ValueError(f"RGB/audio reference disagreement: {column}")
    if "visit" in manifest and not manifest["visit"].reset_index(drop=True).equals(reference["visit"]):
        raise ValueError("RGB/audio reference disagreement: visit")
    if reference["visit"].isna().any():
        raise ValueError("Missing reference visit metadata.")
    manifest = manifest.copy()
    manifest["visit"] = reference["visit"].to_numpy()

    required = {
        "filename", "rgb_archive_member", "sub_id",
        "evaluation_fold", "target"
    }

    if not required.issubset(manifest.columns):
        raise ValueError("Required manifest columns are missing.")

    if manifest[list(required)].isna().any().any():
        raise ValueError("Missing required manifest values.")

    for column in ["filename", "rgb_archive_member"]:
        if manifest[column].duplicated().any():
            raise ValueError(f"Duplicate manifest values: {column}")

    if not manifest["target"].isin([0, 1]).all():
        raise ValueError("Targets must be binary.")

    if not manifest["evaluation_fold"].isin([1, 2, 3, 4, 5]).all():
        raise ValueError("Unexpected evaluation fold.")

    if manifest.groupby("sub_id")["evaluation_fold"].nunique().gt(1).any():
        raise ValueError("Participant fold assignments are inconsistent.")

    lookup = manifest.set_index("rgb_archive_member")["filename"].to_dict()
    feature_columns = [f"rgb_{index:04d}_mean" for index in range(1024)]

    MAX_FILE_BYTES = 64 * 1024 * 1024
    seen = set()
    feature_rows = []
    report_rows = []


    def save_checkpoint():
        # Save successful features and processing outcomes periodically.
        pd.DataFrame(
            feature_rows, columns=["filename", *feature_columns]
        ).to_csv(OUTPUT / "rgb_features_checkpoint.csv", index=False)

        pd.DataFrame(report_rows).to_csv(
            OUTPUT / "rgb_report_checkpoint.csv", index=False
        )


    print(f"Selected development recordings: {len(manifest)}")
    print("Extracting 1024 pooled RGB features per recording...")

    # Read only selected development arrays, one at a time.
    with tarfile.open(ARCHIVE, mode="r|gz") as archive:
        for member in archive:
            if not member.isfile() or member.name not in lookup:
                continue

            if member.name in seen:
                raise ValueError(f"Duplicate archive member: {member.name}")
            seen.add(member.name)

            filename = lookup[member.name]
            row = {
                "filename": filename,
                "status": "FAIL",
                "shape": "",
                "error": "",
            }

            array = None
            windows = None

            try:
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
                    raise ValueError("Expected a two-dimensional numeric array.")

                feature_axes = [
                    axis for axis, size in enumerate(array.shape)
                    if size == 1024
                ]

                if len(feature_axes) != 1:
                    raise ValueError(
                        f"Cannot identify a unique 1024-feature axis: {array.shape}"
                    )

                windows = array if feature_axes[0] == 1 else array.T

                if windows.shape[0] < 1:
                    raise ValueError("No visual windows found.")

                if not np.isfinite(windows).all():
                    raise ValueError("Array contains non-finite values.")

                # Pool over windows without fitting a scaler or PCA.
                values = np.mean(windows, axis=0, dtype=np.float64)

                if not np.isfinite(values).all():
                    raise ValueError("Pooled features contain non-finite values.")

                feature_rows.append({
                    "filename": filename,
                    **dict(zip(feature_columns, values.tolist())),
                })

                row.update({
                    "status": "PASS",
                    "windows": int(windows.shape[0]),
                    "all_zero_array": bool(np.all(windows == 0)),
                })

            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
                print(f"FAIL: {filename} | {row['error']}", flush=True)

            finally:
                array = None
                windows = None

            report_rows.append(row)

            if len(report_rows) % 100 == 0:
                save_checkpoint()
                print(
                    f"Processed {len(report_rows)}/{len(manifest)} | "
                    f"Passed: {len(feature_rows)} | "
                    f"Failed: {len(report_rows) - len(feature_rows)}",
                    flush=True,
                )

    for member_name in sorted(set(lookup) - seen):
        report_rows.append({
            "filename": lookup[member_name],
            "status": "FAIL",
            "error": "Selected archive member not found.",
        })

    features = pd.DataFrame(
        feature_rows, columns=["filename", *feature_columns]
    )
    report = pd.DataFrame(report_rows)

    if report["filename"].duplicated().any():
        raise ValueError("Duplicate extraction report IDs.")

    if set(report["filename"]) != set(manifest["filename"]):
        raise ValueError("Report coverage differs from the selected manifest.")

    passed_ids = set(report.loc[report["status"].eq("PASS"), "filename"])

    if features["filename"].duplicated().any():
        raise ValueError("Duplicate feature IDs.")

    if set(features["filename"]) != passed_ids:
        raise ValueError("Features and successful report IDs differ.")

    if not np.isfinite(features[feature_columns].to_numpy(dtype=float)).all():
        raise ValueError("Final visual features contain non-finite values.")

    successful = manifest[
        manifest["filename"].isin(passed_ids)
    ].sort_values("filename").reset_index(drop=True)

    features = successful[["filename"]].merge(
        features,
        on="filename",
        how="left",
        validate="one_to_one",
    )

    features.to_csv(OUTPUT / "development_rgb_features.csv", index=False)
    report.to_csv(OUTPUT / "rgb_extraction_report.csv", index=False)
    successful.to_csv(
        OUTPUT / "successful_development_manifest.csv", index=False
    )

    summary = {
        "selected_development_recordings": len(manifest),
        "passed": len(features),
        "failed": int(report["status"].eq("FAIL").sum()),
        "features_per_recording": len(feature_columns),
        "participants": int(successful["sub_id"].nunique()),
        "positive_recordings": int(successful["target"].sum()),
        "positive_participants": int(
            successful.loc[successful["target"].eq(1), "sub_id"].nunique()
        ),
        "aggregation": "Mean of each I3D RGB feature across released windows",
        "scope": "Development-only RGB pooling; no scaling, PCA or training",
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
    print("No model training or test-array processing performed.")

if __name__ == "__main__":
    stage_main(__file__, run, SPEC)
