"""Standalone E6 stage derived from notebook cell 120."""
from stage_runtime import stage_main

SPEC = {'name': '03_build_inputs', 'source_cell': 120, 'fields': {'audio_dir': 'dataset/external/asd_feat/derived/audio', 'rgb_dir': 'dataset/external/asd_feat/derived/rgb'}, 'requires': [('audio_dir', 'development_audio_features.csv'), ('audio_dir', 'successful_development_manifest.csv'), ('rgb_dir', 'development_rgb_features.csv'), ('rgb_dir', 'successful_development_manifest.csv'), ('project', 'dataset/external/asd_feat/metadata.tar.gz'), ('project', 'docs/ASD_FEAT_FULL_AUDIO_COVERAGE.json')], 'packages': ['numpy', 'pandas'], 'writes': True}

def run(args):
    from pathlib import Path, PurePosixPath
    import hashlib
    import json
    import tarfile

    import numpy as np
    import pandas as pd

    ROOT = args.project
    DATA = ROOT / "dataset/external/asd_feat"
    OUTPUT = args.output
    OUTPUT.mkdir(parents=True, exist_ok=True)

    audio_path = args.audio_dir / "development_audio_features.csv"
    rgb_path = args.rgb_dir / "development_rgb_features.csv"

    audio = pd.read_csv(audio_path)
    rgb = pd.read_csv(rgb_path)

    audio_manifest = pd.read_csv(
        args.audio_dir / "successful_development_manifest.csv"
    )
    rgb_manifest = pd.read_csv(
        args.rgb_dir / "successful_development_manifest.csv"
    )

    audio_columns = [
        f"mel_{index:03d}_mean" for index in range(112)
    ] + [
        f"mel_{index:03d}_std" for index in range(112)
    ]
    rgb_columns = [f"rgb_{index:04d}_mean" for index in range(1024)]
    behavior_columns = [
        "smile_rate", "smile_prop",
        "lookface_rate", "lookface_prop",
        "lookobject_rate", "lookobject_prop",
        "vocal_rate", "vocal_prop",
        "social_smile_rate", "social_smile_prop",
        "social_vocal_rate", "social_vocal_prop",
    ]

    # Load the human-coded development behavioral table.
    with tarfile.open(DATA / "metadata.tar.gz", "r:gz") as archive:
        members = [
            member for member in archive.getmembers()
            if member.isfile()
            and PurePosixPath(member.name).name
            == "hj_asdtrain_G_Aggregated.csv"
        ]

        if len(members) != 1:
            raise ValueError("Could not identify the development table uniquely.")

        with archive.extractfile(members[0]) as source:
            development = pd.read_csv(source)

    # Use the exact fold source recorded by the full audio coverage audit.
    coverage_evidence = json.loads(
        (ROOT / "docs/ASD_FEAT_FULL_AUDIO_COVERAGE.json")
        .read_text(encoding="utf-8")
    )
    oof_path = ROOT / coverage_evidence["fold_source"]
    oof = pd.read_csv(oof_path)

    tables = {
        "audio": audio,
        "RGB": rgb,
        "audio manifest": audio_manifest,
        "RGB manifest": rgb_manifest,
        "development": development,
        "OOF": oof,
    }

    for name, table in tables.items():
        if table["filename"].isna().any():
            raise ValueError(f"{name}: missing filenames.")
        if table["filename"].duplicated().any():
            raise ValueError(f"{name}: duplicate filenames.")

    ids = sorted(audio["filename"].tolist())
    id_set = set(ids)

    for name in ["RGB", "audio manifest", "RGB manifest"]:
        if set(tables[name]["filename"]) != id_set:
            raise ValueError(f"{name}: recording coverage differs from audio.")

    for name in ["development", "OOF"]:
        if not id_set.issubset(set(tables[name]["filename"])):
            raise ValueError(f"{name}: selected recordings are missing.")

    if audio.columns.tolist() != ["filename", *audio_columns]:
        raise ValueError("Unexpected audio feature schema.")

    if rgb.columns.tolist() != ["filename", *rgb_columns]:
        raise ValueError("Unexpected RGB feature schema.")

    # Align all inputs by recording ID rather than existing row order.
    aligned = {
        name: table.set_index("filename").loc[ids]
        for name, table in tables.items()
    }

    metadata_columns = ["sub_id", "visit", "evaluation_fold", "target"]

    for column in metadata_columns:
        if aligned["audio manifest"][column].isna().any():
            raise ValueError(f"Missing metadata: {column}")

        if not aligned["audio manifest"][column].equals(
            aligned["RGB manifest"][column]
        ):
            raise ValueError(f"Audio/RGB metadata disagreement: {column}")

    for column in ["sub_id", "visit"]:
        if not aligned["audio manifest"][column].equals(
            aligned["development"][column]
        ):
            raise ValueError(f"Development metadata disagreement: {column}")

    for column in ["sub_id", "evaluation_fold", "target"]:
        if not aligned["audio manifest"][column].equals(aligned["OOF"][column]):
            raise ValueError(f"Saved OOF metadata disagreement: {column}")

    # Confirm that the target matches the current-visit annotation.
    normalized = (
        aligned["development"]["asd_longitudinal"]
        .astype(str).str.lower()
        .str.replace(r"[\s_-]+", "", regex=True)
    )
    annotation_target = normalized.map({
        "asd": 1, "nonasd": 0, "notasd": 0
    })

    if annotation_target.isna().any():
        raise ValueError("Unrecognized development outcome labels.")

    metadata = aligned["audio manifest"][metadata_columns].copy()

    if not metadata["target"].isin([0, 1]).all():
        raise ValueError("Targets must be binary.")

    if not np.array_equal(
        annotation_target.to_numpy(dtype=int),
        metadata["target"].to_numpy(dtype=int),
    ):
        raise ValueError("Annotation and saved target disagreement.")

    if not metadata["evaluation_fold"].isin([1, 2, 3, 4, 5]).all():
        raise ValueError("Unexpected evaluation fold.")

    if metadata.groupby("sub_id")["evaluation_fold"].nunique().gt(1).any():
        raise ValueError("A participant appears in multiple evaluation folds.")

    branches = {
        "audio": aligned["audio"][audio_columns].copy(),
        "rgb": aligned["RGB"][rgb_columns].copy(),
        "behavior": aligned["development"][behavior_columns].copy(),
    }

    for name, table in branches.items():
        if not np.isfinite(table.to_numpy(dtype=float)).all():
            raise ValueError(f"{name}: missing or infinite features.")

    combined = pd.concat(
        [branches["audio"], branches["rgb"], branches["behavior"]],
        axis=1,
    )

    if combined.columns.duplicated().any():
        raise ValueError("Duplicate combined feature names.")

    if combined.shape[1] != 1260:
        raise ValueError("Expected 1260 combined features.")

    # Check both classes and participant separation in every saved fold.
    fold_rows = []

    for fold in range(1, 6):
        evaluation = metadata[metadata["evaluation_fold"].eq(fold)]
        fit = metadata[metadata["evaluation_fold"].ne(fold)]

        overlap = set(fit["sub_id"]) & set(evaluation["sub_id"])
        if overlap:
            raise ValueError(f"Fold {fold}: participant overlap.")

        if fit["target"].nunique() != 2 or evaluation["target"].nunique() != 2:
            raise ValueError(f"Fold {fold}: one class is missing.")

        fold_rows.append({
            "fold": fold,
            "fit_recordings": len(fit),
            "evaluation_recordings": len(evaluation),
            "evaluation_positive_recordings": int(evaluation["target"].sum()),
            "evaluation_positive_participants": evaluation.loc[
                evaluation["target"].eq(1), "sub_id"
            ].nunique(),
            "participant_overlap": len(overlap),
        })

    # Save feature matrices separately from targets and participant metadata.
    outputs = {}

    for name, table in {**branches, "combined": combined}.items():
        path = OUTPUT / f"development_{name}_X.csv"
        table.reset_index().to_csv(path, index=False)
        outputs[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{name}: {len(table)} recordings | {table.shape[1]} features")

    metadata.reset_index().to_csv(
        OUTPUT / "development_metadata.csv", index=False
    )
    metadata[["target"]].reset_index().to_csv(
        OUTPUT / "development_y.csv", index=False
    )

    fold_support = pd.DataFrame(fold_rows)
    fold_support.to_csv(OUTPUT / "fold_support.csv", index=False)

    schema = {
        "recordings": len(ids),
        "participants": int(metadata["sub_id"].nunique()),
        "positive_recordings": int(metadata["target"].sum()),
        "feature_groups": {
            "audio": audio_columns,
            "rgb": rgb_columns,
            "behavior": behavior_columns,
        },
        "combined_feature_count": combined.shape[1],
        "target": "Current-visit asd_longitudinal: ASD=1, non-ASD=0",
        "behavior_source": "Human-coded G aggregate development table",
        "fold_source": oof_path.relative_to(ROOT).as_posix(),
        "output_feature_sha256": outputs,
        "preprocessing": "Raw summaries; no fitted scaling or PCA",
        "scope": "Aligned development inputs only; no model training",
    }

    (OUTPUT / "input_schema.json").write_text(
        json.dumps(schema, indent=2), encoding="utf-8"
    )

    print("\nFold support:")
    print(fold_support.to_string(index=False))
    print(f"\nCombined feature table with filename: {(len(ids), 1261)}")
    print(f"Saved in: {OUTPUT}")
    print("Alignment, target verification, and participant separation: PASS")
    print("No model training or test processing performed.")

if __name__ == "__main__":
    stage_main(__file__, run, SPEC)
