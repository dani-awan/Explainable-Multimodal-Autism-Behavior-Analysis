"""Notebook-derived standalone stage. Source cells: [16]."""
from script_runtime import run_stage


def run(PROJECT, original_videos=None):
    from pathlib import Path
    import re
    import pandas as pd

    folder = PROJECT / "dataset/csvs"
    names = ["dataset", "train", "val", "test"]
    cleaned = {}
    problems = []

    for name in names:
        path = folder / f"{name}.csv"
        df = pd.read_csv(path, dtype={"Video_ID": "string"})
        original_rows = len(df)

        # Strip surrounding whitespace from column names and text.
        df.columns = df.columns.str.strip()
        if df.columns.duplicated().any():
            raise ValueError(f"{name}: duplicate column names milay")

        for col in df.select_dtypes(include=["object", "string"]).columns:
            df[col] = df[col].map(
                lambda value: value.strip() if isinstance(value, str) else value
            )

        # Remove empty rows and exact duplicates.
        df = df.replace(r"^\s*$", pd.NA, regex=True)
        empty_rows = int(df.isna().all(axis=1).sum())
        df = df.dropna(how="all")
        duplicate_rows = int(df.duplicated().sum())
        df = df.drop_duplicates().reset_index(drop=True)

        if "Video_ID" not in df.columns:
            problems.append(f"{name}: Video_ID column nahi mila")
        else:
            missing_ids = int(df["Video_ID"].isna().sum())
            duplicate_ids = int(df["Video_ID"].duplicated().sum())
            if missing_ids or duplicate_ids:
                problems.append(
                    f"{name}: missing Video_ID={missing_ids}, "
                    f"duplicate Video_ID={duplicate_ids}"
                )

        labels = [col for col in df.columns if col != "Video_ID"]
        missing_labels = int(df[labels].isna().sum().sum())

        # Require binary target values.
        for col in labels:
            numeric = pd.to_numeric(df[col], errors="coerce")
            invalid = int((~numeric.isin([0, 1]) & df[col].notna()).sum())
            if invalid:
                problems.append(f"{name}: '{col}' mein {invalid} invalid values")
            df[col] = numeric

        if missing_labels:
            problems.append(f"{name}: {missing_labels} missing label values")

        cleaned[name] = df
        print(
            f"{name}: {original_rows} → {len(df)} rows | "
            f"empty removed={empty_rows} | exact duplicates removed={duplicate_rows}"
        )

    # Check clip and source-video overlap between splits.
    def source_video(video_id):
        match = re.fullmatch(r"(.+)_\d+_\d+", str(video_id))
        return match.group(1) if match else str(video_id)

    splits = ["train", "val", "test"]
    for i, first in enumerate(splits):
        for second in splits[i + 1:]:
            a = cleaned[first]["Video_ID"]
            b = cleaned[second]["Video_ID"]

            clip_overlap = len(set(a.dropna()) & set(b.dropna()))
            source_overlap = len(
                set(a.dropna().map(source_video))
                & set(b.dropna().map(source_video))
            )
            print(
                f"{first}/{second}: clip overlap={clip_overlap}, "
                f"source overlap={source_overlap}"
            )

            if clip_overlap or source_overlap:
                problems.append(f"{first}/{second}: split overlap mila")

    # Verify that dataset IDs match the combined split IDs.
    dataset_ids = set(cleaned["dataset"]["Video_ID"].dropna())
    split_ids = set(
        pd.concat([cleaned[s]["Video_ID"] for s in splits]).dropna()
    )
    print(
        "Dataset se splits mein missing clips:",
        len(dataset_ids - split_ids)
    )
    print(
        "Splits mein extra clips:",
        len(split_ids - dataset_ids)
    )

    if dataset_ids != split_ids:
        problems.append("dataset.csv aur train/val/test ke Video_ID match nahi karte")

    # Do not save cleaned files when validation fails.
    if problems:
        print("\nISSUES:")
        for problem in problems:
            print("-", problem)
        raise ValueError("CSV validation failed; cleaned files were not saved.")
    else:
        for name, df in cleaned.items():
            output = folder / f"{name}_clean.csv"
            df.to_csv(output, index=False)
            print("Saved:", output)

        print("\nBasic CSV cleaning aur checks complete.")


if __name__ == "__main__":
    run_stage(
        __file__, run,
        requires=['dataset/csvs/dataset.csv', 'dataset/csvs/train.csv', 'dataset/csvs/val.csv', 'dataset/csvs/test.csv'],
        produces=['dataset/csvs/*_clean.csv'],
        external=(),
        originals=False,
    )
