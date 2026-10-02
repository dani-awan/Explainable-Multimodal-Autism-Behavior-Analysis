"""Notebook-derived standalone stage. Source cells: [64]."""
from script_runtime import run_stage


def run(PROJECT, original_videos=None):
    from pathlib import Path

    import pandas as pd


    FOLDER = PROJECT / "results" / "pose_quality_audit"

    frames = pd.read_csv(FOLDER / "pose_pilot_frames.csv")
    joints = pd.read_csv(FOLDER / "pose_landmarks_audit.csv")

    keys = ["Video_ID", "frame_number"]

    if frames.duplicated(keys).any():
        raise ValueError("Duplicate frame records found.")

    # Use successfully processed frames with exactly one detected pose.
    single_frames = frames.loc[
        frames["status"].eq("PASS") & frames["pose_count"].eq(1),
        keys,
    ]

    single_joints = joints.merge(
        single_frames,
        on=keys,
        how="inner",
        validate="many_to_one",
    )

    if single_joints.duplicated(keys + ["joint_index"]).any():
        raise ValueError("Duplicate joints in single-pose frames.")

    names = {
        11: "Left shoulder",
        12: "Right shoulder",
        13: "Left elbow",
        14: "Right elbow",
        15: "Left wrist",
        16: "Right wrist",
    }

    # Recalculate the existing 0.7 confidence screen explicitly.
    single_joints["passes_screen"] = (
        single_joints["visibility"].ge(0.7)
        & single_joints["presence"].ge(0.7)
        & single_joints["raw_x"].between(0, 1)
        & single_joints["raw_y"].between(0, 1)
        & single_joints[["raw_x", "raw_y", "raw_z"]].notna().all(axis=1)
    )

    total_frames = int(frames["status"].eq("PASS").sum())
    single_count = len(single_frames)
    summary_rows = []

    for joint_index, name in names.items():
        subset = single_joints.loc[
            single_joints["joint_index"].eq(joint_index)
        ]
        count = int(subset["passes_screen"].sum())

        summary_rows.append({
            "Joint": name,
            "Confident_frames": count,
            "Percent_all_processed_frames": (
                100 * count / total_frames if total_frames else 0
            ),
            "Percent_single_pose_frames": (
                100 * count / single_count if single_count else 0
            ),
        })

    summary = pd.DataFrame(summary_rows)

    # Check each complete arm separately without tracking person identity.
    if single_count:
        reliable = single_joints.pivot(
            index=keys,
            columns="joint_index",
            values="passes_screen",
        ).reindex(columns=list(names), fill_value=False).fillna(False)

        left_arm = reliable[[11, 13, 15]].all(axis=1)
        right_arm = reliable[[12, 14, 16]].all(axis=1)

        arm_report = pd.DataFrame({
            "Check": ["Left arm", "Right arm", "At least one arm", "Both arms"],
            "Confident_frames": [
                int(left_arm.sum()),
                int(right_arm.sum()),
                int((left_arm | right_arm).sum()),
                int((left_arm & right_arm).sum()),
            ],
        })
    else:
        arm_report = pd.DataFrame({
            "Check": ["Left arm", "Right arm", "At least one arm", "Both arms"],
            "Confident_frames": [0, 0, 0, 0],
        })

    summary.to_csv(FOLDER / "upper_body_joint_coverage.csv", index=False)
    arm_report.to_csv(FOLDER / "arm_coverage.csv", index=False)

    print("Processed frames:", total_frames)
    print("Single-pose frames:", single_count)
    print("\nJoint coverage:")
    print(summary.round(2).to_string(index=False))
    print("\nArm coverage:")
    print(arm_report.to_string(index=False))
    print("\nCoverage reports saved.")


if __name__ == "__main__":
    run_stage(
        __file__, run,
        requires=['results/pose_quality_audit/pose_pilot_frames.csv', 'results/pose_quality_audit/pose_landmarks_audit.csv'],
        produces=['results/pose_quality_audit/upper_body_joint_coverage.csv', 'results/pose_quality_audit/arm_coverage.csv'],
        external=(),
        originals=False,
    )
