"""Notebook-derived standalone stage. Source cells: [66]."""
from script_runtime import run_stage


def run(PROJECT, original_videos=None):
    from pathlib import Path

    import pandas as pd


    RESULTS = PROJECT / "results"

    folders = {
        "Lite": RESULTS / "pose_quality_audit",
        "Full": RESULTS / "pose_quality_audit_full",
    }

    reports = {
        name: pd.read_csv(folder / "pose_pilot_report.csv")
        for name, folder in folders.items()
    }

    # Verify that both models processed the same clips.
    lite_ids = set(reports["Lite"]["Video_ID"])
    full_ids = set(reports["Full"]["Video_ID"])

    if lite_ids != full_ids:
        raise ValueError("Lite and Full clip IDs differ.")

    rows = []

    for name, report in reports.items():
        if report["Video_ID"].duplicated().any():
            raise ValueError(f"{name}: Duplicate clip IDs.")

        if not report["successful_frames"].eq(5).all():
            raise ValueError(f"{name}: Some sampled frames failed.")

        total = int(report["successful_frames"].sum())
        detected = int(report["frames_with_pose"].sum())
        confident = int(report["upper_body_confident_frames"].sum())

        rows.append({
            "Model": name,
            "Clips": len(report),
            "Processed_frames": total,
            "Pose_detected_frames": detected,
            "Pose_detection_percent": 100 * detected / total,
            "Both_arms_confident_frames": confident,
            "Both_arms_confident_percent": 100 * confident / total,
            "Total_clip_processing_seconds": report["elapsed_seconds"].sum(),
            "Max_sampled_python_rss_mb": (
                report["python_peak_sampled_rss_mb"].max()
            ),
        })

    comparison = pd.DataFrame(rows)
    path = RESULTS / "pose_model_comparison.csv"
    comparison.to_csv(path, index=False)

    print(comparison.round(2).to_string(index=False))
    print("\nSaved:", path)


if __name__ == "__main__":
    run_stage(
        __file__, run,
        requires=['results/pose_quality_audit/pose_pilot_report.csv', 'results/pose_quality_audit_full/pose_pilot_report.csv'],
        produces=['results/pose_model_comparison.csv'],
        external=(),
        originals=False,
    )
