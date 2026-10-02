"""Notebook-derived standalone stage. Source cells: [57]."""
from script_runtime import run_stage


def run(PROJECT, original_videos=None):
    from pathlib import Path

    import numpy as np
    import pandas as pd


    OUTPUT = PROJECT / "results" / "error_analysis"
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Load the fixed test predictions and annotations.
    targets = pd.read_csv(
        PROJECT / "dataset" / "model_inputs" / "test_y.csv"
    ).set_index("Video_ID")

    predictions = pd.read_csv(
        PROJECT / "results" / "baselines"
        / "combined_lr_test_predictions.csv"
    ).set_index("Video_ID")

    if targets.index.has_duplicates or predictions.index.has_duplicates:
        raise ValueError("Duplicate clip IDs found.")

    if set(targets.index) != set(predictions.index):
        raise ValueError("Prediction and annotation IDs differ.")

    labels = targets.columns.tolist()
    predictions = predictions.loc[targets.index, labels]

    if not targets.isin([0, 1]).all().all():
        raise ValueError("Invalid annotation values.")

    if not predictions.isin([0, 1]).all().all():
        raise ValueError("Invalid prediction values.")

    actual = targets.to_numpy(dtype=int)
    predicted = predictions.to_numpy(dtype=int)

    # Remove the final start and end fields to recover source-video IDs.
    parts = targets.index.to_series().str.rsplit("_", n=2)

    if not parts.map(len).eq(3).all():
        raise ValueError("Unexpected clip ID format.")

    sources = parts.str[0].to_numpy()
    unique_sources = np.unique(sources)
    group_count = len(unique_sources)

    if group_count < 2:
        raise ValueError("At least two source-video groups are required.")

    # Store label-level counts for each source-video group.
    group_tp = []
    group_fp = []
    group_fn = []
    group_sizes = []

    for source in unique_sources:
        mask = sources == source
        a = actual[mask]
        p = predicted[mask]

        group_tp.append(((a == 1) & (p == 1)).sum(axis=0))
        group_fp.append(((a == 0) & (p == 1)).sum(axis=0))
        group_fn.append(((a == 1) & (p == 0)).sum(axis=0))
        group_sizes.append(int(mask.sum()))

    group_tp = np.asarray(group_tp)
    group_fp = np.asarray(group_fp)
    group_fn = np.asarray(group_fn)
    group_sizes = np.asarray(group_sizes)


    def calculate_metrics(weights):
        """Calculate metrics using source-group sampling weights."""
        tp = weights @ group_tp
        fp = weights @ group_fp
        fn = weights @ group_fn

        denominator = 2 * tp + fp + fn

        label_f1 = np.divide(
            2 * tp,
            denominator,
            out=np.zeros(len(labels), dtype=float),
            where=denominator != 0,
        )

        micro_denominator = denominator.sum()
        micro_f1 = (
            2 * tp.sum() / micro_denominator
            if micro_denominator > 0 else 0.0
        )

        sampled_clips = int(weights @ group_sizes)
        hamming_loss = (
            (fp.sum() + fn.sum()) / (sampled_clips * len(labels))
        )

        return [label_f1.mean(), micro_f1, hamming_loss]


    # Resample complete source-video groups with replacement.
    rng = np.random.default_rng(42)
    iterations = 2000
    bootstrap_scores = np.empty((iterations, 3))

    for iteration in range(iterations):
        sampled_groups = rng.integers(
            0, group_count, size=group_count
        )
        weights = np.bincount(
            sampled_groups, minlength=group_count
        )
        bootstrap_scores[iteration] = calculate_metrics(weights)

    point_scores = calculate_metrics(
        np.ones(group_count, dtype=int)
    )

    intervals = np.percentile(
        bootstrap_scores, [2.5, 97.5], axis=0
    )

    summary = pd.DataFrame({
        "Metric": ["Macro F1", "Micro F1", "Hamming loss"],
        "Test_score": point_scores,
        "CI_95_lower": intervals[0],
        "CI_95_upper": intervals[1],
        "Bootstrap_iterations": iterations,
        "Source_video_groups": group_count,
    })

    summary_path = OUTPUT / "combined_lr_group_bootstrap_ci.csv"
    summary.to_csv(summary_path, index=False)

    pd.DataFrame(
        bootstrap_scores,
        columns=["macro_f1", "micro_f1", "hamming_loss"],
    ).to_csv(
        OUTPUT / "combined_lr_group_bootstrap_samples.csv",
        index=False,
    )

    print("Test clips:", len(targets))
    print("Original-video groups:", group_count)
    print("\nSource-group bootstrap intervals:")
    print(summary.round(4).to_string(index=False))
    print("\nSaved:", summary_path)


if __name__ == "__main__":
    run_stage(
        __file__, run,
        requires=['results/baselines/combined_lr_test_predictions.csv', 'dataset/model_inputs/test_y.csv'],
        produces=['results/error_analysis/combined_lr_group_bootstrap*.csv'],
        external=(),
        originals=False,
    )
