# Reproducibility Status

## Environment snapshots

- requirements-main-snapshot.txt: installed versions from the notebook environment.
- requirements-pose-snapshot.txt: installed versions from the pose environment.
- docs/reproducibility_inventory.json: Python versions and script inventory.

These files record installed versions. They are not dependency lockfiles.
A fresh installation has not yet been verified. The main environment may
contain packages unrelated to this project.

## Runtime separation

- Main environment: audio features, handcrafted visual features,
  model evaluation, explainability, and Streamlit.
- Pose environment: MediaPipe pose and visual embedding scripts.
- FFmpeg and FFprobe are external dependencies and must be installed separately.

## Remaining verification

- Identify processing steps that still exist only in notebook cells.
- Convert those steps into standalone scripts with project-relative paths.
- Preserve saved fold assignments, feature order, labels, and thresholds.
- Verify installation in a clean environment.
- Verify pipeline execution before claiming full reproducibility.

Python syntax checks do not verify runtime behavior or scientific results.

[Windows validation checks](WINDOWS_VALIDATION_CHECKS.md)
