# Stage order

Existing results do not need to be regenerated simply to install this package.

| Stage | Script | Notebook cells (zero-based) |
| --- | --- | --- |
| 01 | `01_clean_csvs.py` | 16 |
| 02 | `02_check_original_videos.py` | 29 |
| 03 | `03_extract_clips.py` | 31 |
| 04 | `04_build_ready_splits.py` | 32 |
| 05 | `05_extract_audio_features.py` | 35, 36 |
| 06 | `06_validate_audio_features.py` | 37 |
| 07 | `07_verify_silent_audio.py` | 38 |
| 08 | `08_extract_visual_features.py` | 40, 41 |
| 09 | `09_retry_visual_features.py` | 42 |
| 10 | `10_build_model_inputs.py` | 43 |
| 11 | `11_train_logistic_baselines.py` | 44 |
| 12 | `12_select_validation_thresholds.py` | 45 |
| 13 | `13_diagnose_validation_predictions.py` | 46 |
| 14 | `14_train_random_forest.py` | 47 |
| 15 | `15_build_late_fusion.py` | 48 |
| 16 | `16_evaluate_frozen_baselines.py` | 49 |
| 17 | `17_explain_linear_model.py` | 50 |
| 18 | `18_plot_explanations.py` | 51 |
| 19 | `19_analyze_test_errors.py` | 56 |
| 20 | `20_bootstrap_source_groups.py` | 57 |
| 21 | `21_grouped_cross_validation.py` | 58 |
| 22 | `22_summarize_ablation.py` | 59 |
| 23 | `23_summarize_pose_coverage.py` | 64 |
| 24 | `24_compare_pose_models.py` | 66 |
| 25 | `25_embedding_cross_validation.py` | 69 |
| 26 | `26_diagnose_embedding_predictions.py` | 70 |
| 27 | `27_pca_embedding_cross_validation.py` | 71 |
| 28 | `28_compare_audio_and_pca.py` | 72 |
| 29 | `29_check_modality_robustness.py` | 77 |
| 30 | `30_write_baseline_progress.py` | 52 |
| 31 | `31_write_visual_progress.py` | 73 |
| 32 | `32_write_robustness_progress.py` | 78 |
| 33 | `33_capture_environment_versions.py` | 79 |

## Processing groups

- Data preparation: 01–04. Stage 02 requires the original-video directory. Stage 03 validates existing clips and saves an extraction report.
- Audio: 05–07. Stage 06 is a read-only validation.
- Handcrafted visual: 08 followed by 09 when failures remain. Stage 09 intentionally updates stage 08 outputs, so use `--overwrite`. Its historical repair buffers resized decoded frames in memory; use it only for failed clips.
- Baseline model inputs and validation development: 10–15.
- Historical frozen test replay and explanations: 16–20. The recorded primary remains combined_lr; do not interpret this as fresh model selection.
- Training-only grouped CV and retrained ablation: 21–22. Existing fold_assignments.csv is reused.
- Pose audit reporting: 23–24; requires results from the existing pose scripts.
- Run the existing scripts/extract_train_visual_embeddings.py with .venv-pose if training embeddings are missing. Keep that script; it is not supplied or modified by this package.
- Training embedding and PCA experiments: 25–28; these reuse stage 21 folds.
- Frozen-model validation sensitivity: 29.
- Progress documents: 30–32. Stage 32 appends a link to the existing visual report and therefore needs `--overwrite` when that document exists.
- Environment version snapshots: 33, using the main interpreter and the existing pose interpreter.

## Before a rerun

Use `--check` on the desired stage. Inputs must exist and external tools must be found. A file-presence check does not validate CSV content. A missing upstream input is reported before heavy imports or processing.

If generated files already exist, execution stops unless `--overwrite` is supplied. This option permits the stage to update its listed artifacts; it does not change model settings. For experiments with different data, use a separate project copy and `--project`.

## Windows reproduction still required

Run checks on the actual laptop inputs before rerunning processing. Keep the current frozen artifacts. Training and feature scripts have not been proven to reproduce the laptop scores byte-for-byte. Capture the laptop dependency snapshots and retain saved folds, CSV order, and pretrained model hashes.
