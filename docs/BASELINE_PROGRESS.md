# Baseline Experiment Progress

## Project
Explainable Multimodal AI for Autism-Related Behavioral Analysis.

## Completed work
- Cleaned and checked 928 AV-ASD annotations.
- Prepared and verified 527 audio-video clips.
- Preserved splits: 312 training, 108 validation, 107 test clips.
- Verified zero clip and source-video overlap across splits.
- Extracted 44 audio and 14 basic visual features per clip.
- Confirmed 7 clips have entirely zero audio samples.
- Recovered 6 initially failed visual extractions.
- Created aligned feature and target tables with 58 input features.
- Trained audio, visual, and combined Logistic Regression baselines.
- Trained a combined Random Forest baseline.
- Evaluated equal-weight audio-visual probability late fusion.
- Selected thresholds using validation Macro F1.
- Completed held-out test evaluation with frozen configurations.
- Generated exact linear contribution explanations and charts.

## Evaluation protocol
Primary baseline: `combined_lr`.

Models were fitted on the training split only. Logistic Regression
scalers were fitted on training features only. Thresholds were selected
on validation data before test evaluation. Test results were not used
to change the primary model or thresholds.

## Test results
| model | threshold | macro_f1 | micro_f1 | precision | recall | hamming_loss | exact_match_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| audio_lr | 0.3000 | 0.2854 | 0.3432 | 0.2351 | 0.6354 | 0.4364 | 0.0093 |
| visual_lr | 0.3000 | 0.3090 | 0.3499 | 0.2162 | 0.9167 | 0.6112 | 0.0000 |
| combined_lr | 0.3000 | 0.2937 | 0.3544 | 0.2516 | 0.5990 | 0.3916 | 0.0093 |
| combined_rf | 0.3000 | 0.2239 | 0.3348 | 0.2960 | 0.3854 | 0.2748 | 0.0467 |
| late_fusion_lr | 0.4000 | 0.3111 | 0.3602 | 0.2490 | 0.6510 | 0.4150 | 0.0093 |

## Explainability
The primary Logistic Regression model was explained using signed
feature contributions to logits relative to training feature means.
The reconstruction checks passed.

These contributions are not SHAP values and do not establish causality.
Model probability scores are not accuracy percentages.

## Limitations
- This is an available-media subset of the original annotations.
- Features describe basic sound and visual appearance; they do not
  directly encode gaze, pose, speech meaning, or specific behaviors.
- Temporal-change features can reflect camera movement or scene cuts.
- Five short clips required first/last-frame sampling.
- Seven silent clips were retained; the test split contains none.
- Background has only 7 training positives and 1 validation positive.
- False positives remain frequent and performance is limited.
- Repeated validation comparisons may overestimate selected scores.
- Source-video separation does not establish participant separation.
- Grouped cross-validation and uncertainty estimates remain pending.

## Remaining work
- Review per-label test errors and estimate uncertainty.
- Assess robustness and explanation stability.
- Build a research demonstration dashboard.
- Consolidate reusable scripts and documentation.

This project is intended for behavioral research, not clinical diagnosis.
