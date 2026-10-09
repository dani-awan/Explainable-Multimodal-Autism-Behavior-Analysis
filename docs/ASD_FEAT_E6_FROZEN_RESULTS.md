# ASD-FEAT E6 Frozen Research Evaluation

## Model and Protocol

The selected research candidate combines 1024 pooled I3D RGB features and 12 human-coded behavioral features. Audio is excluded from this selected candidate. The broader E6 development study also evaluated audio and three-branch models.

The model was fitted on 1510 development recordings from 259 participants. It uses branch-specific PCA with 32 RGB components, standardized behavioral features and balanced logistic regression. The fixed threshold is 0.5. The observed mean development-fold ASD F1 for the selected candidate was 0.3142; this selection score is not an independent performance estimate for the selected winner.

## Recording-Level Official Test Results

Test support: 162 recordings, 30 participants, 12 positive recordings.

| model | threshold | macro_f1 | asd_f1 | precision | recall | specificity | balanced_accuracy | accuracy | roc_auc | average_precision | mcc | tp | fp | fn | tn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rgb_behavior_pca32_lr | 0.5000 | 0.6114 | 0.3571 | 0.2273 | 0.8333 | 0.7733 | 0.8033 | 0.7778 | 0.8728 | 0.2608 | 0.3572 | 10 | 34 | 2 | 116 |
| dummy_development_prior | 0.5000 | 0.4808 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.5000 | 0.9259 | 0.5000 | 0.0741 | 0.0000 | 0 | 0 | 12 | 150 |

The selected model identified 10 positive recordings, missed 2, and produced 34 false positives. The dummy reference predicts the development positive prevalence and has zero ASD F1 at threshold 0.5. Its high overall accuracy reflects class imbalance, not positive-case detection.

## Interpretation Limits

The 12 positive test recordings represent 1 participant(s). Repeated recordings are correlated; these recording-level results do not establish generalization across positive participants.

The official test had previously been evaluated for E1. It is not an untouched test set. The E6 candidate was selected after exploratory development comparisons, without nested selection validation. Human-coded behavioral features are required. Clinical diagnosis, disease-stage assessment and treatment recommendations are not validated.

## Verification and Evidence

Saved test metrics were recalculated from saved predictions and reproduced. Model fitting and threshold tuning were not performed during test evaluation or documentation. The frozen configuration and source model were preserved.

- Frozen model run: `results/asd_feat/e6_reproduction/12_freeze_rgb_behavior_model/20261008T192951_063340Z`
- Evaluation run: `results/asd_feat/e6_reproduction/15_evaluate_frozen_rgb_behavior/20261008T194502_516518Z`
- Model SHA-256: `2c8e9d9c126e9a7a8134eb9792550254de3747847b96aeaca1e396bae57327d4`
