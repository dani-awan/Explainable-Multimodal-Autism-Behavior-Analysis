# Visual Feature Experiments

## Dataset and evaluation

- Prepared clips: 527.
- Training / validation / test clips: 312 / 108 / 107.
- These experiments used the training split only.
- Training source-video groups: 185.
- Evaluation used the same five saved source-group folds.
- Source overlap between fitting and evaluation portions was zero.
- The prediction threshold remained fixed at 0.5.
- Source-video grouping does not establish participant independence.

## Pose feasibility

The Pose Landmarker Lite audit processed 150 sampled frames.
There were 112 single-pose frames. At least one complete confident
arm was available in 60 frames, and both arms in 29 frames.

The Full model audit found both confident arms in 22 of 150 frames.
It did not improve complete-arm coverage in this audit.

These are coverage measurements, not pose accuracy measurements.
Pose features were not added to the main classifier.

## Pretrained visual embeddings

- Extractor: MobileNetV3 Small ImageEmbedder pretrained on ImageNet.
- Sampling: five uniformly spaced frames per clip.
- Clip representation: mean of the five frame embeddings.
- Embedding dimension: 1024.
- Training extraction: 312 clips passed, zero failed.
- Mean pooling produces a static clip representation.
- These embeddings are visual features, not an independent third modality.
- Backgrounds, overlays, and scene appearance can affect the features.

## Raw embedding diagnostics

| model | actual_positive_labels | predicted_positive_labels | actual_labels_per_clip | predicted_labels_per_clip | true_positives | false_positives | false_negatives | clips_with_no_predictions | hamming_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| audio | 504.0000 | 924.0000 | 1.6154 | 2.9615 | 227.0000 | 697.0000 | 277.0000 | 3.0000 | 0.3122 |
| combined | 504.0000 | 862.0000 | 1.6154 | 2.7628 | 217.0000 | 645.0000 | 287.0000 | 10.0000 | 0.2987 |
| embedding_lr | 504.0000 | 413.0000 | 1.6154 | 1.3237 | 122.0000 | 291.0000 | 382.0000 | 72.0000 | 0.2157 |
| audio_embedding_lr | 504.0000 | 399.0000 | 1.6154 | 1.2788 | 129.0000 | 270.0000 | 375.0000 | 75.0000 | 0.2067 |

The all-negative reference Hamming loss was approximately 0.1615.
Low Hamming loss alone did not establish useful behavior detection.
Raw embedding models predicted fewer positive labels and missed
more actual positive labels than the audio-only model.

## PCA experiment

The 1024 embedding features were reduced to 32 PCA components.
With 44 audio features, the classifier received 76 features.

Embedding scaling, PCA, and final feature scaling were fitted
separately within each fold's fitting portion. Logistic regression
settings and the prediction threshold were unchanged.

## Grouped cross-validation comparison

| model | macro_f1_mean | macro_f1_std | micro_f1_mean | micro_f1_std | hamming_loss_mean | hamming_loss_std |
| --- | --- | --- | --- | --- | --- | --- |
| audio | 0.2728 | 0.0327 | 0.3182 | 0.0421 | 0.3101 | 0.0180 |
| audio_embedding_lr | 0.1979 | 0.0499 | 0.2860 | 0.0587 | 0.2061 | 0.0150 |
| audio_embedding_pca32_lr | 0.2607 | 0.0297 | 0.3374 | 0.0375 | 0.2438 | 0.0196 |
| combined | 0.2702 | 0.0475 | 0.3215 | 0.0593 | 0.2947 | 0.0312 |
| embedding_lr | 0.1814 | 0.0373 | 0.2673 | 0.0517 | 0.2153 | 0.0164 |
| late_fusion | 0.2737 | 0.0515 | 0.3197 | 0.0548 | 0.2949 | 0.0246 |
| visual | 0.2077 | 0.0441 | 0.2337 | 0.0518 | 0.4025 | 0.0152 |

These values are means and sample standard deviations across folds.

## Paired-fold comparison with audio-only

| metric | audio_mean | pca_mean | mean_pca_benefit | benefit_fold_std | folds_pca_better | folds |
| --- | --- | --- | --- | --- | --- | --- |
| macro_f1 | 0.2728 | 0.2607 | -0.0122 | 0.0469 | 1.0000 | 5.0000 |
| micro_f1 | 0.3182 | 0.3374 | 0.0192 | 0.0330 | 4.0000 | 5.0000 |
| hamming_loss | 0.3101 | 0.2438 | 0.0663 | 0.0105 | 5.0000 | 5.0000 |

Positive benefit means the PCA model performed better.
Fold consistency is descriptive and is not a significance test.

## Per-label comparison

| label | support | audio_f1 | pca_f1 | f1_difference | audio_precision | pca_precision | audio_recall | pca_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Background | 7.0000 | 0.0000 | 0.1667 | 0.1667 | 0.0000 | 0.2000 | 0.0000 | 0.1429 |
| Self-Spinning or Spinning Objects | 45.0000 | 0.2676 | 0.3922 | 0.1246 | 0.1959 | 0.3509 | 0.4222 | 0.4444 |
| Upper Limb Stereotypies | 111.0000 | 0.3798 | 0.4298 | 0.0499 | 0.3333 | 0.3969 | 0.4414 | 0.4685 |
| Self-Hitting or Self-Injurious Behavior | 43.0000 | 0.2353 | 0.2444 | 0.0092 | 0.1720 | 0.2340 | 0.3721 | 0.2558 |
| Hyper- or Hyporeactivity to Sensory Input | 70.0000 | 0.3267 | 0.3094 | -0.0173 | 0.2500 | 0.2523 | 0.4714 | 0.4000 |
| Aggressive Behavior | 62.0000 | 0.4832 | 0.4571 | -0.0261 | 0.4138 | 0.4103 | 0.5806 | 0.5161 |
| Non-Typical Language | 42.0000 | 0.2880 | 0.2400 | -0.0480 | 0.2169 | 0.2069 | 0.4286 | 0.2857 |
| Non-Responsiveness to Verbal Interaction | 48.0000 | 0.2953 | 0.2439 | -0.0514 | 0.2178 | 0.2000 | 0.4583 | 0.3125 |
| Absence or Avoidance of Eye Contact | 60.0000 | 0.3488 | 0.2937 | -0.0551 | 0.2679 | 0.2530 | 0.5000 | 0.3500 |
| Object Lining-Up | 16.0000 | 0.1538 | 0.0000 | -0.1538 | 0.1111 | 0.0000 | 0.2500 | 0.0000 |

Per-label scores pool the training out-of-fold predictions.
They are not the same calculation as mean per-fold Macro F1.

## Interpretation

- PCA improved F1 compared with the raw audio-plus-embedding model.
- PCA improved Micro F1 over audio-only in four of five folds.
- PCA reduced Hamming loss over audio-only in all five folds.
- PCA improved Macro F1 over audio-only in only one of five folds.
- Self-Spinning and Upper Limb Stereotypies showed higher pooled F1.
- Object Lining-Up had zero pooled F1 and recall with the PCA model.
- Background had only seven positive training examples.

The PCA model remains an experimental comparison.
The previously frozen primary baseline was not replaced.

## Limits and experiment status

The PCA experiment followed inspection of earlier CV results.
Its scores are exploratory; repeated use of these folds can influence
model development decisions.

The original test results had already been inspected in earlier work.
These new training experiments do not create a new independent test set.

Only the available 527-clip subset was prepared. Class imbalance,
limited source groups, incomplete pose coverage, and static frame
pooling remain relevant limitations.

## Saved evidence

- results/grouped_cv/
- results/embedding_grouped_cv/
- results/embedding_pca_grouped_cv/
- results/pose_quality_audit/
- results/pose_quality_audit_full/
- dataset/features/visual_embeddings_train/


## Additional robustness experiment

See [Missing-Modality Robustness](MODALITY_ROBUSTNESS.md) for the frozen-model validation sensitivity check.
