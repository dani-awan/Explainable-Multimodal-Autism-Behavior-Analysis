# ASD-FEAT E6 Development Experiments

## Scope

This exploratory comparison uses 1510 development recordings
from 259 participants. The target is current-visit
`asd_longitudinal`; this experiment does not establish future diagnosis prediction.

Three feature branches were aligned by recording filename:
224 audio summaries, 1024 pooled I3D RGB features, and 12 human-coded behavioral
features. The behavioral branch requires human annotations. The three branches
do not represent three independent sensor modalities.

Three development recordings lack released audio features and were excluded
from every E6 comparison. All 113 positive development recordings and all
23 participants with positive recordings remain covered.

## Evaluation Protocol

- Reused the existing five participant-group development folds.
- Participant overlap between fitting and evaluation portions was zero.
- Used the same 1510 recordings for all E6 models and ablations.
- Fitted scaling and branch-specific PCA inside each fitting fold only.
- Used 32 PCA components per audio or RGB branch.
- Used balanced logistic regression with C=1.0 and a fixed threshold of 0.5.
- Used equal probability weights for late fusion.
- Did not tune thresholds or evaluate the official test set in E6.

## Initial Model Comparison

| model | asd_f1_mean | asd_f1_std | precision_mean | recall_mean | roc_auc_mean | average_precision_mean |
| --- | --- | --- | --- | --- | --- | --- |
| audio_pca32_lr | 0.2057 | 0.0422 | 0.1214 | 0.6792 | 0.7071 | 0.1807 |
| behavior_lr | 0.2140 | 0.0276 | 0.1290 | 0.6410 | 0.6922 | 0.1741 |
| dummy_prior | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0748 |
| rgb_pca32_lr | 0.2729 | 0.0696 | 0.1733 | 0.6623 | 0.7730 | 0.2280 |
| three_branch_late_fusion | 0.2793 | 0.0362 | 0.1771 | 0.6828 | 0.7915 | 0.2703 |
| three_branch_pca32_lr | 0.2920 | 0.0327 | 0.1995 | 0.5889 | 0.7721 | 0.2370 |

Scores above are unweighted means across evaluation folds. They differ from
metrics calculated after pooling all out-of-fold predictions.

## Pooled Recording-Level Metrics

| model | macro_f1 | asd_f1 | precision | recall | specificity | balanced_accuracy | accuracy | roc_auc | average_precision | mcc | tp | fp | fn | tn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| behavior_lr | 0.4918 | 0.2121 | 0.1272 | 0.6372 | 0.6464 | 0.6418 | 0.6457 | 0.6846 | 0.1579 | 0.1541 | 72 | 494 | 41 | 903 |
| audio_pca32_lr | 0.4767 | 0.2074 | 0.1226 | 0.6726 | 0.6106 | 0.6416 | 0.6152 | 0.6944 | 0.1403 | 0.1515 | 76 | 544 | 37 | 853 |
| rgb_pca32_lr | 0.5583 | 0.2757 | 0.1740 | 0.6637 | 0.7452 | 0.7044 | 0.7391 | 0.7637 | 0.1783 | 0.2382 | 75 | 356 | 38 | 1041 |
| three_branch_pca32_lr | 0.5875 | 0.2966 | 0.1988 | 0.5841 | 0.8096 | 0.6968 | 0.7927 | 0.7585 | 0.2065 | 0.2501 | 66 | 266 | 47 | 1131 |
| dummy_prior | 0.4806 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.5000 | 0.9252 | 0.4870 | 0.0728 | 0.0000 | 0 | 0 | 113 | 1397 |
| three_branch_late_fusion | 0.5603 | 0.2805 | 0.1766 | 0.6814 | 0.7430 | 0.7122 | 0.7384 | 0.7852 | 0.2476 | 0.2464 | 77 | 359 | 36 | 1038 |

## Branch Ablation

Positive full-model benefit means the three-branch early-fusion model
outperformed the reduced model.

| experiment | metric | full_model_mean | reduced_model_mean | mean_full_model_benefit | folds_full_model_better | folds |
| --- | --- | --- | --- | --- | --- | --- |
| without_audio | asd_f1 | 0.2920 | 0.3142 | -0.0222 | 1 | 5 |
| without_audio | macro_f1 | 0.5847 | 0.5937 | -0.0091 | 1 | 5 |
| without_audio | precision | 0.1995 | 0.2121 | -0.0126 | 1 | 5 |
| without_audio | recall | 0.5889 | 0.6558 | -0.0669 | 0 | 5 |
| without_audio | balanced_accuracy | 0.6991 | 0.7272 | -0.0281 | 0 | 5 |
| without_audio | roc_auc | 0.7721 | 0.8070 | -0.0349 | 0 | 5 |
| without_audio | average_precision | 0.2370 | 0.2828 | -0.0458 | 0 | 5 |
| without_rgb | asd_f1 | 0.2920 | 0.2585 | 0.0335 | 3 | 5 |
| without_rgb | macro_f1 | 0.5847 | 0.5440 | 0.0407 | 5 | 5 |
| without_rgb | precision | 0.1995 | 0.1630 | 0.0365 | 5 | 5 |
| without_rgb | recall | 0.5889 | 0.6525 | -0.0636 | 2 | 5 |
| without_rgb | balanced_accuracy | 0.6991 | 0.6916 | 0.0075 | 3 | 5 |
| without_rgb | roc_auc | 0.7721 | 0.7720 | 0.0001 | 2 | 5 |
| without_rgb | average_precision | 0.2370 | 0.2676 | -0.0306 | 3 | 5 |
| without_behavior | asd_f1 | 0.2920 | 0.2237 | 0.0682 | 5 | 5 |
| without_behavior | macro_f1 | 0.5847 | 0.5337 | 0.0510 | 5 | 5 |
| without_behavior | precision | 0.1995 | 0.1438 | 0.0557 | 5 | 5 |
| without_behavior | recall | 0.5889 | 0.5242 | 0.0647 | 2 | 5 |
| without_behavior | balanced_accuracy | 0.6991 | 0.6414 | 0.0577 | 4 | 5 |
| without_behavior | roc_auc | 0.7721 | 0.7278 | 0.0443 | 4 | 5 |
| without_behavior | average_precision | 0.2370 | 0.1822 | 0.0548 | 4 | 5 |

## Findings

Removing audio produced RGB plus human-coded behavior, with mean ASD F1 0.3142,
ROC-AUC 0.8070, and average precision 0.2828. Its ASD F1 exceeded the full
model in four of five folds; ROC-AUC and average precision exceeded it in
all five folds.

This finding concerns the current audio summaries, PCA representation, and
classifier configuration. It does not establish that audio is generally
uninformative.

Removing human-coded behavior reduced ASD F1 in every fold. Removing RGB
reduced macro F1 and precision in every fold, while other metrics showed
mixed changes.

The full three-branch model's saved out-of-fold probabilities were reproduced
before ablation. Original models and predictions were preserved.

## Limitations

- Precision remains low, and false positives remain substantial.
- Repeated recordings from one participant are not independent participants.
- Fold fitting sets overlap; fold consistency is descriptive and is not a
  statistical significance test.
- Results are exploratory development evidence, not independent clinical
  validation.
- Human-coded behavioral inputs limit claims of fully automated inference.
- Branch removal measures performance changes after refitting; it does not
  establish causal contributions.
- The official ASD-FEAT test set was previously evaluated for E1 and contains
  positive recordings from only one participant. Future E6 test results must
  disclose that history and limitation.
- No final E6 model was frozen for official test evaluation by this report.

## Evidence Locations

- Initial comparison: `results/asd_feat/e6_grouped_cv/20261007T202805_649935Z`
- Ablation: `results/asd_feat/e6_ablation/20261007T203536_456317Z`
- Aligned inputs: `dataset/external/asd_feat/derived/e6_inputs`
