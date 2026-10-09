# Selected E6 Research Model Card

- Model: RGB plus human-coded behavior, PCA32 and balanced logistic regression.
- Task: Binary current-visit ASD annotation prediction, at recording level.
- Inputs: 1036 features: 1024 pooled I3D RGB features and 12 human-coded behavioral measurements.
- Development fit: 1510 recordings, 259 participants, 113 positive recordings.
- Selection: Highest observed mean ASD F1 among exploratory development CV and branch-ablation candidates.
- Decision threshold: 0.5, fixed before this E6 test inference.
- Artifact: `results/asd_feat/e6_reproduction/12_freeze_rgb_behavior_model/20261008T192951_063340Z/rgb_behavior_pca32_lr_frozen.joblib`.
- Model SHA-256: `2c8e9d9c126e9a7a8134eb9792550254de3747847b96aeaca1e396bae57327d4`.
- Test ASD F1: 0.3571.
- Test precision / recall: 0.2273 / 0.8333.
- Test balanced accuracy: 0.8033.
- Test ROC-AUC / average precision: 0.8728 / 0.2608.

## Intended Scope

Exploratory research and demonstration using released feature arrays and human-coded behavioral annotations. A raw video alone does not provide all required inputs through this pipeline. This is separate from the AV-ASD multilabel behavior model.

## Limitations

The 12 positive test recordings represent 1 participant(s). Repeated recordings are correlated; these recording-level results do not establish generalization across positive participants.

The official test had previously been evaluated for E1. It is not an untouched test set. The E6 candidate was selected after exploratory development comparisons, without nested selection validation. Human-coded behavioral features are required. Clinical diagnosis, disease-stage assessment and treatment recommendations are not validated.

No validated clinical diagnosis, disease-stage assignment or treatment guidance is provided. Further independent participant-level validation is required before claims beyond exploratory research.
