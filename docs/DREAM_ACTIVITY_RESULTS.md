# DREAM therapy-activity development results

Team: Muhammad Danial, Hamza Rasheed, Muhammad Fareed.

This separate exploratory task classifies IM/JA/TT therapy activity labels from 54 body-relative position descriptors. It uses 2,615 quality-filtered sessions from 56 participants. Five stratified group folds keep participants separate; all 15 fold models passed saved-inference checks.

## Mean participant-fold results

| Model | Macro F1 | Fold SD | Balanced accuracy | Accuracy | Macro OVR ROC-AUC |
|---|---:|---:|---:|---:|---:|
| activity_lr | 0.5819 | 0.0436 | 0.6456 | 0.6262 | 0.7882 |
| activity_rf | 0.5911 | 0.0229 | 0.6035 | 0.7316 | 0.8034 |
| dummy_prior | 0.2716 | 0.0008 | 0.3333 | 0.6876 | 0.5000 |

## Pooled out-of-fold results

| Model | Macro F1 | Balanced accuracy | Accuracy | Macro OVR ROC-AUC |
|---|---:|---:|---:|---:|
| activity_lr | 0.5808 | 0.6447 | 0.6260 | 0.7868 |
| activity_rf | 0.5936 | 0.6029 | 0.7315 | 0.8001 |
| dummy_prior | 0.2716 | 0.3333 | 0.6876 | 0.4907 |

## Interpretation

Both learned models exceed the dummy baseline in mean Macro F1. Random Forest has the higher mean Macro F1 and accuracy in this run; Logistic Regression has higher balanced accuracy/macro recall. The majority TT class makes accuracy alone inadequate. The small LR/RF Macro F1 difference has not been tested for statistical significance.

Each model uses fit participants only. LR scaling is fitted within each fold; its class weights and RF balanced-subsample weights are computed from training data. Predictions use argmax class scores, without threshold tuning. Fold means and pooled metrics are different summaries and must not be interchanged.

This is activity recognition among children who already have ASD. It is not autism diagnosis, severity prediction, therapy-effect estimation, or external validation of the AV-ASD/ASD-FEAT frozen models. Original frozen artifacts remain preserved.

## Limits and verification

The cohort was filtered for coordinate availability. The descriptors summarize positions relative to the shoulder centre; they do not normalize body size or rotation and do not model temporal order. Sensor configuration, session posture and missingness may confound labels. No untouched independent test cohort was evaluated.

This documentation stage recalculated all fold and pooled metrics from saved OOF rows and verified participant/label/fold alignment. It loaded no models and changed no predictions. Source hashes and pooled confusion matrices appear in DREAM_ACTIVITY_EVIDENCE.json.

Dataset source: https://doi.org/10.5878/17p8-6k13. Observed catalogue license: CC BY-NC-SA 4.0. Keep session-level files local; these documents contain aggregate research findings.
