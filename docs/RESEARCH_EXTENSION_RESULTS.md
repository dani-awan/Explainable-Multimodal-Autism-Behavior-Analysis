# Research extension results

Team: Muhammad Danial, Hamza Rasheed, Muhammad Fareed.

Six exploratory candidates were evaluated using the existing five participant-disjoint folds: 1,510 recordings from 259 participants, with 113 positive recordings. Threshold 0.5 was fixed. No official test was evaluated and no frozen model was replaced.

| Model | Mean ASD F1 | SD | Precision | Recall | ROC-AUC | AP |
|---|---:|---:|---:|---:|---:|---:|
| behavior_xgb | 0.2369 | 0.0258 | 0.1683 | 0.4187 | 0.6836 | 0.1984 |
| rgb_behavior_xgb | 0.2827 | 0.1684 | 0.3525 | 0.2511 | 0.8173 | 0.2700 |
| three_branch_xgb | 0.2127 | 0.1665 | 0.2651 | 0.1900 | 0.8196 | 0.2636 |
| mean_pool_lr | 0.3092 | 0.0357 | 0.2088 | 0.6478 | 0.8070 | 0.2823 |
| rgb_lstm_behavior | 0.2999 | 0.0775 | 0.2588 | 0.3911 | 0.7546 | 0.2317 |
| rgb_tcn_behavior | 0.2510 | 0.0863 | 0.2006 | 0.3650 | 0.7943 | 0.2177 |

## Interpretation

The temporal matched mean-pooling LR control has higher mean ASD F1 than LSTM and TCN in this fixed configuration. This does not establish that temporal learning is generally ineffective. XGBoost uses raw features, whereas the earlier selected LR uses PCA; algorithm and representation differences must both be considered. Previous reported RGB-plus-behavior LR mean ASD F1 was 0.3142; the new extensions provide no reason to silently replace its frozen artifact.

SHAP explained all held-out XGBoost predictions in raw logit space, with reconstruction and saved inference checks. Temporal models also passed saved inference and OOF coverage checks. All 30 fold models are new experimental artifacts.

Stored RGB array order was retained in 32 contiguous bins. Timestamps and audio synchronization were not independently verified. Behavior inputs remain human-coded. SHAP reflects fitted feature attribution, not causal or clinical evidence.

These extensions were designed after earlier official-test results had been seen. They are exploratory development comparisons, without nested model selection or significance claims. Fold standard deviations are not confidence intervals. Model scores are uncalibrated and do not diagnose autism or determine severity.

## Remaining work

Dataset permission review, verified literature synthesis and DREAM feasibility remain separate reviews. The research report, presentation and README must be updated to reflect these new findings before final publication.

## Verification

All 15 fold metric rows per experiment were recalculated from saved OOF predictions. Recording IDs, targets, participant IDs, fold assignments and threshold predictions matched the development metadata. Source hashes are recorded in RESEARCH_EXTENSION_EVIDENCE.json. No models were loaded or fitted during this documentation stage.
