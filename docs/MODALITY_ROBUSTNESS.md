# Missing-Modality Robustness

## Method

- Model: frozen combined logistic regression.
- Data: 108 validation clips.
- Prediction threshold: 0.30.
- Model retraining: none.
- Missing features were replaced with the scaler's saved training means.
- Conditions: both available, audio replaced, visual replaced.

## Results

| condition | clips | threshold | macro_f1 | micro_f1 | hamming_loss | false_positives | false_negatives | predicted_labels_per_clip | macro_f1_change | micro_f1_change | hamming_loss_change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| both_available | 108.0000 | 0.3000 | 0.3362 | 0.3730 | 0.3704 | 336.0000 | 64.0000 | 4.2130 | 0.0000 | 0.0000 | 0.0000 |
| audio_mean_imputed | 108.0000 | 0.3000 | 0.2683 | 0.3617 | 0.3269 | 270.0000 | 83.0000 | 3.4259 | -0.0679 | -0.0114 | -0.0435 |
| visual_mean_imputed | 108.0000 | 0.3000 | 0.2778 | 0.3548 | 0.3704 | 327.0000 | 73.0000 | 4.0463 | -0.0584 | -0.0182 | 0.0000 |

## Interpretation

Both replacement conditions reduced Macro F1 and Micro F1.

Replacing audio reduced false positives from 336 to 270, while
false negatives increased from 64 to 83. Its lower Hamming loss
therefore does not establish an overall improvement.

Replacing visual features increased false negatives from 64 to 73.

The frozen model is sensitive to both feature groups under these
replacement conditions.

## Limits

This is a synthetic sensitivity check, not a retrained ablation.
Training-mean replacement does not reproduce every form of real
missing audio or missing visual data.

The validation split had already been used for model development.
These results are descriptive, not a new independent evaluation.

The primary baseline and its frozen threshold remain unchanged.
