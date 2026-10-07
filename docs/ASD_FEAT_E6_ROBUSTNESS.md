# ASD-FEAT E6 Missing-Branch Robustness

Development recordings: 1510. Participants: 259. Fixed threshold: 0.5.

The existing three-branch early-fusion fold models were evaluated without retraining.
Each unavailable branch was replaced with raw feature means calculated from that fold's fitting rows only.

| Condition | Mean ASD F1 | Mean precision | Mean recall | Mean ROC-AUC | Mean average precision |
| --- | --- | --- | --- | --- | --- |
| all_available | 0.291976051224732 | 0.1995259369570613 | 0.5889026915113872 | 0.7720981183115643 | 0.23702498472524164 |
| audio_mean_imputed | 0.3036645985400633 | 0.23237248258987392 | 0.4996672313194052 | 0.7859425954547413 | 0.24077476918814006 |
| behavior_mean_imputed | 0.20415494451043084 | 0.1385374135010193 | 0.40932354601919824 | 0.7167804649368826 | 0.1738092318970455 |
| rgb_mean_imputed | 0.15067284960901983 | 0.13090893602521508 | 0.19721588556371167 | 0.6729109712406036 | 0.16314868922523265 |

## Findings

- Audio mean imputation improved mean ASD F1 but reduced recall.
- RGB mean imputation produced the largest mean ASD F1 decline.
- Behavioral mean imputation also reduced performance.
- These findings concern this fixed model and representation.

## Verification and limitations

- Baseline probability reproduction and out-of-fold coverage passed.
- Participant overlap was zero in every evaluation fold.
- No models were retrained and no official test evaluation was performed.
- This simulation does not establish real-world missing-input reliability.
- The behavioral branch uses human-coded inputs.
- Previously reported E1 test results include positive recordings from only one participant.

Evidence folder: results/asd_feat/e6_robustness/20261007T204230_919006Z
