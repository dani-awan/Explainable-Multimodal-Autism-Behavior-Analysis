# E6 Standalone Windows Validation

## Completed Checks

- Development recordings: 1510
- Participants: 259
- Evaluation folds: 5
- Fixed prediction threshold: 0.5
- Participant overlap in every fold: 0

| Check | Result |
| --- | --- |
| Training CV reproduction against original results | PASS |
| OOF prediction rows compared | 1510 |
| Fold-result rows compared | 30 |
| Maximum numeric difference in CV comparison | 0.0 |
| Explanation logit reconstruction and OOF reproduction | PASS |
| Explained recordings | 1510/1510 |
| Ablation full-model reproduction and OOF coverage | PASS |
| Missing-branch baseline reproduction and OOF coverage | PASS |

## Evidence Folders

All paths below are relative to the project root.

- CV validation: results/asd_feat/e6_reproduction/20261008_115710_8871053/cv_validation
- Explanations: results/asd_feat/e6_reproduction/08_explain_models/20261008T185834_650176Z
- Ablation: results/asd_feat/e6_reproduction/06_branch_ablation/20261008T190006_874650Z
- Robustness: results/asd_feat/e6_reproduction/07_missing_branch_robustness/20261008T190251_527238Z

## Scope

This record summarizes successful Windows execution logs.
Original experiment outputs were preserved.
Archive extraction and input rebuilding were not repeated in this validation.
No official E6 test evaluation was performed.
Mean-imputation robustness is simulated and does not establish real-world missing-input reliability.
