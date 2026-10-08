# ASD-FEAT E6 standalone research scripts

This package converts the supplied notebook source into separate, reusable development stages. It complements the existing AV-ASD scripts; ASD-FEAT and AV-ASD remain different datasets and prediction tasks.

The E6 development experiment used 1510 recordings from 259 participants, with 113 positive recordings from 23 participants. Its input branches were 224 audio summaries, 1024 pooled I3D RGB features and 12 human-coded behavioral features. The target is the current-visit annotation, not a future diagnosis or clinical stage.

## Start here on Windows

Download `AV_ASD_E6_Standalone_Scripts.zip` to Downloads and extract its contents into `F:\AV-ASD-Project\AV-ASD-main`. The ZIP contains project-relative paths, without an extra top-level folder.

Run this in **PowerShell**, not a Python notebook cell:

```powershell
& "$env:USERPROFILE\anaconda3\envs\practice\python.exe" "F:\AV-ASD-Project\AV-ASD-main\scripts\asd_feat_e6\00_check_project.py"
```

This command reads files and inspects package presence. It does not train models or alter results. Stage commands also default to a read-only preflight; execution requires `--run`.

## Included stages

| Script | Purpose | Source notebook cell |
| --- | --- | --- |
| 00_check_project.py | Inspect scripts and existing project prerequisites | New |
| 01_extract_audio.py | Pool released audio arrays into 224 features | 116 |
| 02_extract_rgb.py | Pool released RGB arrays into 1024 features, verify visit metadata | 119 |
| 03_build_inputs.py | Align feature branches, labels and participant folds | 120 |
| 04_grouped_cv.py | Fit five-fold development baselines and equal-weight late fusion | 122 |
| 05_compare_folds.py | Compare matched-fold metrics descriptively | 123 |
| 06_branch_ablation.py | Refit each reduced model; reproduce full-model OOF predictions | 124 |
| 07_missing_branch_robustness.py | Simulate training-mean replacement using saved fold models | 126 |
| 08_explain_models.py | Reconstruct linear logits and OOF predictions | 129 |
| 09_write_development_report.py | Write evidence and Markdown for selected runs | 125 |
| 10_inspect_saved_model.py | Inspect a trusted fold-1 saved bundle | 128 |
| 11_verify_cv_reproduction.py | Compare a new development CV run with a reference | New |

Two dashboard page sources are exported from cells 127 and 130 under `docs/e6_dashboard_sources/` with `.py.txt` extensions. Existing dashboard pages are not replaced by extracting this ZIP. These sources use the historical report paths recorded in the notebook; if needed, copy them into `dashboard/pages/` using the `.py` filenames after backing up your local pages.

## Execution rules

- Output folders default to timestamped directories under `results/asd_feat/e6_reproduction/`.
- Existing output folders are rejected. No overwrite option is supplied.
- `--input-dir`, `--audio-dir`, `--rgb-dir`, `--base-run` and `--ablation-run` explicitly select upstream artifacts where applicable.
- Numerical settings remain fixed: 32 PCA components per audio/RGB branch, balanced logistic regression, C=1, random seed 42 and threshold 0.5. Scaling and PCA are fitted inside each training fold.
- Archive reads are sequential, load one selected array at a time and use `allow_pickle=False`. Extraction checkpoints are saved within the new output folder. They do not implement automatic resume.
- Use the original, trusted model bundles. Joblib loading executes Python serialization; these scripts are for your project's own artifacts.
- No official E6 test processing or model selection is added. E1 previously exposed the official test set, whose positive recordings came from one participant; that limitation still applies.
- `--check` confirms paths and Python package presence; it does not validate all CSV contents, hashes, archive checksums or model compatibility. Those checks run inside the selected stage.

See `docs/E6_STANDALONE_RUN_ORDER.md` for commands and `docs/E6_STANDALONE_VALIDATION.json` for the package checks. Keep raw data, feature CSVs, fitted models, experiment outputs and notebook outputs out of GitHub according to the project's existing `.gitignore`.
