# E6 standalone run order

All commands below are PowerShell commands. These scripts use the main research Python environment, not the MediaPipe environment. Run individual stages in order and inspect their output before continuing.

## Read-only checks first

```powershell
$project = "F:\AV-ASD-Project\AV-ASD-main"
$python = Join-Path $env:USERPROFILE "anaconda3\envs\practice\python.exe"
$stages = Join-Path $project "scripts\asd_feat_e6"

& $python (Join-Path $stages "00_check_project.py")
& $python (Join-Path $stages "04_grouped_cv.py") --check
& $python (Join-Path $stages "08_explain_models.py") --check
```

The default reference CV folder is the original `20261007T202805_649935Z` run. `--base-run` selects another run explicitly; scripts never silently select the latest folder.

## Reproduce training from existing validated inputs

Extraction and input preparation are already complete in the existing project. They do not need to be repeated just to reproduce training.

```powershell
$stamp = Get-Date -Format "yyyyMMdd_HHmmss_fffffff"
$session = Join-Path $project "results\asd_feat\e6_reproduction\$stamp"
$cv = Join-Path $session "grouped_cv"

& $python (Join-Path $stages "04_grouped_cv.py") --run --output $cv
if ($LASTEXITCODE -ne 0) { throw "CV failed. Stop here." }

& $python (Join-Path $stages "11_verify_cv_reproduction.py") --run --candidate-run $cv --output (Join-Path $session "cv_validation")
if ($LASTEXITCODE -ne 0) { throw "CV reproduction differs. Inspect the error before continuing." }
```

Stage 11 checks input hashes, numerical settings, recording metadata, predictions, probabilities and matched-fold metrics. It writes a new report and preserves both runs. Different software versions may cause differences; a mismatch requires investigation, not replacement of reference results.

## Optional downstream reproduction

These commands use the new CV run from above and the existing validated inputs:

```powershell
& $python (Join-Path $stages "05_compare_folds.py") --run --base-run $cv --output (Join-Path $session "comparison")
if ($LASTEXITCODE -ne 0) { throw "Comparison failed." }

$ablation = Join-Path $session "ablation"
& $python (Join-Path $stages "06_branch_ablation.py") --run --base-run $cv --output $ablation
if ($LASTEXITCODE -ne 0) { throw "Ablation failed." }

& $python (Join-Path $stages "07_missing_branch_robustness.py") --run --base-run $cv --output (Join-Path $session "robustness")
if ($LASTEXITCODE -ne 0) { throw "Robustness check failed." }

& $python (Join-Path $stages "08_explain_models.py") --run --base-run $cv --output (Join-Path $session "explanations")
if ($LASTEXITCODE -ne 0) { throw "Explanation reconstruction failed." }

& $python (Join-Path $stages "09_write_development_report.py") --run --base-run $cv --ablation-run $ablation --output (Join-Path $session "documentation")
```

Ablation refits reduced models. Robustness and explanation stages use saved models without fitting them. Fold consistency is descriptive, not a significance test. Mean imputation is a simulation, not evidence of real missing-input reliability.

## Optional rebuild from archives

Use this only when auditing extraction or rebuilding inputs. Required archive downloads and manifests must already be present. Publisher checksums should already have been verified. Archive stages read selected development arrays, without extracting full archives onto disk.

```powershell
$stamp = Get-Date -Format "yyyyMMdd_HHmmss_fffffff"
$session = Join-Path $project "results\asd_feat\e6_reproduction\rebuild_$stamp"
$audio = Join-Path $session "audio"
$rgb = Join-Path $session "rgb"
$inputs = Join-Path $session "inputs"

& $python (Join-Path $stages "01_extract_audio.py") --run --output $audio
if ($LASTEXITCODE -ne 0) { throw "Audio extraction failed." }

& $python (Join-Path $stages "02_extract_rgb.py") --run --output $rgb
if ($LASTEXITCODE -ne 0) { throw "RGB extraction failed." }

& $python (Join-Path $stages "03_build_inputs.py") --run --audio-dir $audio --rgb-dir $rgb --output $inputs
if ($LASTEXITCODE -ne 0) { throw "Input alignment failed." }

$cv = Join-Path $session "grouped_cv"
& $python (Join-Path $stages "04_grouped_cv.py") --run --input-dir $inputs --output $cv
```

For all downstream stages that accept `--input-dir`, pass `$inputs` when using this rebuilt run. Input hashes are checked against the selected CV configuration. Rebuilding inputs changes the schema's provenance path, so stage 11's strict original-input hash comparison is intended for the first workflow, which reuses the original input files.

## Changes from the notebook

Project paths and run paths are CLI parameters. New outputs replace in-place writes to historical CV folders and canonical docs. RGB visit metadata is joined from the audio reference after verifying IDs, participant IDs, targets and folds. Report text derives counts from the selected run and omits hardcoded conclusions tied to old scores. The numerical training and pooling methods remain unchanged. The Streamlit pages retain their original historical result folders.

Package validation used synthetic data in a Linux environment. It does not certify a completed 1510-recording Windows rerun or independently verify the historical scientific results.
