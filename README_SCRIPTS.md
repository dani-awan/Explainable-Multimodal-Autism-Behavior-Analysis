# AV-ASD standalone research scripts

This is an additive script package for the existing AV-ASD project. Copy its
`scripts` and `docs` folders and the two dependency input files into the project
root. Keep the existing notebook, dashboard, datasets, model bundles, results,
and MediaPipe extraction scripts.

## First check on Windows

Run this in PowerShell after copying the package:

```powershell
cd "F:\AV-ASD-Project\AV-ASD-main"
& "$env:USERPROFILE\anaconda3\envs\practice\python.exe" ".\scripts\reproducible\00_check_project.py"
```

This command checks required input paths. It does not extract videos, train
models, change results, or install packages. `READY` means the listed input paths
exist; it does not certify the contents or the installed dependencies.

## Run a specific stage

```powershell
python .\scripts\reproducible\29_check_modality_robustness.py --check
```

Use the project's main Python interpreter for stages 01–33. For an intentional
rerun that updates existing generated outputs:

```powershell
python .\scripts\reproducible\29_check_modality_robustness.py --overwrite
```

Use `--project` to point a script at a separate experiment copy. By default,
scripts locate the project root from their location under `scripts/reproducible`.
Use `--help` to see the options. Keep Python assertions enabled; `python -O`
is rejected because it would disable validation checks.

For original-video metadata checks, specify the video directory explicitly:

```powershell
python .\scripts\reproducible\02_check_original_videos.py --original-videos "F:\AV-ASD-Videos\original_videos" --check
```

Every stage is independently executable. Imports do not start extraction or
training. There is deliberately no command that reruns the entire workflow
without inspecting its outputs.

## What was preserved

- Original train/validation/test membership and source-video grouping.
- 44 audio features, 14 handcrafted visual features, and their column order.
- LR: StandardScaler, one-vs-rest, C=1, balanced class weights, lbfgs,
  max_iter=3000, random_state=42, one worker.
- RF: the notebook's saved settings and validation threshold search.
- Historical threshold search: 0.3–0.8; training CV threshold: 0.5.
- Five source-group folds; stage 21 reuses existing saved assignments.
- Mean-pooled 1024 visual embeddings and the 32-component PCA experiment.
- Frozen combined-LR robustness threshold of 0.3.

Audio extraction and final saving now execute in the same script rather than
depending on notebook variables. Visual extraction includes its function and
full processing loop; the existing FFmpeg repair remains a separate stage.
Figures are saved without blocking desktop windows. CSV validation failures and
incomplete repaired coverage produce an error exit.

## Existing scripts retained

The working MediaPipe and download scripts are already in the user's project.
This package does not overwrite them. In particular, the source of
`scripts/extract_train_visual_embeddings.py` was not embedded in the uploaded
notebook. Use that existing script with the existing `.venv-pose` interpreter.
Stages 23–24 summarize existing pose reports; they do not run a new pose audit.

## Dependencies

The installed versions on the laptop are recorded in
`requirements-main-snapshot.txt` and `requirements-pose-snapshot.txt`. Keep those
files. This package provides `requirements-main.in` and `requirements-pose.in`
as direct-dependency lists, not replacements for those version snapshots.
Do not install packages into the working environments merely to add these scripts.

FFmpeg and FFprobe must be available on PATH. Main and pose environments remain
separate. An environment recreated from these lists has not been verified.

## Validation limits

The converted model and report stages were exercised with synthetic feature
values in an isolated Linux fixture. All scripts were parsed and all command-line
entry points were checked. See `docs/VALIDATION_RESULTS.json` for the tested scope.

The real Windows videos, fitted laptop models, MediaPipe extraction, and
Streamlit server were not executed here. Synthetic checks validate script
integration, not reproduction of the reported research scores. Verify the
Windows workflow before claiming full reproducibility.

The original test results were already inspected. A rerun does not create a new
independent test evaluation. PCA follow-up comparisons remain exploratory.

See `docs/RUN_ORDER.md` for stage order and `docs/SCRIPT_SOURCE_MAP.json` for
the notebook cell mapping. The source notebook is not changed or included in
this package. No video, audio, preview image, notebook output, or model bundle
is included.
