# Preparation and Privacy Notes

## Inputs and Provenance

This package is based on the uploaded `data set clean.ipynb`, four ready CSVs, and `video_check_report.csv`. Original uploads were retained unchanged. Ready annotation IDs, labels, and split membership were preserved.

The review notebook works with the supplied files. To rerun preparation, obtain the original `dataset.csv`, `train.csv`, `val.csv`, and `test.csv` under their applicable permissions and place them in `dataset/csvs`. Source videos are local-only and not included.

## Local Preparation Setup

Install the Python requirements, select that environment in your notebook editor, and install FFmpeg/FFprobe separately with both tools on PATH. Set `AV_ASD_VIDEO_DIR` to the original video folder before launching the notebook kernel. On Windows PowerShell, use `$env:AV_ASD_VIDEO_DIR = "<your-original-video-folder>"` in the shell used to launch the editor or notebook server. Restart an already-open editor/kernel if it does not inherit the setting.

Run preparation sections in order. The cleaning stage creates the required `_clean.csv` files. The metadata stage regenerates absolute paths for your machine. The extraction stage writes `clip_verification_report.csv`; the ready-subset stage requires PASS evidence for every metadata-valid clip. It stops on failures instead of silently including failed clips.

## Changes to the Notebook

The preparation notebook retains the final relevant workflow, with portable configuration and English comments. Obsolete folder-discovery, downloader-inspection, downloader-patching, exploratory troubleshooting, personal working-directory prints, and old output cells were omitted. The downloader itself was not provided and is not recreated here.

The extraction code now persists its verification results, and subset generation checks that report before saving. This packaged workflow was syntax-checked but was not rerun on local videos in this environment. The submitted notebook's final output is evidence of the user's earlier completed extraction run.

## Privacy Changes

- Removed notebook outputs, execution counts, and local editor/environment metadata.
- Replaced hard-coded machine paths with repository-relative locations and an environment variable.
- Converted metadata report paths to `original_videos/<filename>`; kept source IDs and measurements.
- Added exclusions for media, environments, credentials, and temporary files.

Source IDs remain necessary research identifiers and may identify the original online media. These files are not anonymous participant data. Keep the repository private and follow dataset redistribution terms.

## Verification Scope

The uploaded annotation files were checked for counts, binary labels, missing values, duplicate IDs, clip/source overlap, exact split label consistency, and metadata-valid coverage. Media files were not supplied, so the 527 PASS / 0 FAIL result is reported from the uploaded notebook, not independently reproduced.

Metadata stream/duration checks are not exhaustive playback checks. No participant-level grouping or clinical validity is established.

## GitHub Upload

Extract the ZIP and upload its contents, preserving `notebooks`, `dataset`, and `docs` folders. Upload the updated README at the repository root, replacing the earlier progress version. Browser uploads do not enforce `.gitignore`; select only these packaged files.
