# Final submission checklist

## Completed according to supplied evidence

- AV-ASD dataset and split alignment, audio and visual features, baselines and frozen evaluation.
- ASD-FEAT participant grouping, aligned audio/RGB/behavior inputs, five-fold comparisons, branch ablation and simulated missing-branch checks.
- Selected RGB plus human-coded behavior model frozen and evaluated with unchanged threshold 0.5.
- Linear explanation reconstruction and same-environment reproduction.
- Local Streamlit research dashboard, compatible CSV upload and HTML report demonstration.
- Ten input-validation function checks on Windows.
- Consolidated report, editable presentation and demonstration guide in this package.

## Finish on the Windows laptop

1. Extract this package into `F:\AV-ASD-Project\AV-ASD-main`.
2. Run `scripts/check_final_submission.py` and review any FAIL output. The script does not retrain or change experiment files.
3. Open the report PDF and presentation. Confirm team names, formatting and any institution-required template. The report is a concise research prototype report, not a publication-ready paper.
4. Follow `docs/FINAL_DEMO_GUIDE.md` once and save the exported demo HTML report locally.
5. Commit only reviewed code, aggregate documentation and the intended presentation. Check the GitHub PR state; earlier pushes do not prove that every PR has been merged.
6. Keep a dated backup of the complete local project and environments outside C. Do not relocate an active Conda installation or environment by copying its folder.

## Review before wider publication or deployment

- Verify release-specific dataset licensing and redistribution permissions.
- Verify remaining proposal bibliography entries and required academic citations.
- Disclose prior E1 test exposure and exploratory E6 candidate selection.
- Preserve the one-positive-participant limitation and recording-level interpretation.
- Do not describe uncalibrated model scores as a clinical diagnosis, severity level or validated chance of autism.

Independent participant validation, calibration and verified raw-media extraction are future research. They are not implemented capabilities of this feature-based demonstration.
