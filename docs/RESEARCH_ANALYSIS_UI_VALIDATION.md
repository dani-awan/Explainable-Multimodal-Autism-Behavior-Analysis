# Research Analysis UI Validation

Validated on Windows on 2026-10-09.

## Completed demonstration

- Exported one existing test recording as a compatible feature CSV.
- Uploaded the CSV through the Research Analysis dashboard.
- Input contained 1036 finite numeric features.
- Uploaded-input model score: 16.61%.
- Score matched the existing-recording demonstration at displayed precision.
- Prediction: below the frozen threshold of 0.5.
- Branch contributions and explanation appeared in the exported report.
- HTML report download succeeded.

## Scope and remaining work

This verifies one successful CSV upload and report-export demonstration.
It does not validate arbitrary uploaded files or clinical performance.

The score is an uncalibrated model output, not a validated autism risk percentage.
The model does not diagnose autism or assess severity.

Raw video, photo and audio extraction are not implemented.
The selected model requires compatible I3D RGB features and human-coded behavior values.
