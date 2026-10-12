# Video Upload - Final Validation Status

## Implemented
- Local MP4 upload using the frozen AV-ASD combined Logistic Regression.
- 44 audio and 14 visual features; 10 behavioral labels.
- Frozen prediction threshold: 0.3.
- Readable recording summary and linear feature contributions.
- HTML and JSON report downloads.
- Changing the uploaded video clears the previous report.

## Completed checks
- Existing-video feature and prediction reproduction: 5/5 PASS.
- Synthetic upload cases: 5/5 PASS.
- Explanation reconstruction: 107 recordings, 1070 label checks, PASS.
- Maximum reconstructed-score difference: 6.66133814775e-16.
- Exported summary and all ten label explanation tables inspected.

## Scope
This page uses the AV-ASD model separately from ASD-FEAT.
Checks verify tested software behavior and explanation arithmetic.
Accuracy on arbitrary new videos has not been established.
Scores are not validated autism probabilities, diagnosis or severity.

## Remaining submission work
- Update README and demonstration guide for the MP4 extension.
- Add this extension to the research report and presentation.
- Review and publish the new code through GitHub after local completion.