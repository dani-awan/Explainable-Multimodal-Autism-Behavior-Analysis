# Final demonstration guide

Team: Muhammad Danial, Hamza Rasheed and Muhammad Freeh.

## Launch the existing project

Run this command in PowerShell. Copy the command only, without the `PS C:\...>` prompt.

```powershell
& "$env:USERPROFILE\anaconda3\envs\practice\python.exe" -m streamlit run "F:\AV-ASD-Project\AV-ASD-main\dashboard\app.py" --server.address 127.0.0.1 --server.port 8501
```

Keep this terminal open and open http://127.0.0.1:8501 in your browser. If an existing server is already running, use that server. If the port is occupied by an old Streamlit session, stop that session with Ctrl+C before starting a new one.

## Five-minute presentation

1. Introduce the two separate research tasks: AV-ASD behavior-label recognition and ASD-FEAT current-visit annotation classification. Show the report or presentation.
2. Open the E6 experiments page. Explain participant-disjoint folds, branch comparisons and the selected RGB plus human-coded behavior candidate. Its mean development-fold ASD F1 is 0.3142.
3. Open the E6 Frozen Test page. Show ASD F1 0.3571, precision 0.2273 and recall 0.8333. State that all 12 positive test recordings come from one participant. These results do not establish clinical performance.
4. Open Research Analysis. Choose the existing test-recording demonstration and Recording 1. Run analysis and show the displayed 16.61% fitted model score, threshold and branch contributions.
5. Download the HTML report. Then choose compatible feature CSV upload, use the matching `dataset/demo_inputs/recording_1_features.csv`, confirm provenance and run analysis. The matching prepared input should reproduce the same displayed score.

The wording of controls can vary with the installed dashboard version. Use the existing recording mode first if the demo CSV is unavailable. Do not make up feature values to replace missing data.

## Explain the output accurately

Suggested wording: “The model score for this prepared research input is 16.61%, below the fixed 0.5 threshold. This is an uncalibrated model output; it does not establish or exclude autism.”

The selected model requires 1,024 compatible pooled I3D RGB features and 12 human-coded behavioral measurements. It does not currently extract these automatically from arbitrary uploaded photos, audio or video. Branch contributions explain the fitted logit rather than clinical symptoms or causal effects.

## Keep the presentation focused

Use the prepared local demonstration. Do not retrain models, adjust thresholds or regenerate official test predictions during the demo. Keep identifiable media and recording-level feature rows local. The report and slides contain aggregate research findings.

If there is an error, retain its complete text and run `scripts/check_final_submission.py`. This check inspects presence, selected syntax and the frozen model hash without loading or fitting a model. It does not replace the previously completed experiment validation.


## AV-ASD MP4 demonstration update

This workflow is separate from the ASD-FEAT prepared-feature demonstration.

1. Open Video Behavior Analysis.
2. Upload a permitted MP4 with an audio track.
3. Limits: 120 seconds, 7200 frames and 200 MiB.
4. Analyze the video and show ten label scores at threshold 0.3.
5. Show the recording summary and top feature contributions.
6. Download the HTML or JSON report.
7. Select another video; the previous report should clear.

Silent audio is accepted and flagged. Missing audio is rejected.
Above-threshold predictions do not confirm that a behavior occurred.
Scores do not establish autism diagnosis, stage or clinical probability.
Contributions explain model arithmetic, not behavioral causes.

Completed checks:
- Existing-video reproduction: 5/5 PASS.
- Synthetic upload cases: 5/5 PASS.
- Explanation reconstruction: 107 recordings, 1070 checks, PASS.
- Report exports inspected and previous-report clearing confirmed.

See docs/VIDEO_UPLOAD_FINAL_STATUS.md for the validation summary.
The research report and presentation include this extension.