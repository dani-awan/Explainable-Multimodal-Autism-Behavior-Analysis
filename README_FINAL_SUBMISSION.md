# Final research submission package

This package adds final report files, a twelve-slide editable presentation, a demo guide and a read-only project check. It preserves the existing README, dashboard, trained artifacts and experiment outputs.

Contents:

- `docs/FINAL_RESEARCH_REPORT.pdf`: final eight-page research prototype report.
- `docs/FINAL_RESEARCH_REPORT.docx`: editable report.
- `docs/FINAL_RESEARCH_REPORT.md`: text version for repository review.
- `presentations/FINAL_RESEARCH_PRESENTATION.pptx`: twelve editable slides with presenter notes.
- `docs/FINAL_DEMO_GUIDE.md`: local dashboard demonstration.
- `docs/FINAL_SUBMISSION_CHECKLIST.md`: remaining submission and publication checks.
- `scripts/check_final_submission.py`: local presence, syntax, package and artifact-hash check.

No media, feature rows, trained model files or personal credentials are included. Metrics are consolidated from supplied project evidence. No experiment was retrained during preparation of this package. The current product is a research prototype accepting compatible prepared features; arbitrary raw-media analysis and clinical assessment are not implemented.

## PowerShell verification after extraction

```powershell
& "$env:USERPROFILE\anaconda3\envs\practice\python.exe" "F:\AV-ASD-Project\AV-ASD-main\scripts\check_final_submission.py"
```

Read FAIL lines before taking further action. A successful file check does not establish clinical validity, licensing permission or independent experimental reproduction.
