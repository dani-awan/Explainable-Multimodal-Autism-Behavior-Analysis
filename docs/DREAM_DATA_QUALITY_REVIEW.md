# DREAM data-quality review

Team: Muhammad Danial, Hamza Rasheed, Muhammad Fareed.

The verified full release contains 3,121 session JSON files representing 61 participants. Every session parsed successfully. The completed audit is a structural/numeric coverage review, not a training-readiness or diagnostic-validation claim.

## Quality findings

| Category | Sessions |
|---|---:|
| PARTIAL_MISSING | 2991 |
| STRUCTURAL_OR_NUMERIC_FINDINGS | 128 |
| NO_FINITE_COORDINATES | 1 |
| EMPTY | 1 |

Sessions with any finite coordinates: 3119. Sessions with at least one complete joint XYZ observation: 3041. Complete triplets do not prove tracking accuracy.

## Corrected task counts

The initial audit grouped by the complete task dictionary. This review groups by the ability field instead.

| Ability code | Sessions |
|---|---:|
| IM | 450 |
| JA | 413 |
| TT | 1934 |
| UNSPECIFIED | 324 |

Blank abilities are labelled UNSPECIFIED, not an invented class. TT, JA and IM labels are therapy activities, not autism diagnosis or severity stages.

## Provisional activity cohort

Selected 2615 sessions from 56 participants using a fixed basic availability rule: Known TT/JA/IM task; aligned coordinate lengths; COMPLETE_NUMERIC or PARTIAL_MISSING; at least 32 frames with one complete joint XYZ.

Selection uses saved audit metadata and does not clean raw sequences. Actual tracking coverage, feature extraction and participant-disjoint class support still need validation before an activity experiment.

## Interpretation

RET and SHT are therapy conditions among participants with ASD. They cannot be relabelled as autism-positive versus autism-negative. DREAM motion/gaze measurements do not match the AV-ASD or ASD-FEAT frozen input schemas.

The time field was a string in the inspected sessions. Per-frame timing and timestamp units were not independently established. No diagnostic model, severity model or external evaluation of either frozen classifier was performed.

Dataset source: https://doi.org/10.5878/17p8-6k13. Catalogue license: CC BY-NC-SA 4.0. Local recording-level rows remain in the results directory; this document contains aggregate findings.

Original coordinate audit files remain preserved. New corrected metadata, provisional selection and source hashes are stored in the separate review output.
