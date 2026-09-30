# An Explainable Multimodal AI Framework for Autism-Related Behavioral Analysis

### Using Audio-Visual and Behavioral Features

An applied AI/ML research and prototype project based on the September 2026 proposal prepared by **Hamza Rasheed** for the **NETSOL Technologies Pakistan AI & Machine Learning Trainee Program**.

**Current phase:** Dataset preparation and clip verification.  
**Proposed duration:** 13 weeks.  
**Scope:** Research into observable behavioral patterns; not medical diagnosis.

## What Is This Project?

This project investigates whether combining audio, visual, and structured behavioral information provides more useful analysis of autism-related behavioral patterns than using a single modality.

The proposed framework will process each available modality independently, establish baseline models, compare multimodal fusion strategies, and explain which features or modalities influence predictions. A lightweight interface will demonstrate the results and their limitations.

The project does not assume that multimodal models will always perform better. Their value will be tested through controlled experiments, modality ablation, and a consistent evaluation protocol.

## Problem and Research Motivation

Observable behaviors vary between individuals and recording conditions. A single input may miss information present in another modality. Small datasets, class imbalance, unavailable media, data leakage, and limited interpretability can also make results difficult to reproduce or generalize.

The proposal addresses these challenges through careful dataset auditing, unimodal comparisons, early and late fusion experiments, explainability, and explicit reporting of limitations.

## Aim and Objectives

**Aim:** Design and evaluate an explainable multimodal AI framework integrating audio-visual and structured behavioral features for autism-related behavioral analysis.

1. Develop reproducible preprocessing pipelines.
2. Establish baselines for each available modality.
3. Compare early-fusion and late-fusion strategies.
4. Investigate temporal modeling where ordered sequences support it.
5. Identify influential features and modality contributions.
6. Conduct modality-ablation and robustness experiments.
7. Build a lightweight behavioral-analysis demonstration interface.
8. Document privacy, fairness, dataset limitations, and external-validation needs.

## Research Questions

- How effectively can individual modalities characterize the annotated behaviors?
- Does multimodal fusion improve useful evaluation metrics under the same protocol?
- How do early and late fusion compare?
- Which features and modalities influence predictions most strongly?
- How stable are results under grouped validation and modality ablation?

## Dataset Strategy

| Dataset | Proposed Role | Status in This Project |
|---|---|---|
| AV-ASD | Primary audio-visual behavioral benchmark | Annotation checks and local media preparation underway |
| ASD-FEAT | Complementary derived-behavioral-feature experiments | Candidate; access, files, and licensing still to be verified |
| DREAM | Optional temporal experiments using processed behavioral features | Supplementary candidate; access and suitability still to be verified |

These roles follow the supplied proposal. Supplementary datasets have not yet been integrated. Separate datasets will not be treated as aligned modalities for the same samples unless a valid correspondence exists. Published clinical studies with inaccessible raw data are literature references, not implementation dependencies.

### Current AV-ASD Annotation Overview

The annotation dataset contains **928 clips**, **569 unique original video IDs**, and **11 columns**: `Video_ID` plus ten binary behavioral labels. It is a multi-label task; a clip may have multiple positive labels.

| Split | Complete Annotation Set | Currently Metadata-Valid Clips |
|---|---:|---:|
| Train | 553 | 312 |
| Validation | 193 | 108 |
| Test | 182 | 107 |
| Total | 928 | 527 |

### Behavioral Labels

1. Absence or Avoidance of Eye Contact
2. Aggressive Behavior
3. Hyper- or Hyporeactivity to Sensory Input
4. Non-Responsiveness to Verbal Interaction
5. Non-Typical Language
6. Object Lining-Up
7. Self-Hitting or Self-Injurious Behavior
8. Self-Spinning or Spinning Objects
9. Upper Limb Stereotypies
10. Background

These annotations describe behaviors; they are not individual diagnostic labels.

## Current Progress

The following status reflects the latest reported local checks on **30 September 2026**. It is separate from the proposed research plan.

### Completed Checks

- Basic CSV cleaning and consistency checks completed.
- No missing values or exact duplicate rows reported; cleaning retained all 928 rows.
- Zero clip-ID and original-video-ID overlap across train, validation, and test.
- Combined split IDs match the complete annotation set, with no missing or extra IDs.
- Cleaned CSVs saved as `dataset_clean.csv`, `train_clean.csv`, `val_clean.csv`, and `test_clean.csv`.
- 296 available original video files reported, covering 529 annotated clips.
- Three incomplete `.part` files reported and excluded from completed media.
- Source metadata and timestamp checks completed using FFprobe.

| Source Check Result | Clips |
|---|---:|
| Metadata valid | 527 |
| Source video missing | 399 |
| Requested interval exceeds source duration | 2 |
| Total | 928 |

The clips `mkin5up15XM_46_56` and `mkin5up15XM_57_63` were excluded because their requested intervals exceed the available source duration. The check report was saved as `video_check_report.csv`.

### In Progress

FFmpeg clip extraction and verification are ongoing. The initial five trial clips passed audio-stream, video-stream, and duration checks. The final extraction count is pending.

Metadata-valid counts are not final successfully extracted clip counts. Feature extraction, model training, fusion, explainability, and dashboard development have not yet been completed.

## Proposed Methodology

1. **Dataset audit:** Record provenance, permissions, labels, missingness, available modalities, and participant grouping where provided.
2. **Preprocessing:** Segment and resample audio; sample video frames or derive visual features; clean and normalize structured inputs.
3. **Exploratory analysis:** Examine class distribution, modality availability, sequence lengths, and demographics where available.
4. **Unimodal baselines:** Evaluate individual modalities using suitable interpretable or lightweight models.
5. **Early fusion:** Concatenate compatible features before classification.
6. **Late fusion:** Combine modality-specific outputs, with the exact combination method documented per experiment.
7. **Temporal modeling:** Test LSTM, TCN, or lightweight attention only where sequence data and resources justify it.
8. **Explainability:** Use SHAP or comparable attribution methods and modality-contribution analysis.
9. **Ablation and robustness:** Remove modalities, inspect errors, and report fold variation or uncertainty where feasible.
10. **Prototype:** Display behavioral predictions, input modalities, scores, explanations, and limitations in a lightweight interface.

All model families and feature choices remain proposed until implemented and evaluated. Behavioral label columns are prediction targets, not input features.

## Planned Experiments

| Experiment | Input / Focus | Main Comparison |
|---|---|---|
| E1 | Structured behavioral features, if available | Interpretable baseline |
| E2 | Audio features or embeddings | Audio-only performance |
| E3 | Visual / temporal features | Visual-only performance |
| E4 | Audio and visual early fusion | Multimodal versus unimodal |
| E5 | Audio and visual late fusion | Fusion-strategy comparison |
| E6 | Compatible audio, visual, and behavioral inputs, if available | Combined model versus modality ablations |
| E7 | Best validated model | Feature and modality attribution |

## Evaluation and Reproducibility

- Per-label precision, recall, and F1; macro and micro F1 for multi-label evaluation.
- Per-label confusion matrices and error analysis.
- ROC-AUC where appropriate and computable for the evaluated labels.
- Grouped validation where grouping information is available, with class distribution reported.
- Training and validation data used for model selection and threshold tuning; test data reserved for final evaluation.
- Preprocessing fitted on training data only.
- Fixed seeds, documented dependencies, feature settings, and experiment configurations.
- No promised accuracy target; results will be reported as measured.

Current source-video separation does not prove participant-level separation. Participant identity may be unavailable, and this limitation must be documented rather than treated as resolved.

## Technology Stack

| Area | Tools | Implementation Status |
|---|---|---|
| Data preparation | Python, Pandas, NumPy | Used in current preparation workflow |
| Media processing | yt-dlp, FFmpeg, FFprobe | Used for downloads and clip preparation |
| Visualization | Matplotlib, Seaborn | Proposed analysis stack |
| Classical ML | Scikit-learn, XGBoost | Proposed |
| Deep learning | PyTorch or TensorFlow | To be selected as needed |
| Audio | Librosa, optional pretrained embeddings | Proposed |
| Vision | OpenCV, MediaPipe, optional pretrained encoders | Proposed |
| Explainability | SHAP or comparable attribution tools | Proposed |
| Prototype | Streamlit or FastAPI with a lightweight UI | Proposed |
| Collaboration | Git / GitHub | Repository created |

Setup and execution commands will be documented alongside the uploaded scripts and their verified dependencies.

## Expected Deliverables

- Dataset audit and dataset documentation.
- Reproducible preprocessing code and analysis notebooks.
- Unimodal baseline results.
- Early- and late-fusion comparisons.
- Explainability outputs and modality-ablation results.
- Lightweight demonstration interface.
- Final technical report and presentation.

## Indicative Timeline

| Period | Planned Work |
|---|---|
| Weeks 1-2 | Literature, dataset access, problem definition, and permissions audit |
| Weeks 3-4 | Preprocessing, exploratory analysis, and dataset documentation |
| Weeks 5-6 | Unimodal baselines and initial evaluation |
| Weeks 7-9 | Multimodal fusion and optional temporal experiments |
| Weeks 10-11 | Explainability, ablation, error analysis, and robustness |
| Weeks 12-13 | Prototype, technical report, and presentation |

This is the proposal schedule, not a claim that these milestones are complete.

## Privacy, Responsible Use, and Limitations

Use data only under applicable access permissions and licenses. Do not collect or publish identifiable participant recordings without the necessary consent, institutional approval, and supervision. Raw videos, extracted clips, credentials, authentication cookies, and private configuration are intended to remain outside this repository; exclusion must be configured before committing project files.

Available media represent a subset of the complete annotations. Class imbalance, source availability, recording conditions, and uncertain participant grouping may affect evaluation. Small-sample results must not be presented as evidence of clinical readiness.

Feature attribution describes model behavior; it does not establish causality or a clinical explanation. Model scores are not diagnoses or calibrated clinical risk estimates by default.

**This project is an academic behavioral-analysis prototype, not a medical diagnostic device or a replacement for professional assessment.**

## Proposal Attribution

This README is adapted from *An Explainable Multimodal AI Framework for Autism-Related Behavioral Analysis: Using Audio-Visual and Behavioral Features*, version 1.0, September 2026, prepared by **Hamza Rasheed** for the NETSOL trainee program.

The supplied proposal is marked confidential. Its full document and reference list are not reproduced here. Candidate dataset descriptions reflect the proposal and require independent access and licensing verification before use.

## Immediate Next Steps

1. Finish extraction and review the final verification report.
2. Create annotation subsets for successfully verified clips while retaining original split assignments.
3. Review class balance and available audio/visual inputs in those subsets.
4. Add preparation scripts, sanitized notebooks, reports, and dependency instructions to the repository.
5. Begin reproducible feature extraction and baseline experiments.
