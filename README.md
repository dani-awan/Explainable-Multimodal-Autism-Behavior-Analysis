# An Explainable Multimodal AI Framework for Autism-Related Behavioral Analysis

> **An AI/ML research and prototype project for analyzing autism-related behavioral patterns using audio, visual/behavioral, and structured behavioral features.**

---

## 📌 Project Overview

This project proposes an **Explainable Multimodal Artificial Intelligence (AI) Framework** for autism-related behavioral analysis.

The framework investigates whether combining multiple observable behavioral modalities — including **audio, visual/behavioral, and structured behavioral features** — can provide more useful and informative analysis than relying on a single modality.

The project will systematically compare:

* Unimodal vs. multimodal learning
* Early vs. late multimodal fusion
* Static vs. temporal behavioral modeling where supported by the data
* Explainable AI techniques for understanding model outputs
* Model performance under modality ablation and robustness experiments

The project is designed as an **AI/ML research and prototype system**, not as an autonomous medical diagnostic system.

---

## 🎯 Problem Statement

Autism-related behavioral patterns can vary considerably between individuals and may involve differences in social communication, interaction, attention, sensory responses, and repetitive or restricted behaviors.

Artificial intelligence can potentially analyze observable behavioral signals from sources such as video, audio, and structured behavioral observations. However, existing research faces several challenges:

* Many studies focus on only one modality.
* Available datasets can be small or heterogeneous.
* Dataset imbalance and demographic bias can affect model performance.
* Multimodal models are not always compared against strong unimodal baselines.
* Model predictions may be difficult to interpret.
* Reproducibility and external validation remain important challenges.
* Privacy is particularly important when working with child audio/video data.

Therefore, this project investigates a reproducible and explainable multimodal framework that evaluates complementary behavioral signals while keeping claims proportional to the available evidence.

---

## 🎯 Aim

> **To design and evaluate an explainable multimodal AI framework that integrates audio-visual and structured behavioral features for autism-related behavioral analysis.**

---

## 🚀 Objectives

The project will pursue the following objectives:

1. Develop reproducible preprocessing pipelines for available behavioral modalities.
2. Establish unimodal baseline models for audio, visual/behavioral, and structured behavioral data.
3. Compare early-fusion and late-fusion multimodal learning strategies.
4. Investigate temporal modeling when sequential behavioral data are available.
5. Apply Explainable AI (XAI) techniques to identify influential behavioral features and modality contributions.
6. Conduct modality ablation experiments to determine the contribution of individual modalities.
7. Evaluate robustness using appropriate cross-validation and participant-level data splitting.
8. Develop a lightweight demonstration dashboard for communicating model outputs and explanations.
9. Document dataset limitations, privacy considerations, fairness concerns, reproducibility requirements, and future external-validation needs.

---

# 🔬 Research Questions

### RQ1

How effectively can individual audio, visual, and structured behavioral modalities characterize autism-related behavioral patterns?

### RQ2

Does multimodal fusion provide more useful behavioral analysis than individual modalities under the same evaluation protocol?

### RQ3

How do early-fusion and late-fusion strategies compare for multimodal behavioral analysis?

### RQ4

Which behavioral features and modalities contribute most strongly to model predictions?

### RQ5

How stable are model results under cross-validation and modality ablation?

---

# 💡 Research Hypotheses

### H1

Multimodal representations can capture complementary information that may not be fully represented by a single behavioral modality.

### H2

A multimodal fusion model may improve at least one primary evaluation metric compared with relevant unimodal baselines when evaluated using the same protocol.

### H3

Explainable AI techniques can identify modality-specific behavioral signals that materially influence model predictions.

### H4

Removing an informative modality will produce a measurable change in model performance.

---

# 🧩 Multimodal Framework

The proposed framework follows the general pipeline:

```text
                    ┌─────────────────────┐
                    │   Research Dataset  │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
        ┌──────────┐     ┌──────────┐    ┌──────────────┐
        │  Audio   │     │  Visual  │    │ Behavioral   │
        │ Features │     │ Features │    │   Features   │
        └────┬─────┘     └────┬─────┘    └──────┬───────┘
             │                │                  │
             ▼                ▼                  ▼
        ┌──────────┐     ┌──────────┐    ┌──────────────┐
        │ Audio    │     │ Visual/  │    │ Structured   │
        │ Model    │     │ Temporal │    │ Model        │
        └────┬─────┘     │ Model    │    └──────┬───────┘
             │           └────┬─────┘           │
             │                │                 │
             └────────────────┼─────────────────┘
                              │
                     ┌────────▼────────┐
                     │ Multimodal      │
                     │ Fusion          │
                     └────────┬────────┘
                              │
                     ┌────────▼────────┐
                     │ Prediction /    │
                     │ Analysis Output │
                     └────────┬────────┘
                              │
                 ┌────────────▼────────────┐
                 │ Explainable AI (XAI)   │
                 │ Feature + Modality     │
                 │ Contributions          │
                 └────────────┬────────────┘
                              │
                     ┌────────▼────────┐
                     │ Evaluation &    │
                     │ Ablation Study  │
                     └─────────────────┘
```

---

# 📊 Modalities

The framework considers three main types of information.

## 1. Audio

Potential audio information may include:

* Acoustic features
* Speech-related features
* Prosodic characteristics
* Audio embeddings
* Pretrained speech representations

Possible tools:

* Librosa
* Pretrained audio/speech encoders
* Standard machine learning classifiers

---

## 2. Visual / Behavioral

Potential visual information may include:

* Facial/visual behavioral features
* Pose information
* Attention-related features
* Gaze-related information where available
* Body movement
* Temporal behavioral patterns
* Precomputed visual embeddings

Possible tools:

* OpenCV
* MediaPipe
* Pretrained visual encoders
* Lightweight temporal models

---

## 3. Structured Behavioral Features

Structured features may include available behavioral observations or derived features such as:

* Look-Face
* Look-Object
* Smile
* Social Vocalization
* Other dataset-specific behavioral indicators

The exact features will depend on the selected dataset and its available variables.

---

# 🔀 Multimodal Fusion

Two primary fusion strategies will be investigated.

## Early Fusion

Features from multiple modalities are combined before classification.

```text
Audio Features ─────┐
                    │
Visual Features ────┼──► Feature Concatenation ──► Model ──► Output
                    │
Behavior Features ──┘
```

### Purpose

To investigate whether a joint feature representation improves behavioral analysis.

---

## Late Fusion

Each modality is processed separately before combining model outputs.

```text
Audio ─────► Audio Model ──────┐
                               │
Visual ─────► Visual Model ────┼──► Fusion ──► Final Output
                               │
Behavior ──► Behavior Model ───┘
```

### Purpose

To compare independent modality-specific learning with joint feature learning.

---

# ⏱️ Temporal Modeling

Where the dataset provides ordered behavioral sequences, temporal modeling may be investigated.

Potential approaches include:

* LSTM
* Temporal Convolutional Networks (TCN)
* Lightweight temporal attention

Temporal modeling will remain an optional research component and will only be used when the available dataset supports meaningful sequential analysis.

---

# 🤖 Machine Learning Approach

The project will begin with relatively interpretable and resource-efficient baseline models.

### Structured Data

Potential models:

* Logistic Regression
* Random Forest
* XGBoost

### Audio

Potential approach:

```text
Audio
  ↓
Preprocessing
  ↓
Feature Extraction / Embedding
  ↓
Classifier
  ↓
Evaluation
```

### Visual / Temporal Data

Potential approaches:

* Lightweight neural networks
* Temporal models
* Pretrained feature extractors

### Multimodal Data

Potential approaches:

* Early fusion
* Late fusion
* Lightweight multimodal architectures

Model complexity will be proportional to dataset size and available computational resources.

---

# 🔍 Explainable AI

Explainability is an important component of this project.

The system should not only provide an output but also investigate:

> **"Which features or modalities contributed to the model's prediction?"**

Potential techniques include:

* SHAP
* Feature importance
* Modality contribution analysis
* Local and global explanations

Example:

```text
Model Output
     │
     ▼
┌──────────────────────────┐
│ Explainable AI           │
├──────────────────────────┤
│ Audio contribution       │
│ Visual contribution      │
│ Behavioral contribution  │
│ Feature importance       │
└──────────────────────────┘
```

### Important

Feature importance or SHAP values will be interpreted as **model evidence**, not as proof of causation or clinical explanation.

---

# 🧪 Ablation Study

Ablation experiments will investigate how much each modality contributes to the overall system.

Example:

| Experiment         | Audio | Visual | Behavioral |
| ------------------ | ----: | -----: | ---------: |
| Full Model         |     ✓ |      ✓ |          ✓ |
| Without Audio      |     ✗ |      ✓ |          ✓ |
| Without Visual     |     ✓ |      ✗ |          ✓ |
| Without Behavioral |     ✓ |      ✓ |          ✗ |
| Audio Only         |     ✓ |      ✗ |          ✗ |
| Visual Only        |     ✗ |      ✓ |          ✗ |
| Behavioral Only    |     ✗ |      ✗ |          ✓ |

This helps determine whether the modalities provide complementary information.

---

# 📈 Evaluation

The following evaluation measures may be used depending on the dataset and task:

* Precision
* Recall
* F1-score
* ROC-AUC where appropriate
* Confusion Matrix
* Cross-validation
* Fold-to-fold performance variation
* Modality ablation results

Participant-level splitting will be used whenever participant identifiers are available to reduce the risk of data leakage.

---

# 🧪 Core Experiments

The planned experiments are:

| Experiment | Input                       | Approach                                 | Purpose                             |
| ---------- | --------------------------- | ---------------------------------------- | ----------------------------------- |
| E1         | Behavioral                  | Logistic Regression / RF / XGBoost       | Structured baseline                 |
| E2         | Audio                       | Audio features / embeddings + classifier | Audio contribution                  |
| E3         | Visual                      | Visual / temporal model                  | Visual contribution                 |
| E4         | Audio + Visual              | Early Fusion                             | Multimodal comparison               |
| E5         | Audio + Visual              | Late Fusion                              | Fusion comparison                   |
| E6         | Audio + Visual + Behavioral | Best validated fusion                    | Primary multimodal experiment       |
| E7         | Best Model                  | XAI                                      | Feature and modality interpretation |

---

# 🗂️ Dataset Strategy

The project will initially focus on publicly available or appropriately derived research datasets.

Potential datasets include:

### AV-ASD

Audio-visual behavioral data relevant to autism-related behavior recognition.

**Use:** Primary candidate for audio-visual experiments, subject to access and licensing verification.

---

### ASD-FEAT

A derived infant video feature dataset containing behavioral features such as:

* Look-Face
* Look-Object
* Smile
* Social Vocalization

**Use:** Complementary structured/derived behavioral feature experiments.

---

### DREAM Dataset

A dataset involving processed behavioral information such as:

* 3D skeleton information
* Head orientation
* Eye gaze
* Metadata

**Use:** Optional temporal/behavioral experiment where the required data are accessible.

---

## ⚠️ Dataset Access Policy

Before using any dataset, the project will verify:

* Data license
* Access requirements
* Dataset provenance
* Available modalities
* Participant counts
* Label definitions
* Missing data
* Privacy restrictions

Private clinical data, identifiable child recordings, or newly collected participant data will **not** be used without appropriate ethics approval, informed consent, data governance, and supervision.

---

# 🔐 Privacy & Ethical Considerations

Because the project may involve behavioral data from children or vulnerable populations, privacy and responsible AI are central considerations.

The project will follow these principles:

* Use public or appropriately licensed research datasets.
* Avoid uploading sensitive raw datasets to GitHub.
* Avoid personally identifiable information.
* Use participant-level train/test separation where possible.
* Document dataset limitations and demographic coverage.
* Avoid unsupported clinical claims.
* Do not present model outputs as medical diagnoses.
* Do not interpret feature importance as causal evidence.
* Document data provenance and preprocessing steps.

---

# ⚠️ Project Scope & Disclaimer

This project is an **AI/ML research and prototype framework for autism-related behavioral analysis**.

It is **not intended to**:

* Replace clinicians or healthcare professionals.
* Provide an autonomous medical diagnosis.
* Make clinical decisions.
* Claim that a particular feature causes autism.
* Be deployed as a clinical diagnostic device without appropriate validation and regulatory/ethical processes.

Model outputs should be interpreted as research/model-analysis results rather than clinical diagnoses.

---

# 🏗️ Proposed Project Structure

```text
An-Explainable-Multimodal-AI-Framework-for-Autism-Related-Behavioral-Analysis/
│
├── README.md
├── .gitignore
├── requirements.txt
│
├── docs/
│   ├── proposal/
│   ├── literature-review/
│   └── presentations/
│
├── data/
│   └── README.md
│
├── notebooks/
│   ├── 01_eda/
│   ├── 02_audio/
│   ├── 03_visual/
│   └── 04_multimodal/
│
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── audio/
│   ├── visual/
│   ├── behavioral/
│   ├── multimodal/
│   ├── explainability/
│   └── evaluation/
│
├── models/
│
├── results/
│   ├── figures/
│   ├── tables/
│   └── experiments/
│
└── app/
    └── dashboard/
```

The structure may evolve as the research progresses.

---

# 🛠️ Technology Stack

### Programming

* Python

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* XGBoost

### Deep Learning

* PyTorch or TensorFlow

### Audio Processing

* Librosa
* Pretrained audio/speech embeddings where appropriate

### Computer Vision

* OpenCV
* MediaPipe
* Pretrained visual encoders where appropriate

### Explainable AI

* SHAP
* Feature attribution methods

### Prototype

* Streamlit or FastAPI
* Lightweight frontend/interface

### Development

* Git
* GitHub

---

# 📅 Project Timeline

The project is planned over approximately **13 weeks**.

| Weeks | Activities                                                                          |
| ----- | ----------------------------------------------------------------------------------- |
| 1–2   | Literature review, dataset verification, problem definition, ethics/licensing audit |
| 3–4   | Data preprocessing, EDA, dataset documentation                                      |
| 5–6   | Unimodal baseline models and evaluation                                             |
| 7–9   | Multimodal fusion and temporal experiments                                          |
| 10–11 | Explainability, ablation, error analysis, robustness                                |
| 12–13 | Prototype, final report, presentation                                               |

---

# 📦 Expected Deliverables

The project is expected to produce:

* Research proposal
* Literature review
* Dataset audit/documentation
* Reproducible preprocessing pipeline
* Unimodal baseline models
* Multimodal fusion model
* Explainability module
* Ablation study
* Evaluation results
* Lightweight demonstration dashboard
* Technical documentation
* Final research report
* Project presentation

---

# ⚠️ Research Risks

| Risk                              | Mitigation                                                               |
| --------------------------------- | ------------------------------------------------------------------------ |
| Dataset access limitations        | Verify access during Weeks 1–2 and maintain derived-feature alternatives |
| Small dataset                     | Participant-level CV and careful interpretation                          |
| Class imbalance                   | Class-aware metrics and appropriate validation                           |
| Data leakage                      | Participant-level train/test separation                                  |
| Limited computing resources       | Lightweight models and pretrained embeddings                             |
| Multimodal synchronization issues | Use aligned/derived features or late fusion                              |
| Privacy concerns                  | Public/derived datasets and no identifiable raw data                     |
| Overclaiming                      | Behavioral-analysis terminology and explicit limitations                 |
| Scope expansion                   | Lock minimum viable experiments before optional extensions               |

---

# 🔬 Reproducibility

To improve reproducibility, the project will aim to document:

* Dataset versions
* Dataset sources
* Preprocessing procedures
* Feature definitions
* Model configurations
* Random seeds
* Train/validation/test strategy
* Evaluation metrics
* Experiment configurations
* Software dependencies
* Model versions
* Known limitations

---

# 👥 Team

**Project Lead / Researcher**

* Hamza Rasheed

**Team Members**

* Add team member names here

> Team roles and responsibilities will be documented as the project progresses.

---

# 📚 References

1. Rakotomanana, R., & Rouhafzay, A. (2025). *A Scoping Review of AI-Based Approaches for Detecting Autism Traits Using Voice and Behavioral Data*. Bioengineering. DOI: 10.3390/bioengineering12111136.

2. Deng et al. (2024). *Hear Me, See Me, Understand Me: Audio-Visual Autism Behavior Recognition*. arXiv:2406.02554.

3. Liaqat et al. (2026). *ASD-FEAT: A Multi-Modal Infant Video-Derived Dataset for Early ASD Risk Prediction*. Zenodo. DOI: 10.5281/zenodo.22261227.

4. Billing, E. A. (2019). *Open DREAM Dataset*. DREAM Project Dataset Release.

5. Wang et al. (2025). *Systematic Review and Meta-analysis of Video-Assisted Telehealth Autism Screening/Diagnosis*. Digital Health. DOI: 10.1177/20552076251386705.

6. Bae et al. (2025). *Multimodal AI for risk stratification in autism spectrum disorder: integrating voice and screening tools*. npj Digital Medicine, 8, 538. DOI: 10.1038/s41746-025-01914-6.

7. *Challenges in translating AI-driven ASD/ADHD diagnosis: A methodological systematic review*. (2026).

8. Agrawal, A., & Agrawal, A. (2025). *Explainable AI in early autism detection: a literature review of interpretable machine learning approaches*. Discover Mental Health, 5, 98. DOI: 10.1007/s44192-025-00232-3.

9. *A Scoping Review of Machine Learning and Deep Learning Methods for Autism Spectrum Disorder Diagnosis and Analysis*. (2026).

10. *Data-Driven Approaches for Autism Detection: A Comprehensive Review of Machine Learning Algorithms and Datasets*. (2026).

---

# 📌 Current Research Pipeline

```text
Literature Review
       ↓
Dataset Audit
       ↓
Data Preprocessing
       ↓
EDA
       ↓
Unimodal Baselines
       ↓
Audio + Visual + Behavioral
       ↓
Early / Late Fusion
       ↓
Temporal Modeling
       ↓
Explainable AI
       ↓
Ablation & Robustness
       ↓
Evaluation
       ↓
Prototype Dashboard
       ↓
Final Report & Presentation
```

---

# 🚀 Project Vision

The long-term objective is to develop a **reproducible, explainable, privacy-conscious multimodal AI research framework** that helps investigate how different observable behavioral signals can be combined for autism-related behavioral analysis.

The project emphasizes:

**Multimodal Learning + Explainability + Reproducibility + Responsible AI**

rather than treating model prediction as a standalone clinical diagnosis.

---

## 📄 Project Information

**Project:** An Explainable Multimodal AI Framework for Autism-Related Behavioral Analysis

**Organization:** NETSOL Technologies Pakistan

**Program:** AI & Machine Learning Trainee Program

**Project Type:** Applied AI/ML Research & Prototype

**Duration:** 13 Weeks

**Version:** 1.0

**Date:** September 2026

**Status:** Research & Development
