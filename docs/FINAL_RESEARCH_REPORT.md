# An Explainable Multimodal AI Framework for Autism-Related Behavioral Analysis

Final research prototype report

## Research scope and abstract

Team: Hamza Rasheed, Muhammad Danial and Muhammad Freeh. Applied AI and machine learning research prototype. Prepared 9 October 2026.

This report consolidates two separate experiments: AV-ASD clip-level recognition of ten behavioral labels and ASD-FEAT recording-level classification of current-visit ASD-associated annotations. It explains the implemented models, empirical comparisons, reproducibility evidence and limitations. The framework supports research demonstration using prepared features. It does not establish clinical diagnosis, severity or future outcome prediction.

Abstract. We compared lightweight audio, visual and structured behavioral classifiers, feature fusion and probability fusion. The available AV-ASD subset contained 527 clips split into 312 training, 108 validation and 107 test clips. The validation-selected combined logistic-regression baseline reached test Macro F1 0.2937 and Micro F1 0.3544. ASD-FEAT development comparisons used 1,510 aligned recordings from 259 participants with five participant-disjoint evaluation folds. RGB plus human-coded behavior gave the highest observed mean development ASD F1, 0.3142, among the reported exploratory configurations. Its frozen evaluation on 162 recordings reached ASD F1 0.3571, precision 0.2273 and recall 0.8333. All twelve positive test recordings came from one participant, so the test cannot establish generalization across positive participants. Exact linear contribution checks, same-environment reproduction and a feature-upload dashboard support inspection and demonstration. Independent validation, verified media extraction and licensing clarification remain necessary for broader use.

Evidence basis. Numeric results come from supplied experiment reports and their consolidated review, not from new training during preparation of this report. The separately supplied handoff records dashboard validation and Windows reproduction. This is a submission report for an exploratory prototype, rather than a clinical validation study.

## Research questions and dataset design

The research asks whether combining observable information improves performance under the same evaluation protocol, how early and late fusion compare, and which input branches influence fitted scores. The study tests these questions with resource-efficient classifiers and makes model complexity proportional to the available data and laptop resources.

| Track | Development | Evaluation | Inputs |
| --- | --- | --- | --- |
| AV-ASD | 312 train; 108 validation | 107 test clips | 44 audio + 14 visual |
| ASD-FEAT E1 | 1513 recordings; 259 participants | 162 recordings; 30 participants | 12 human-coded values |
| ASD-FEAT E6 | 1510 recordings; 259 participants | 162 recordings; 30 participants | 224 audio + 1024 RGB + 12 behavior |

AV-ASD and ASD-FEAT have different targets, sample units and protocols. Their scores must not be pooled or presented as one project accuracy. AV-ASD predicts behavior labels on clips. ASD-FEAT predicts the released current-visit asd_longitudinal annotation on recordings, using previously extracted representations and human-coded measurements.

AV-ASD availability introduces selection effects. From 928 candidate clips, 399 lacked an available source video and two exceeded the available video duration. The usable subset has 527 clips. Clip and original-video overlap checks were zero between splits, but original-video separation does not prove participant separation. Background has zero positive annotations in the available test subset. Seven silent clips were retained and flagged.

ASD-FEAT E1 uses 1,513 development recordings. E6 aligns 1,510 recordings across branches, excluding three recordings without released audio features consistently across comparisons. All 113 positive development recordings and 23 participants with positive recordings remain covered. Development and official-test participant overlap is zero. Five development evaluation folds contain 3 to 6 participants with positive recordings each.

Privacy and licensing. Identifiable media, feature rows and model artifacts remain local. The supplied E1 report records custom research-use terms; that is not evidence of unrestricted redistribution. AV-ASD reuse permissions remain unverified in the supplied dataset card. Publication and redistribution require release-specific permission checks.

## Methods and experiment governance

AV-ASD audio summaries cover MFCC statistics, RMS energy and zero-crossing rate. The handcrafted visual branch summarizes color, brightness, contrast, edges and sampled-frame differences. These features do not automatically measure clinical symptoms. StandardScaler and one-vs-rest balanced logistic regression form the interpretable baseline. Random forest and late-fusion baselines provide additional comparisons. The primary combined_lr model was selected with validation Macro F1 and retains threshold 0.30.

ASD-FEAT audio inputs contain 224 pooled summaries. The RGB branch contains 1,024 pooled features from released I3D arrays. The structured branch contains twelve human-coded rates and proportions describing smile, face-looking, object-looking, vocalization, social smile and social vocalization. These are related representations of observed behavior, not three independent sensor channels.

Within each participant-grouped fold, preprocessing fits only on the fitting portion. Audio and RGB branches use StandardScaler and randomized PCA with 32 components per branch. The combined representation receives output scaling and balanced logistic regression with C=1, lbfgs, max_iter=3000 and random_state=42. The threshold is fixed at 0.5. Late fusion uses equal weights on modality-specific probabilities.

Retrained ablation removes a branch and fits a reduced model on the same records and folds. Missing-branch robustness instead keeps each full fold model fixed and replaces one raw input branch with that fitting fold's feature means. These experiments answer different questions and cannot be treated as causal attribution or real-world missing-input validation.

Model selection is exploratory. RGB plus behavior was chosen after development comparison and ablation, without nested selection validation. The selected development score may therefore be optimistic. The official test had already been evaluated for E1, and the overall research program must disclose this exposure. No thresholds or models were changed to produce the final reports.

## ASD FEAT development results

RGB plus human-coded behavior, represented by the without_audio ablation, gave the highest observed mean ASD F1. Its ASD F1 exceeded the full three-branch early-fusion model in four of five folds. Its mean ROC-AUC and average precision also exceeded the full model. This supports selecting that candidate for the prototype under the tested protocol. It does not establish that audio is generally uninformative.

| Candidate | Mean ASD F1 | Mean ROC-AUC | Mean AP |
| --- | --- | --- | --- |
| Audio PCA32 LR | 0.2057 | 0.7071 | 0.1807 |
| Behavior LR on E6 rows | 0.2140 | 0.6922 | 0.1741 |
| RGB PCA32 LR | 0.2729 | 0.7730 | 0.2280 |
| Three-branch late fusion | 0.2793 | 0.7915 | 0.2703 |
| Three-branch early fusion | 0.2920 | 0.7721 | 0.2370 |
| RGB plus behavior | 0.3142 | 0.8070 | 0.2828 |

The three-branch late-fusion model has higher mean ROC-AUC and average precision than the three-branch early-fusion model, while early fusion has higher mean ASD F1. Thus the preferred fusion strategy depends on the metric. Mean fold scores and metrics on pooled out-of-fold predictions are separate summaries.

Ablation supports the value of human-coded behavior in this representation: removing it reduces mean ASD F1 from 0.2920 to 0.2237. Removing RGB gives 0.2585. The fixed-model robustness simulation gives mean ASD F1 0.3037 with audio imputed, 0.2042 with behavior imputed and 0.1507 with RGB imputed, compared with 0.2920 with all branches available. The strongest simulated degradation follows RGB replacement.

Five folds share overlapping fitting sets. Fold consistency is descriptive and is not a significance test. Positive support is limited, repeated recordings are correlated, and the selected winner has no nested independent development estimate.

## Frozen ASD FEAT evaluation

The selected RGB plus behavior model uses 1,036 raw numeric inputs, branch-specific RGB PCA32, standardized behavioral values, output scaling and balanced logistic regression. It fits on all 1,510 development recordings and uses threshold 0.5. The saved-model round-trip inference check passed before official test inference.

| Metric | E1 behavior LR | Selected RGB + behavior |
| --- | --- | --- |
| ASD F1 | 0.2683 | 0.3571 |
| Macro F1 | 0.5102 | 0.6114 |
| Precision | 0.1571 | 0.2273 |
| Recall | 0.9167 | 0.8333 |
| Specificity | 0.6067 | 0.7733 |
| Balanced accuracy | 0.7617 | 0.8033 |
| ROC-AUC | 0.8661 | 0.8728 |
| Average precision | 0.4514 | 0.2608 |
| TP / FP / FN / TN | 11 / 59 / 1 / 91 | 10 / 34 / 2 / 116 |

The official test contains 162 recordings from 30 participants. Twelve positive recordings represent only one participant. The selected model identifies 10 positive recordings, misses two and produces 34 false positives. These are recording counts, not unique people.

Compared descriptively with E1, the selected model has higher observed F1 and precision and fewer false positives, but lower recall and lower average precision. Neither model establishes clinical utility. The dummy's 0.9259 overall accuracy exceeds both models because it predicts the majority class and detects no positive recordings. Overall accuracy alone is inadequate for this imbalanced task.

The one-positive-participant limitation prevents credible estimates of sensitivity across new positive participants. E1 participant bootstrap retained 1,275 of 2,000 attempted replicates after excluding single-class samples. Its identical recall bounds reflect repeatedly sampling the same positive participant, not certainty. No E6 population-level confidence interval is asserted here.

Frozen artifact SHA-256: 2c8e9d9c126e9a7a8134eb9792550254de3747847b96aeaca1e396bae57327d4. Documentation recalculated saved metrics from existing predictions. This report does not rerun official test inference.

## AV ASD results and reproduction

The validation-selected combined logistic-regression baseline retains primary status even though late fusion has the highest observed test F1. Choosing a replacement using these inspected test results would change the evaluation protocol. The test contains 107 clips from 56 original-video groups.

| Model | Threshold | Macro F1 | Micro F1 | Hamming loss |
| --- | --- | --- | --- | --- |
| Audio LR | 0.30 | 0.2854 | 0.3432 | 0.4364 |
| Visual LR | 0.30 | 0.3090 | 0.3499 | 0.6112 |
| Combined LR primary | 0.30 | 0.2937 | 0.3544 | 0.3916 |
| Combined random forest | 0.30 | 0.2239 | 0.3348 | 0.2748 |
| Late fusion LR | 0.40 | 0.3111 | 0.3602 | 0.4150 |

Primary combined LR precision is 0.2516, recall 0.5990 and exact-match accuracy 0.0093. Error analysis records 342 false-positive label assignments, 77 false negatives and one clip with every label correct. Background-positive recognition cannot be assessed because its test support is zero.

Source-group bootstrap with 2,000 iterations reports conditional 95% intervals of 0.2498 to 0.3305 for Macro F1, 0.3000 to 0.4064 for Micro F1 and 0.3631 to 0.4216 for Hamming loss. These describe sample uncertainty conditional on the fixed model and available source groups, not clinical or participant-level generalization.

Saved AV-ASD inference reproduced probabilities to a maximum absolute difference of 1.11e-16 with zero prediction mismatches. Isolated retraining from existing features reproduced the three LR models' validation probabilities with zero maximum difference. E6 standalone Windows CV reproduction also matched 1,510 out-of-fold rows and 30 fold-result rows exactly. These verify same-environment reproduction from prepared inputs, not independent reconstruction of all media and preprocessing.

## Explainability and research demonstration

Exact linear explanations multiply transformed feature values by fitted logistic-regression coefficients. Their sum plus the intercept reconstructs the fitted decision logit. Summing transformed contributions within the branch indices gives RGB and behavior contributions. These are explanations of fitted scores. They are not SHAP values, observed symptom measurements or causal effects.

The supplied E6 run records successful explanation reconstruction and out-of-fold reproduction for all 1,510 development recordings. A selected three-branch example has contributions of -0.9150 for audio, -0.2410 for RGB and +0.4054 for behavior, with score 0.0283. This example concerns the three-branch CV model, not the selected two-branch frozen model.

The Streamlit Research Analysis page verifies a trusted local frozen artifact, accepts an existing test-recording demonstration or a compatible one-row CSV, restores the frozen feature order and exports HTML and JSON reports. It requires 1,024 compatible pooled I3D RGB features and twelve human-coded values. Raw photo, audio and video extraction is not implemented.

Recording 1 replay and its exported CSV both produce the displayed score 16.61%, below threshold 0.5. The uploaded HTML report records 1,036 finite values and contributions -0.740099 for RGB and +0.969165 for behavior, with intercept -1.842596 and reconstructed logit -1.613530. This score is an uncalibrated model output, not a validated chance of autism. Neither threshold outcome establishes or excludes autism.

Windows validation reports ten successful checks: valid and reordered columns are accepted; missing or extra columns, multiple rows, an empty table, missing values, infinite values, text values and duplicate column names are rejected. Those checks exercise the validation function with synthetic tables. They do not constitute browser automation or validation of arbitrary feature provenance.

## Discussion conclusion and evidence register

The framework supplies reproducible exploratory evidence on two distinct research tasks. The ASD-FEAT comparisons favor pooled RGB plus human-coded behavior under the tested classifier and representation, while the AV-ASD primary baseline remains limited. Multimodal fusion does not uniformly outperform each unimodal baseline across metrics, and additional branches can reduce performance.

Remaining scientific limitations include imbalanced labels, correlated recordings, single-positive-participant test support, exploratory winner selection, prior test exposure, download-dependent AV-ASD inclusion and unverified participant separation on AV-ASD. Demographic fairness, external validity, calibrated clinical risk and fully automated human-behavior extraction are not established.

Future work should prioritize independent evaluation across multiple positive participants and verified feature provenance. Nested development evaluation can assess candidate selection, and calibration would require a documented development-only protocol followed by suitable independent evaluation. Raw-media integration requires the original I3D checkpoint, matching preprocessing and a validated source for the twelve behavioral measurements. Temporal sequence models remain optional and require preserved ordered data.

Conclusion. The implemented research prototype compares feature branches and fusion, records ablation and simulated robustness, reproduces prepared-input experiments and exports model explanations. Its present deliverable is a local feature-based research dashboard. It should not be presented as a clinical assessment or an automated upload-and-diagnose service.

Evidence register. The supplied FINAL_EXPERIMENT_REVIEW.txt consolidates ASD_FEAT_E1_RESULTS.md, ASD_FEAT_E6_DEVELOPMENT_RESULTS.md, ASD_FEAT_E6_ROBUSTNESS.md, ASD_FEAT_E6_FROZEN_RESULTS.md, ASD_FEAT_E6_SELECTED_MODEL_CARD.md, MODEL_CARD.md and TRAINING_REPRODUCTION.md. The supplied work handoff records RESEARCH_ANALYSIS_UI_VALIDATION.md, RESEARCH_INPUT_VALIDATION.json, standalone Windows reproduction and explanation outputs. This consolidation checks consistency of supplied reports, not a fresh audit of every local artifact.

Primary literature. Deng, S., et al. (2024). Hear Me, See Me, Understand Me: Audio-Visual Autism Behavior Recognition. arXiv:2406.02554. https://arxiv.org/abs/2406.02554. The paper introduces audio-visual behavior recognition and an AV-ASD dataset. Our handcrafted subset experiment is not a reproduction of its foundation-model architecture.

ASD-FEAT release reference. The supplied release reports identify DOI 10.5281/zenodo.22261227. This report preserves that identifier as supplied; live release metadata and custom licensing were not independently retrieved during finalization. The proposal's remaining bibliography entries require verification before a publication-ready paper. No unverified author lists or literature results are invented.
