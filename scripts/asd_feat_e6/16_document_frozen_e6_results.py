"""Document frozen E6 results and generate a dashboard page; no model fitting."""
from stage_runtime import stage_main

SPEC = {
    'name': '16_document_frozen_e6_results', 'source_cell': None,
    'fields': {
        'frozen_run': 'results/asd_feat/e6_reproduction/12_freeze_rgb_behavior_model/20261008T192951_063340Z',
        'evaluation_run': 'results/asd_feat/e6_reproduction/15_evaluate_frozen_rgb_behavior/20261008T194502_516518Z',
    },
    'requires': [('frozen_run', 'frozen_model_config.json')]
    + [('evaluation_run', name) for name in ['frozen_test_evaluation.json', 'frozen_test_comparison.csv', 'frozen_test_predictions.csv']],
    'packages': ['numpy', 'pandas', 'sklearn'], 'writes': True,
}


def run(args):
    from datetime import datetime, timezone
    import hashlib
    import json
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (f1_score, precision_score, recall_score,
        balanced_accuracy_score, accuracy_score, roc_auc_score,
        average_precision_score, matthews_corrcoef, confusion_matrix)

    def sha256(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    config_path = args.frozen_run / 'frozen_model_config.json'
    evaluation_path = args.evaluation_run / 'frozen_test_evaluation.json'
    comparison_path = args.evaluation_run / 'frozen_test_comparison.csv'
    predictions_path = args.evaluation_run / 'frozen_test_predictions.csv'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    evaluation = json.loads(evaluation_path.read_text(encoding='utf-8'))
    comparison = pd.read_csv(comparison_path)
    predictions = pd.read_csv(predictions_path)
    if evaluation['status'] != 'PASS' or not evaluation['official_e6_test_evaluated'] or evaluation['retraining'] or evaluation['threshold_tuning']:
        raise ValueError('Expected completed frozen inference without retraining or tuning.')
    if sha256(config_path) != evaluation['frozen_config_sha256'] or config['model_sha256'] != evaluation['model_sha256']:
        raise ValueError('Evaluation does not match frozen model/configuration.')
    if comparison['model'].duplicated().any():
        raise ValueError('Duplicate comparison rows.')
    if predictions['filename'].isna().any() or predictions['filename'].duplicated().any() or predictions['sub_id'].isna().any():
        raise ValueError('Missing prediction metadata or duplicate recording IDs.')
    y = predictions['target'].to_numpy()
    probability = predictions['rgb_behavior_probability'].to_numpy(dtype=float)
    threshold = float(config['threshold'])
    prediction = (probability >= threshold).astype(int)
    if not np.isfinite(probability).all() or not ((probability >= 0)&(probability <= 1)).all() or set(np.unique(y)) != {0, 1}:
        raise ValueError('Invalid saved predictions or labels.')
    if not np.array_equal(prediction, predictions['rgb_behavior_prediction'].to_numpy()):
        raise ValueError('Saved predictions differ from frozen threshold.')
    if len(predictions) != evaluation['test_recordings'] or predictions['sub_id'].nunique() != evaluation['test_participants']:
        raise ValueError('Evaluation support differs from saved prediction table.')
    positive_participants = int(predictions.loc[predictions['target'].eq(1), 'sub_id'].nunique())
    if positive_participants != evaluation['positive_test_participants'] or int(y.sum()) != evaluation['positive_test_recordings']:
        raise ValueError('Positive support differs from evaluation evidence.')
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    recalculated = {
        'macro_f1': f1_score(y, prediction, labels=[0, 1], average='macro', zero_division=0),
        'asd_f1': f1_score(y, prediction, zero_division=0),
        'precision': precision_score(y, prediction, zero_division=0),
        'recall': recall_score(y, prediction, zero_division=0),
        'specificity': float(tn/(tn+fp)), 'balanced_accuracy': balanced_accuracy_score(y, prediction),
        'accuracy': accuracy_score(y, prediction), 'roc_auc': roc_auc_score(y, probability),
        'average_precision': average_precision_score(y, probability), 'mcc': matthews_corrcoef(y, prediction),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
    }
    rows = comparison[comparison['model'].eq('rgb_behavior_pca32_lr')]
    if len(rows) != 1:
        raise ValueError('Selected model comparison row is missing or duplicated.')
    score = rows.iloc[0]
    for name, value in recalculated.items():
        if not np.isclose(float(score[name]), value, rtol=1e-7, atol=1e-8):
            raise ValueError('Saved metric does not reproduce: '+name)
    for row in evaluation['metrics']:
        saved = comparison[comparison['model'].eq(row['model'])]
        if len(saved) != 1:
            raise ValueError('JSON/CSV model disagreement.')
        for name, value in row.items():
            if name != 'model' and not np.isclose(float(saved.iloc[0][name]), float(value), rtol=1e-7, atol=1e-8):
                raise ValueError('JSON/CSV metric disagreement: '+name)

    def table(frame):
        lines = ['| '+' | '.join(frame.columns)+' |', '| '+' | '.join(['---']*len(frame.columns))+' |']
        for values in frame.itertuples(index=False, name=None):
            cells = [f'{v:.4f}' if isinstance(v, float) else str(v) for v in values]
            lines.append('| '+' | '.join(cells)+' |')
        return '\n'.join(lines)

    limits = (
        f'The {evaluation["positive_test_recordings"]} positive test recordings represent '
        f'{positive_participants} participant(s). Repeated recordings are correlated; '
        'these recording-level results do not establish generalization across positive participants.\n\n'
        'The official test had previously been evaluated for E1. It is not an untouched test set. '
        'The E6 candidate was selected after exploratory development comparisons, without nested selection validation. '
        'Human-coded behavioral features are required. Clinical diagnosis, disease-stage assessment and treatment recommendations are not validated.'
    )
    results = f'''# ASD-FEAT E6 Frozen Research Evaluation

## Model and Protocol

The selected research candidate combines 1024 pooled I3D RGB features and 12 human-coded behavioral features. Audio is excluded from this selected candidate. The broader E6 development study also evaluated audio and three-branch models.

The model was fitted on {config['development_recordings']} development recordings from {config['development_participants']} participants. It uses branch-specific PCA with 32 RGB components, standardized behavioral features and balanced logistic regression. The fixed threshold is {threshold:.1f}. The observed mean development-fold ASD F1 for the selected candidate was {config['mean_fold_asd_f1']:.4f}; this selection score is not an independent performance estimate for the selected winner.

## Recording-Level Official Test Results

Test support: {evaluation['test_recordings']} recordings, {evaluation['test_participants']} participants, {evaluation['positive_test_recordings']} positive recordings.

{table(comparison)}

The selected model identified {int(tp)} positive recordings, missed {int(fn)}, and produced {int(fp)} false positives. The dummy reference predicts the development positive prevalence and has zero ASD F1 at threshold {threshold:.1f}. Its high overall accuracy reflects class imbalance, not positive-case detection.

## Interpretation Limits

{limits}

## Verification and Evidence

Saved test metrics were recalculated from saved predictions and reproduced. Model fitting and threshold tuning were not performed during test evaluation or documentation. The frozen configuration and source model were preserved.

- Frozen model run: `{args.frozen_run.relative_to(args.project).as_posix()}`
- Evaluation run: `{args.evaluation_run.relative_to(args.project).as_posix()}`
- Model SHA-256: `{config['model_sha256']}`
'''
    card = f'''# Selected E6 Research Model Card

- Model: RGB plus human-coded behavior, PCA32 and balanced logistic regression.
- Task: Binary current-visit ASD annotation prediction, at recording level.
- Inputs: {config['input_feature_count']} features: 1024 pooled I3D RGB features and 12 human-coded behavioral measurements.
- Development fit: {config['development_recordings']} recordings, {config['development_participants']} participants, {config['positive_recordings']} positive recordings.
- Selection: Highest observed mean ASD F1 among exploratory development CV and branch-ablation candidates.
- Decision threshold: {threshold:.1f}, fixed before this E6 test inference.
- Artifact: `{config['model_artifact']}`.
- Model SHA-256: `{config['model_sha256']}`.
- Test ASD F1: {score['asd_f1']:.4f}.
- Test precision / recall: {score['precision']:.4f} / {score['recall']:.4f}.
- Test balanced accuracy: {score['balanced_accuracy']:.4f}.
- Test ROC-AUC / average precision: {score['roc_auc']:.4f} / {score['average_precision']:.4f}.

## Intended Scope

Exploratory research and demonstration using released feature arrays and human-coded behavioral annotations. A raw video alone does not provide all required inputs through this pipeline. This is separate from the AV-ASD multilabel behavior model.

## Limitations

{limits}

No validated clinical diagnosis, disease-stage assignment or treatment guidance is provided. Further independent participant-level validation is required before claims beyond exploratory research.
'''
    # Export only aggregate evidence; participant identifiers remain in local results.
    evidence = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'PASS',
        'selected_model': 'rgb_behavior_pca32_lr', 'branches': ['rgb', 'behavior'],
        'threshold': threshold, 'feature_count': config['input_feature_count'],
        'development_recordings': config['development_recordings'],
        'development_participants': config['development_participants'],
        'test_recordings': evaluation['test_recordings'], 'test_participants': evaluation['test_participants'],
        'positive_test_recordings': evaluation['positive_test_recordings'], 'positive_test_participants': positive_participants,
        'comparison': comparison.to_dict(orient='records'), 'metric_recalculation': 'PASS',
        'model_sha256': config['model_sha256'],
        'source_sha256': {p.relative_to(args.project).as_posix(): sha256(p) for p in [config_path, evaluation_path, comparison_path, predictions_path]},
        'interpretation_limits': limits, 'scope': 'Aggregate research documentation; no fitting, tuning or new predictions',
    }
    page = '''from pathlib import Path
import json
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "docs/ASD_FEAT_E6_FROZEN_TEST_EVIDENCE.json"
st.title("E6 Selected Research Model: Frozen Test Results")
if not PATH.is_file():
    st.error("Frozen E6 aggregate evidence is missing.")
    st.stop()
evidence = json.loads(PATH.read_text(encoding="utf-8"))
comparison = pd.DataFrame(evidence["comparison"])
primary = comparison.loc[comparison["model"].eq(evidence["selected_model"])].iloc[0]
st.caption("RGB + human-coded behavior | Fixed threshold: " + str(evidence["threshold"]))
columns = st.columns(4)
columns[0].metric("Test recordings", evidence["test_recordings"])
columns[1].metric("Participants", evidence["test_participants"])
columns[2].metric("ASD F1", f"{primary['asd_f1']:.4f}")
columns[3].metric("Balanced accuracy", f"{primary['balanced_accuracy']:.4f}")
st.warning(evidence["interpretation_limits"])
st.subheader("Frozen recording-level comparison")
st.dataframe(comparison, hide_index=True, width="stretch")
st.caption("ROC-AUC is a ranking metric. F1 is not an accuracy percentage. High dummy accuracy reflects class imbalance.")
st.subheader("Selected-model prediction outcomes")
columns = st.columns(4)
for column, key, label in zip(columns, ["tp", "fp", "fn", "tn"], ["True positives", "False positives", "False negatives", "True negatives"]):
    column.metric(label, int(primary[key]))
st.subheader("Model and verification")
st.write("Input features:", evidence["feature_count"])
st.write("Development recordings:", evidence["development_recordings"])
st.write("Development participants:", evidence["development_participants"])
st.write("Saved metric recalculation:", evidence["metric_recalculation"])
st.caption("This page displays saved aggregate research results; it performs no inference or model fitting.")
'''
    compile(page, '7_E6_Frozen_Test.py', 'exec')
    args.output.mkdir(parents=True, exist_ok=False)
    for name, contents in [('ASD_FEAT_E6_FROZEN_RESULTS.md', results), ('ASD_FEAT_E6_SELECTED_MODEL_CARD.md', card),
                           ('ASD_FEAT_E6_FROZEN_TEST_EVIDENCE.json', json.dumps(evidence, indent=2)), ('7_E6_Frozen_Test.py', page)]:
        destination = args.output / name
        destination.write_text(contents, encoding='utf-8')
        print('Saved:', destination)
    print('Saved test metric recalculation: PASS')
    print('Aggregate documentation and dashboard source ready. No models or predictions changed.')


if __name__ == '__main__':
    stage_main(__file__, run, SPEC)
