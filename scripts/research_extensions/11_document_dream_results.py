"""Recalculate DREAM activity CV metrics and install aggregate documentation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--activity-run', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (f1_score, accuracy_score, balanced_accuracy_score,
        precision_score, recall_score, roc_auc_score, confusion_matrix)
    root = args.project.resolve()
    run = args.activity_run.resolve()
    validation = json.loads((run / 'validation.json').read_text(encoding='utf-8'))
    require(validation['status'] == 'PASS' and validation['diagnostic_evaluation_performed'] is False,
            'Completed activity experiment evidence required.')
    config = json.loads((run / 'experiment_config.json').read_text(encoding='utf-8'))
    labels = ['IM', 'JA', 'TT']
    models = ['activity_lr', 'activity_rf', 'dummy_prior']
    require(config['target_labels'] == labels, 'Unexpected activity class ordering.')
    folds = pd.read_csv(run / 'fold_results.csv')
    oof = pd.read_csv(run / 'oof_predictions.csv')
    assignments = pd.read_csv(run / 'fold_assignments.csv')
    inference = pd.read_csv(run / 'saved_inference_validation.csv')
    require(len(folds) == 15 and not folds.duplicated(['model', 'fold']).any() and
        folds['participant_overlap'].eq(0).all(), 'Invalid fold evidence.')
    require(len(inference) == 15 and inference['status'].eq('PASS').all() and
        not inference.duplicated(['model', 'fold']).any(), 'Incomplete saved inference evidence.')
    require(len(assignments) == 2615 and assignments['member'].is_unique and
        assignments['participant_id'].nunique() == 56, 'Unexpected activity cohort.')
    require(assignments.groupby('participant_id')['evaluation_fold'].nunique().eq(1).all(), 'Participant leakage.')
    require(set(oof['model']) == set(models) and len(oof) == 7845, 'OOF model/row coverage mismatch.')
    reference = assignments.set_index('member')

    def metrics(table):
        y = table['target_activity']
        pred = table['prediction']
        scores = table[[f'score_{label}' for label in labels]].to_numpy(dtype=float)
        require(y.isin(labels).all() and pred.isin(labels).all(), 'Unknown target/prediction label.')
        require(np.isfinite(scores).all() and (scores >= 0).all() and (scores <= 1).all() and
            np.allclose(scores.sum(axis=1), 1, rtol=1e-6, atol=1e-7), 'Invalid fitted class scores.')
        require(np.array_equal(pred.to_numpy(), np.asarray(labels)[scores.argmax(axis=1)]), 'Argmax prediction mismatch.')
        return {'macro_f1': f1_score(y, pred, labels=labels, average='macro', zero_division=0),
            'weighted_f1': f1_score(y, pred, labels=labels, average='weighted', zero_division=0),
            'macro_precision': precision_score(y, pred, labels=labels, average='macro', zero_division=0),
            'macro_recall': recall_score(y, pred, labels=labels, average='macro', zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(y, pred), 'accuracy': accuracy_score(y, pred),
            'macro_ovr_roc_auc': roc_auc_score(y, scores, labels=labels, multi_class='ovr', average='macro')}

    summaries, pooled = [], []
    matrices = {}
    for model in models:
        table = oof[oof['model'] == model]
        require(len(table) == 2615 and table['member'].is_unique and set(table['member']) == set(reference.index), 'Incomplete session coverage.')
        aligned = reference.loc[table['member']]
        for column in ['participant_id', 'evaluation_fold', 'target_activity']:
            require(np.array_equal(table[column].to_numpy(), aligned[column].to_numpy()), 'OOF assignment mismatch: ' + column)
        require(set(table['evaluation_fold']) == {1, 2, 3, 4, 5}, 'Invalid evaluation folds.')
        recalculated = []
        for fold in range(1, 6):
            held = table[table['evaluation_fold'] == fold]
            fit_participants = set(assignments.loc[assignments['evaluation_fold'] != fold, 'participant_id'])
            require(not fit_participants.intersection(held['participant_id']), 'Participant overlap.')
            row = metrics(held)
            saved = folds[(folds['model'] == model) & (folds['fold'] == fold)]
            require(len(saved) == 1, 'Missing model fold.')
            for key, value in row.items():
                require(np.isclose(value, float(saved.iloc[0][key]), rtol=1e-8, atol=1e-10), 'Metric recalculation failed: ' + model + '/' + key)
            recalculated.append(row)
        frame = pd.DataFrame(recalculated)
        summary = {'model': model}
        for key in frame.columns:
            summary[key + '_mean'] = float(frame[key].mean())
            summary[key + '_std'] = float(frame[key].std(ddof=1))
        summaries.append(summary)
        pooled.append({'model': model, **metrics(table)})
        matrices[model] = confusion_matrix(table['target_activity'], table['prediction'], labels=labels).tolist()
    require(set(zip(inference['model'], inference['fold'])) == {(model, fold) for model in models for fold in range(1, 6)}, 'Saved inference set mismatch.')
    evidence = {'status': 'METRICS_RECALCULATED_PASS', 'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'team': ['Muhammad Danial', 'Hamza Rasheed', 'Muhammad Freeh'],
        'scope': 'Exploratory DREAM therapy-activity recognition; distinct from autism classification',
        'sessions': 2615, 'participants': 56, 'features': 54, 'folds': 5,
        'labels': labels, 'activity_support': assignments['target_activity'].value_counts().to_dict(),
        'summary': summaries, 'pooled_metrics': pooled, 'pooled_confusion_matrices': matrices,
        'participant_overlap': 0, 'oof_coverage': 'PASS', 'saved_inference_evidence': 'PASS',
        'source_run': str(run), 'source_sha256': {name: sha(run / name) for name in
            ['validation.json', 'experiment_config.json', 'fold_results.csv', 'oof_predictions.csv',
             'fold_assignments.csv', 'saved_inference_validation.csv']},
        'diagnostic_evaluation_performed': False,
        'limitations': ['Selected quality-filtered subset, not all 3121 sessions or all 61 participants.',
            'All participants have ASD; therapy activity is the target, not diagnostic status.',
            'Static position summaries cannot establish therapeutic efficacy or clinical severity.',
            'Task-specific recording setup, posture and missingness may confound performance.',
            'Fold SD is descriptive variability, not a confidence interval or significance test.',
            'No independent test cohort or validation of either previous frozen classifier.']}
    docs = root / 'docs'
    page = root / 'dashboard/pages/12_DREAM_Activity_Results.py'
    docs.mkdir(parents=True, exist_ok=True); page.parent.mkdir(parents=True, exist_ok=True)
    targets = [docs / 'DREAM_ACTIVITY_RESULTS.md', docs / 'DREAM_ACTIVITY_EVIDENCE.json', page]
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    backup = root / 'documentation_backups' / ('dream_results_' + stamp)
    for path in targets:
        if path.exists():
            destination = backup / path.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    text = ['# DREAM therapy-activity development results', '', 'Team: Muhammad Danial, Hamza Rasheed, Muhammad Freeh.', '',
        'This separate exploratory task classifies IM/JA/TT therapy activity labels from 54 body-relative position descriptors. It uses 2,615 quality-filtered sessions from 56 participants. Five stratified group folds keep participants separate; all 15 fold models passed saved-inference checks.', '',
        '## Mean participant-fold results', '', '| Model | Macro F1 | Fold SD | Balanced accuracy | Accuracy | Macro OVR ROC-AUC |', '|---|---:|---:|---:|---:|---:|']
    for row in summaries:
        text.append('| ' + row['model'] + ' | ' + ' | '.join(f"{row[key]:.4f}" for key in ['macro_f1_mean','macro_f1_std','balanced_accuracy_mean','accuracy_mean','macro_ovr_roc_auc_mean']) + ' |')
    text += ['', '## Pooled out-of-fold results', '', '| Model | Macro F1 | Balanced accuracy | Accuracy | Macro OVR ROC-AUC |', '|---|---:|---:|---:|---:|']
    for row in pooled:
        text.append('| ' + row['model'] + ' | ' + ' | '.join(f"{row[key]:.4f}" for key in ['macro_f1','balanced_accuracy','accuracy','macro_ovr_roc_auc']) + ' |')
    text += ['', '## Interpretation', '',
        'Both learned models exceed the dummy baseline in mean Macro F1. Random Forest has the higher mean Macro F1 and accuracy in this run; Logistic Regression has higher balanced accuracy/macro recall. The majority TT class makes accuracy alone inadequate. The small LR/RF Macro F1 difference has not been tested for statistical significance.', '',
        'Each model uses fit participants only. LR scaling is fitted within each fold; its class weights and RF balanced-subsample weights are computed from training data. Predictions use argmax class scores, without threshold tuning. Fold means and pooled metrics are different summaries and must not be interchanged.', '',
        'This is activity recognition among children who already have ASD. It is not autism diagnosis, severity prediction, therapy-effect estimation, or external validation of the AV-ASD/ASD-FEAT frozen models. Original frozen artifacts remain preserved.', '',
        '## Limits and verification', '',
        'The cohort was filtered for coordinate availability. The descriptors summarize positions relative to the shoulder centre; they do not normalize body size or rotation and do not model temporal order. Sensor configuration, session posture and missingness may confound labels. No untouched independent test cohort was evaluated.', '',
        'This documentation stage recalculated all fold and pooled metrics from saved OOF rows and verified participant/label/fold alignment. It loaded no models and changed no predictions. Source hashes and pooled confusion matrices appear in DREAM_ACTIVITY_EVIDENCE.json.', '',
        'Dataset source: https://doi.org/10.5878/17p8-6k13. Observed catalogue license: CC BY-NC-SA 4.0. Keep session-level files local; these documents contain aggregate research findings.']
    targets[0].write_text('\n'.join(text)+'\n', encoding='utf-8')
    targets[1].write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    source = '''"""DREAM aggregate activity results; no model inference."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st

st.title("DREAM: Therapy-Activity Recognition")
path = Path(__file__).resolve().parents[2] / "docs/DREAM_ACTIVITY_EVIDENCE.json"
if not path.is_file():
    st.info("Run the DREAM results documentation stage first.")
    st.stop()
evidence = json.loads(path.read_text(encoding="utf-8"))
if evidence.get("status") != "METRICS_RECALCULATED_PASS":
    st.error("Validated activity evidence is unavailable.")
    st.stop()
st.caption("IM / JA / TT activity labels | 2,615 sessions | 56 participants | five participant folds")
summary = pd.DataFrame(evidence["summary"])
columns = ["model", "macro_f1_mean", "macro_f1_std", "balanced_accuracy_mean", "accuracy_mean", "macro_ovr_roc_auc_mean"]
st.dataframe(summary[columns].round(4), hide_index=True, width="stretch")
st.bar_chart(summary.set_index("model")[["macro_f1_mean"]], horizontal=True)
st.subheader("Pooled out-of-fold metrics")
st.dataframe(pd.DataFrame(evidence["pooled_metrics"]).round(4), hide_index=True, width="stretch")
st.write("Both learned models improve Macro F1 over the dummy baseline. Random Forest has higher Macro F1 in this run; LR has higher balanced accuracy. No significance claim is made.")
model = st.selectbox("Pooled confusion matrix", summary["model"].tolist())
labels = evidence["labels"]
st.caption("Rows: true activity. Columns: predicted activity.")
st.dataframe(pd.DataFrame(evidence["pooled_confusion_matrices"][model], index=labels, columns=labels), width="stretch")
st.info("All participants have ASD. This page reports therapy-activity recognition, not diagnosis, severity or treatment effectiveness. It does not externally validate the AV-ASD or ASD-FEAT classifiers.")
st.caption("Team: Muhammad Danial, Hamza Rasheed, Muhammad Freeh")
'''
    compile(source, str(page), 'exec')
    page.write_text(source, encoding='utf-8')
    for path in targets:
        print('Saved:', path)
    print('All fold metrics, participant alignment and dashboard syntax: PASS')
    print('No models loaded, fitted or replaced. No predictions changed.')


if __name__ == '__main__':
    main()
