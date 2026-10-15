"""Participant-disjoint exploratory DREAM therapy-activity classification."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import warnings


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--feature-run', type=Path, required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.base import clone
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
        precision_score, recall_score, roc_auc_score, classification_report, confusion_matrix)
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    root = args.project.resolve()
    folder = args.feature_run.resolve()
    evidence_path = folder / 'feature_validation.json'
    evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
    require(evidence['status'] == 'FEATURE_EXTRACTION_COMPLETE' and evidence['feature_sessions'] == 2615,
            'Expected completed 2615-session feature extraction.')
    X_path, metadata_path = folder / 'activity_X.csv', folder / 'activity_metadata.csv'
    require(sha(X_path) == evidence['feature_sha256'] and sha(metadata_path) == evidence['metadata_sha256'], 'Feature inputs changed.')
    X_table = pd.read_csv(X_path)
    metadata = pd.read_csv(metadata_path)
    columns = evidence['feature_columns']
    labels = ['IM', 'JA', 'TT']
    require(len(columns) == 54 and X_table.columns.tolist() == ['member', *columns], 'Feature schema/order mismatch.')
    for table in [X_table, metadata]:
        require(table['member'].notna().all() and table['member'].is_unique, 'Invalid session IDs.')
    require(set(X_table['member']) == set(metadata['member']), 'Session set mismatch.')
    metadata = metadata.set_index('member').loc[X_table['member']].reset_index()
    require(metadata['target_activity'].isin(labels).all() and metadata['participant_id'].notna().all(), 'Invalid activity or participant.')
    require(metadata['participant_id'].nunique() == 56, 'Unexpected participant support.')
    X = X_table[columns]
    require(np.isfinite(X.to_numpy(dtype=float)).all(), 'Non-finite features. Do not silently impute this cohort.')
    y = metadata['target_activity'].to_numpy()
    groups = metadata['participant_id'].to_numpy()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(splitter.split(X, y, groups))
    assignment = np.zeros(len(X), dtype=int)
    support = []
    for fold, (fit, evaluate) in enumerate(splits, start=1):
        overlap = set(groups[fit]) & set(groups[evaluate])
        require(not overlap and set(y[fit]) == set(labels) and set(y[evaluate]) == set(labels), 'Fold participant/class support invalid.')
        require(not assignment[evaluate].any(), 'Repeated evaluation assignment.')
        assignment[evaluate] = fold
        support.append({'fold': fold, 'fit_sessions': len(fit), 'evaluation_sessions': len(evaluate),
            'fit_participants': len(set(groups[fit])), 'evaluation_participants': len(set(groups[evaluate])),
            'participant_overlap': 0, **{f'evaluation_{label}': int((y[evaluate] == label).sum()) for label in labels}})
    require((assignment > 0).all(), 'Incomplete fold assignment.')
    metadata['evaluation_fold'] = assignment
    require(metadata.groupby('participant_id')['evaluation_fold'].nunique().eq(1).all(), 'Participant assigned multiple folds.')
    print('DREAM activity CV preflight: PASS | 2615 sessions | 56 participants | 54 features', flush=True)
    print('Target: IM / JA / TT therapy activity | no diagnostic endpoint', flush=True)
    print(pd.DataFrame(support).to_string(index=False), flush=True)
    if not args.run:
        print('Inspection only. Use --run to fit new activity models.')
        return
    output = root / 'results/research_extensions/dream_activity_cv' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output.mkdir(parents=True, exist_ok=False)
    models = {
        'activity_lr': make_pipeline(StandardScaler(), LogisticRegression(C=1.0,
            class_weight='balanced', solver='lbfgs', max_iter=3000, random_state=42)),
        'activity_rf': RandomForestClassifier(n_estimators=200, max_depth=8,
            min_samples_leaf=5, max_features='sqrt', class_weight='balanced_subsample', n_jobs=2, random_state=42),
        'dummy_prior': DummyClassifier(strategy='prior')}
    config = {'scope': 'Exploratory DREAM therapy-activity classification; not ASD diagnosis or severity',
        'target_labels': labels, 'prediction_rule': 'argmax fitted class scores',
        'primary_descriptive_metric': 'mean participant-fold macro F1',
        'split': 'StratifiedGroupKFold, five folds, shuffle=True, random_state=42',
        'inputs': '54 mean/SD body-relative position descriptors; no participant, ADOS, therapy condition or demographic predictors',
        'models': {name: {key: str(value) for key, value in model.get_params(deep=True).items()} for name, model in models.items()},
        'input_hashes': {'features': sha(X_path), 'metadata': sha(metadata_path), 'validation': sha(evidence_path)},
        'feature_run': str(folder), 'feature_columns': columns, 'python': sys.version,
        'versions': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'joblib']},
        'script_sha256': sha(Path(__file__)),
        'limitations': ['Distinct activity labels and input representation from AV-ASD and ASD-FEAT.',
            'All participants have ASD; no autism-negative class.', 'Quality-filtered subset excludes unavailable measurements.',
            'Recording setup, task-specific posture and missingness can confound the result.',
            'No nested CV, prospective external evaluation or causal interpretation.']}
    (output / 'experiment_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    metadata.to_csv(output / 'fold_assignments.csv', index=False)
    pd.DataFrame(support).to_csv(output / 'fold_support.csv', index=False)
    print('Output:', output, flush=True)

    def metrics(truth, probability):
        pred = np.asarray(labels)[probability.argmax(axis=1)]
        return {'macro_f1': f1_score(truth, pred, labels=labels, average='macro', zero_division=0),
            'weighted_f1': f1_score(truth, pred, labels=labels, average='weighted', zero_division=0),
            'macro_precision': precision_score(truth, pred, labels=labels, average='macro', zero_division=0),
            'macro_recall': recall_score(truth, pred, labels=labels, average='macro', zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(truth, pred), 'accuracy': accuracy_score(truth, pred),
            'macro_ovr_roc_auc': roc_auc_score(truth, probability, labels=labels, average='macro', multi_class='ovr')}

    rows, tables, checks = [], [], []
    for fold, (fit, evaluate) in enumerate(splits, start=1):
        for name, template in models.items():
            print(f'Fold {fold}/5: {name}', flush=True)
            model = clone(template)
            with threadpool_limits(limits=2), warnings.catch_warnings():
                warnings.simplefilter('error', ConvergenceWarning)
                model.fit(X.iloc[fit], y[fit])
            require(model.classes_.tolist() == labels, 'Class score order mismatch.')
            score = model.predict_proba(X.iloc[evaluate])
            require(score.shape == (len(evaluate), 3) and np.isfinite(score).all() and
                    np.allclose(score.sum(axis=1), 1.0), 'Invalid class probabilities.')
            path = output / f'{name}_fold{fold}.joblib'
            joblib.dump({'model': model, 'input_columns': columns, 'target_labels': labels,
                         'evaluation_fold': fold, 'scope': config['scope']}, path)
            reproduced = joblib.load(path)['model'].predict_proba(X.iloc[evaluate])
            require(np.allclose(score, reproduced, rtol=1e-12, atol=1e-12), 'Saved model inference failed.')
            checks.append({'model': name, 'fold': fold, 'maximum_score_difference': float(np.max(np.abs(score-reproduced))), 'status': 'PASS'})
            rows.append({'model': name, 'fold': fold, 'participant_overlap': 0, **metrics(y[evaluate], score)})
            table = metadata.iloc[evaluate].copy()
            table['model'] = name
            table['prediction'] = np.asarray(labels)[score.argmax(axis=1)]
            for index, label in enumerate(labels):
                table[f'score_{label}'] = score[:, index]
            tables.append(table)
        pd.DataFrame(rows).to_csv(output / 'fold_results.csv', index=False)
        print(f'Fold {fold}/5 complete | participant overlap: 0', flush=True)
    predictions = pd.concat(tables, ignore_index=True)
    pooled = []
    for name in models:
        table = predictions[predictions['model'] == name]
        require(len(table) == len(X) and table['member'].is_unique and set(table['member']) == set(metadata['member']), 'OOF coverage failed.')
        score = table[[f'score_{label}' for label in labels]].to_numpy()
        pooled.append({'model': name, **metrics(table['target_activity'], score)})
        pd.DataFrame(classification_report(table['target_activity'], table['prediction'], labels=labels,
            output_dict=True, zero_division=0)).T.to_csv(output / f'{name}_classification_report.csv')
        pd.DataFrame(confusion_matrix(table['target_activity'], table['prediction'], labels=labels),
            index=pd.Index(labels, name='true_activity'), columns=labels).to_csv(output / f'{name}_confusion_matrix.csv')
    predictions.to_csv(output / 'oof_predictions.csv', index=False)
    scores = list(metrics(y, np.full((len(y), 3), 1/3)))
    summary = pd.DataFrame(rows).groupby('model')[scores].agg(['mean', 'std'])
    summary.columns = ['_'.join(value) for value in summary.columns]
    summary.to_csv(output / 'cv_summary.csv')
    pd.DataFrame(pooled).to_csv(output / 'pooled_oof_metrics.csv', index=False)
    pd.DataFrame(checks).to_csv(output / 'saved_inference_validation.csv', index=False)
    require(sha(X_path) == config['input_hashes']['features'] and sha(metadata_path) == config['input_hashes']['metadata'], 'Inputs changed during execution.')
    validation = {'status': 'PASS', 'sessions_per_model': len(X), 'participants': 56,
        'fold_models': 15, 'participant_overlap': 0, 'oof_coverage': 'PASS', 'saved_inference': 'PASS',
        'diagnostic_evaluation_performed': False, 'scope': config['scope'],
        'completed_at_utc': datetime.now(timezone.utc).isoformat()}
    (output / 'validation.json').write_text(json.dumps(validation, indent=2), encoding='utf-8')
    print('\nMean fold results:\n' + summary.round(4).to_string(), flush=True)
    print('\nPooled OOF results:\n' + pd.DataFrame(pooled).round(4).to_string(index=False), flush=True)
    print('Activity OOF coverage, participant separation and saved inference: PASS')
    print('Saved:', output)
    print('This measures therapy-activity recognition, not autism diagnosis. Original frozen models preserved.')


if __name__ == '__main__':
    main()
