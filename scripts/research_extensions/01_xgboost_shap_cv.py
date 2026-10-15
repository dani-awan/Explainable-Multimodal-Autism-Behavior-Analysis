"""Exploratory development-only XGBoost CV and held-out Tree SHAP audit."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--run', action='store_true', help='Fit new development-fold models.')
    args = parser.parse_args()
    root = args.project.resolve()
    folder = root / 'dataset/external/asd_feat/derived/e6_inputs'
    paths = {n: folder / n for n in ['development_combined_X.csv',
        'development_metadata.csv', 'development_y.csv', 'input_schema.json']}
    for path in paths.values():
        require(path.is_file(), f'Missing input: {path}')
    import joblib
    import numpy as np
    import pandas as pd
    import shap
    import xgboost as xgb
    from scipy.special import expit
    from sklearn.metrics import (f1_score, precision_score, recall_score,
        balanced_accuracy_score, roc_auc_score, average_precision_score,
        confusion_matrix, matthews_corrcoef)
    from threadpoolctl import threadpool_limits

    schema = json.loads(paths['input_schema.json'].read_text(encoding='utf-8-sig'))
    groups = schema['feature_groups']
    require({k: len(v) for k, v in groups.items()} ==
            {'audio': 224, 'rgb': 1024, 'behavior': 12}, 'Unexpected feature groups.')
    columns = groups['audio'] + groups['rgb'] + groups['behavior']
    require(len(set(columns)) == len(columns), 'Duplicate schema features.')
    require(sha(paths['development_combined_X.csv']) ==
        schema['output_feature_sha256']['development_combined_X.csv'], 'Feature hash mismatch.')
    features = pd.read_csv(paths['development_combined_X.csv'])
    metadata = pd.read_csv(paths['development_metadata.csv'])
    targets = pd.read_csv(paths['development_y.csv'])
    require(features.columns.tolist() == ['filename', *columns], 'Feature order mismatch.')
    ids = features['filename'].tolist()
    for name, table in [('features', features), ('metadata', metadata), ('targets', targets)]:
        require(table['filename'].notna().all() and table['filename'].is_unique,
                f'{name}: missing or duplicate recording IDs.')
        require(set(table['filename']) == set(ids), f'{name}: recording set mismatch.')
    metadata = metadata.set_index('filename').loc[ids].reset_index()
    targets = targets.set_index('filename').loc[ids].reset_index()
    require(targets['target'].isin([0, 1]).all(), 'Non-binary targets.')
    y = targets['target'].to_numpy(dtype=int)
    require(np.array_equal(y, metadata['target'].to_numpy()), 'Target disagreement.')
    require(metadata['sub_id'].notna().all(), 'Missing participants.')
    folds = metadata['evaluation_fold'].to_numpy()
    require(np.isin(folds, [1, 2, 3, 4, 5]).all(), 'Unexpected fold IDs.')
    require(metadata.groupby('sub_id')['evaluation_fold'].nunique().eq(1).all(),
            'Participant leakage across folds.')
    require(len(features) == 1510 and metadata['sub_id'].nunique() == 259,
            'Unexpected development cohort.')
    require(np.isfinite(features[columns].to_numpy(dtype=float)).all(), 'Non-finite features.')
    require(int(y.sum()) == 113, 'Unexpected positive support.')
    candidates = {'behavior_xgb': groups['behavior'],
        'rgb_behavior_xgb': groups['rgb'] + groups['behavior'],
        'three_branch_xgb': columns}
    settings = dict(n_estimators=200, max_depth=3, learning_rate=0.05,
        min_child_weight=5, subsample=1.0, colsample_bytree=1.0,
        reg_lambda=5.0, objective='binary:logistic', eval_metric='logloss',
        tree_method='hist', device='cpu', n_jobs=2, random_state=42,
        base_score=0.5)
    print('Development preflight: PASS | 1510 recordings | 259 participants', flush=True)
    print('Fixed threshold: 0.5 | 3 candidates | 5 existing participant folds', flush=True)
    if not args.run:
        print('Inspection only. Use --run for new experiments.')
        return
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = root / 'results/research_extensions/xgboost_shap' / stamp
    output.mkdir(parents=True, exist_ok=False)
    config = {'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Exploratory development-only comparison; official test excluded',
        'threshold': 0.5, 'selection_endpoint': 'mean fold ASD F1 (descriptive)',
        'settings': settings, 'candidate_columns': candidates,
        'weighting': 'fit-fold negative count / positive count',
        'shap': {'method': 'TreeExplainer', 'model_output': 'raw',
                 'feature_perturbation': 'tree_path_dependent',
                 'evaluation': 'held-out recordings only', 'additivity_atol': 0.0001},
        'input_hashes': {name: sha(path) for name, path in paths.items()},
        'script_sha256': sha(Path(__file__)), 'python': sys.version,
        'versions': {n: importlib.metadata.version(n) for n in
                     ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'shap', 'joblib']},
        'limitations': ['Exploratory extensions after previous test exposure.',
            'No nested CV or significance test; repeated recordings are correlated.',
            'Uncalibrated outputs are not diagnostic probabilities.',
            'Tree SHAP describes fitted scores, not causal or clinical effects.']}
    (output / 'experiment_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print('Output:', output, flush=True)

    def metrics(truth, score):
        prediction = (score >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
        return {'asd_f1': f1_score(truth, prediction, zero_division=0),
            'macro_f1': f1_score(truth, prediction, average='macro', zero_division=0),
            'precision': precision_score(truth, prediction, zero_division=0),
            'recall': recall_score(truth, prediction, zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(truth, prediction),
            'roc_auc': roc_auc_score(truth, score),
            'average_precision': average_precision_score(truth, score),
            'mcc': matthews_corrcoef(truth, prediction),
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)}

    rows, validation, oof = [], [], []
    importance = {name: np.zeros(len(cols)) for name, cols in candidates.items()}
    for fold in range(1, 6):
        fit = folds != fold
        evaluate = folds == fold
        overlap = set(metadata.loc[fit, 'sub_id']) & set(metadata.loc[evaluate, 'sub_id'])
        require(not overlap, f'Fold {fold}: participant overlap.')
        require(set(y[fit]) == {0, 1} and set(y[evaluate]) == {0, 1}, 'Single-class fold.')
        weight = float((y[fit] == 0).sum() / (y[fit] == 1).sum())
        for name, cols in candidates.items():
            print(f'Fold {fold}/5: {name} | fitting and explaining...', flush=True)
            train = features.loc[fit, cols]
            held_out = features.loc[evaluate, cols]
            model = xgb.XGBClassifier(**settings, scale_pos_weight=weight)
            with threadpool_limits(limits=2):
                model.fit(train, y[fit])
                score = model.predict_proba(held_out)[:, 1]
                margin = model.predict(held_out, output_margin=True)
                explainer = shap.TreeExplainer(model, model_output='raw',
                    feature_perturbation='tree_path_dependent')
                values = np.asarray(explainer.shap_values(held_out, check_additivity=True))
            require(values.shape == held_out.shape, 'Unexpected SHAP shape.')
            base = float(np.asarray(explainer.expected_value).item())
            reconstructed = base + values.sum(axis=1)
            require(np.isfinite(values).all() and np.isfinite(score).all(), 'Non-finite output.')
            require(np.allclose(reconstructed, margin, rtol=1e-5, atol=1e-4),
                    f'{name}: SHAP margin reconstruction failed.')
            require(np.allclose(expit(reconstructed), score, rtol=1e-5, atol=1e-5),
                    f'{name}: SHAP probability reconstruction failed.')
            bundle = {'model': model, 'input_columns': cols, 'threshold': 0.5,
                'evaluation_fold': fold, 'target': 'target', 'fit_participants':
                sorted(metadata.loc[fit, 'sub_id'].astype(str).unique().tolist())}
            path = output / f'{name}_fold{fold}.joblib'
            joblib.dump(bundle, path)
            reproduced = joblib.load(path)['model'].predict_proba(held_out)[:, 1]
            require(np.array_equal(reproduced, score), 'Saved model inference mismatch.')
            model.save_model(output / f'{name}_fold{fold}.ubj')
            np.savez_compressed(output / f'{name}_fold{fold}_shap.npz', values=values,
                base_value=base, columns=np.asarray(cols, dtype=str),
                filenames=np.asarray(metadata.loc[evaluate, 'filename'], dtype=str))
            importance[name] += np.abs(values).sum(axis=0)
            rows.append({'model': name, 'fold': fold, 'evaluation_recordings': int(evaluate.sum()),
                'participant_overlap': 0, 'scale_pos_weight': weight, **metrics(y[evaluate], score)})
            validation.append({'model': name, 'fold': fold, 'recordings': int(evaluate.sum()),
                'maximum_margin_difference': float(np.max(np.abs(reconstructed - margin))),
                'saved_inference': 'PASS', 'shap_reconstruction': 'PASS'})
            table = metadata.loc[evaluate, ['filename', 'sub_id', 'evaluation_fold', 'target']].copy()
            table['model'] = name
            table['score'] = score
            table['prediction'] = (score >= 0.5).astype(int)
            oof.append(table)
            pd.DataFrame(rows).to_csv(output / 'fold_results.csv', index=False)
            pd.DataFrame(validation).to_csv(output / 'shap_validation.csv', index=False)
    predictions = pd.concat(oof, ignore_index=True)
    pooled = []
    for name, cols in candidates.items():
        table = predictions.loc[predictions['model'] == name]
        require(len(table) == len(ids) and table['filename'].is_unique and
                set(table['filename']) == set(ids), f'{name}: OOF coverage failed.')
        pooled.append({'model': name, **metrics(table['target'], table['score'].to_numpy())})
        ranked = pd.DataFrame({'feature': cols,
            'mean_absolute_shap_logit': importance[name] / len(ids)})
        ranked.sort_values('mean_absolute_shap_logit', ascending=False).to_csv(
            output / f'{name}_shap_importance.csv', index=False)
    predictions.to_csv(output / 'oof_predictions.csv', index=False)
    scores = ['asd_f1', 'macro_f1', 'precision', 'recall', 'balanced_accuracy',
              'roc_auc', 'average_precision', 'mcc']
    summary = pd.DataFrame(rows).groupby('model')[scores].agg(['mean', 'std'])
    summary.columns = ['_'.join(col) for col in summary.columns]
    summary.to_csv(output / 'cv_summary.csv')
    pd.DataFrame(pooled).to_csv(output / 'pooled_oof_metrics.csv', index=False)
    require(all(sha(path) == config['input_hashes'][name] for name, path in paths.items()),
            'Input changed during execution.')
    evidence = {'status': 'PASS', 'recordings_per_model': 1510, 'models': 3,
        'fold_models': 15, 'explained_recording_predictions': len(predictions),
        'participant_overlap': 0, 'oof_coverage': 'PASS', 'shap_reconstruction': 'PASS',
        'saved_inference': 'PASS', 'official_test_evaluated': False,
        'completed_at_utc': datetime.now(timezone.utc).isoformat()}
    (output / 'validation.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print('\nMean fold results:\n', summary.round(4).to_string(), flush=True)
    print('\nPooled OOF results:\n', pd.DataFrame(pooled).round(4).to_string(index=False), flush=True)
    print('\nOOF coverage, SHAP reconstruction and saved inference: PASS', flush=True)
    print('Saved:', output, flush=True)
    print('Official test not evaluated. Existing frozen models were not loaded or changed.')


if __name__ == '__main__':
    main()
