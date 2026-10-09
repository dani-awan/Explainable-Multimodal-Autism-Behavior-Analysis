"""Evaluate the frozen research candidate at its fixed threshold; no fitting."""
from stage_runtime import stage_main

SPEC = {
    'name': '15_evaluate_frozen_rgb_behavior', 'source_cell': None,
    'fields': {
        'frozen_run': 'results/asd_feat/e6_reproduction/12_freeze_rgb_behavior_model/20261008T192951_063340Z',
        'test_dir': 'results/asd_feat/e6_reproduction/14_prepare_rgb_behavior_test/20261008T194036_697250Z',
        'input_dir': 'dataset/external/asd_feat/derived/e6_inputs',
    },
    'requires': [('frozen_run', 'frozen_model_config.json'),
                 ('frozen_run', 'rgb_behavior_pca32_lr_frozen.joblib'),
                 ('input_dir', 'development_metadata.csv')]
    + [('test_dir', name) for name in ['test_input_validation.json', 'test_rgb_behavior_X.csv', 'test_y.csv', 'test_metadata.csv']],
    'packages': ['numpy', 'pandas', 'sklearn', 'joblib', 'threadpoolctl'], 'writes': True,
}


def run(args):
    from datetime import datetime, timezone
    import hashlib
    import json
    import joblib
    import numpy as np
    import pandas as pd
    import sklearn
    from sklearn.metrics import (f1_score, precision_score, recall_score,
        balanced_accuracy_score, accuracy_score, roc_auc_score,
        average_precision_score, matthews_corrcoef, confusion_matrix)
    from threadpoolctl import threadpool_limits

    def sha256(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    frozen_path = args.frozen_run / 'frozen_model_config.json'
    validation_path = args.test_dir / 'test_input_validation.json'
    frozen = json.loads(frozen_path.read_text(encoding='utf-8'))
    validation = json.loads(validation_path.read_text(encoding='utf-8'))
    model_path = args.project / frozen['model_artifact']
    if frozen['candidate'] != 'without_audio' or frozen['branches'] != ['rgb', 'behavior'] or frozen['threshold'] != 0.5:
        raise ValueError('Unexpected frozen candidate or threshold.')
    if validation['status'] != 'PASS' or sha256(frozen_path) != validation['frozen_config_sha256']:
        raise ValueError('Test validation does not match the frozen configuration.')
    if sha256(model_path) != frozen['model_sha256'] or validation['model_sha256'] != frozen['model_sha256']:
        raise ValueError('Frozen model hash mismatch.')
    for name, expected in validation['output_sha256'].items():
        if sha256(args.test_dir / name) != expected:
            raise ValueError('Prepared test file changed: '+name)
    development_path = args.input_dir / 'development_metadata.csv'
    if sha256(development_path) != frozen['input_sha256'][development_path.name]:
        raise ValueError('Development metadata differs from frozen provenance.')
    X_table = pd.read_csv(args.test_dir / 'test_rgb_behavior_X.csv')
    targets = pd.read_csv(args.test_dir / 'test_y.csv')
    metadata = pd.read_csv(args.test_dir / 'test_metadata.csv')
    development = pd.read_csv(development_path)
    columns = frozen['input_columns']
    if X_table.columns.tolist() != ['filename', *columns] or columns != validation['input_columns']:
        raise ValueError('Test feature order differs from frozen inputs.')
    for name, table in [('features', X_table), ('targets', targets), ('metadata', metadata)]:
        if table['filename'].isna().any() or table['filename'].duplicated().any():
            raise ValueError(name+': missing or duplicate recording IDs.')
        if table['filename'].tolist() != X_table['filename'].tolist():
            raise ValueError('Feature/target/metadata recording order differs.')
    y = targets['target'].to_numpy()
    if not np.isin(y, [0, 1]).all() or set(np.unique(y)) != {0, 1}:
        raise ValueError('Expected both binary target classes for this evaluation.')
    if not np.array_equal(y, metadata['target'].to_numpy()):
        raise ValueError('Test target tables disagree.')
    if metadata[['sub_id', 'visit']].isna().any().any():
        raise ValueError('Missing participant or visit metadata.')
    if set(metadata['sub_id']) & set(development['sub_id']) or set(metadata['filename']) & set(development['filename']):
        raise ValueError('Development/test participant or recording overlap.')
    if len(y) != validation['recordings'] or int(y.sum()) != validation['positive_recordings']:
        raise ValueError('Test support differs from prepared inputs.')
    X = X_table[columns]
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError('Non-finite test features.')
    # Load only the project's trusted frozen artifact; never call fit.
    bundle = joblib.load(model_path)
    if bundle['input_columns'] != columns or bundle['threshold'] != frozen['threshold']:
        raise ValueError('Model bundle and frozen configuration differ.')
    model = bundle['model']
    if model.n_features_in_ != len(columns) or list(model.feature_names_in_) != columns:
        raise ValueError('Fitted model schema differs from frozen inputs.')
    if list(model.classes_) != [0, 1]:
        raise ValueError('Unexpected fitted classifier class order.')
    print('Test recordings:', len(X))
    print('Test participants:', metadata['sub_id'].nunique())
    positive_participants = int(metadata.loc[metadata['target'].eq(1), 'sub_id'].nunique())
    print('Positive test participants:', positive_participants)
    print('Frozen threshold:', frozen['threshold'])
    print('Running frozen-model inference only...', flush=True)
    with threadpool_limits(limits=2):
        probabilities = np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    if probabilities.shape != (len(X),) or not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError('Invalid frozen-model probabilities.')
    threshold = float(frozen['threshold'])
    prior = frozen['positive_recordings'] / frozen['development_recordings']
    reference_probabilities = np.full(len(y), prior, dtype=float)

    def metrics(name, probability):
        prediction = (probability >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
        return {
            'model': name, 'threshold': threshold,
            'macro_f1': f1_score(y, prediction, average='macro', labels=[0, 1], zero_division=0),
            'asd_f1': f1_score(y, prediction, zero_division=0),
            'precision': precision_score(y, prediction, zero_division=0),
            'recall': recall_score(y, prediction, zero_division=0),
            'specificity': float(tn/(tn+fp)),
            'balanced_accuracy': balanced_accuracy_score(y, prediction),
            'accuracy': accuracy_score(y, prediction),
            'roc_auc': roc_auc_score(y, probability),
            'average_precision': average_precision_score(y, probability),
            'mcc': matthews_corrcoef(y, prediction),
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        }

    comparison = pd.DataFrame([metrics('rgb_behavior_pca32_lr', probabilities), metrics('dummy_development_prior', reference_probabilities)])
    prediction = (probabilities >= threshold).astype(int)
    predictions = metadata.copy()
    predictions['rgb_behavior_probability'] = probabilities
    predictions['rgb_behavior_prediction'] = prediction
    predictions['dummy_prior_probability'] = reference_probabilities
    predictions['dummy_prior_prediction'] = (reference_probabilities >= threshold).astype(int)
    predictions['outcome'] = np.select(
        [(y==1)&(prediction==1), (y==0)&(prediction==1), (y==1)&(prediction==0)],
        ['True positive', 'False positive', 'False negative'], default='True negative')
    predictions['scope'] = 'Recording-level research prediction'
    # Write into a new folder without modifying the frozen source artifact.
    if sha256(model_path) != frozen['model_sha256'] or sha256(frozen_path) != validation['frozen_config_sha256']:
        raise ValueError('Frozen model/configuration changed during evaluation.')
    args.output.mkdir(parents=True, exist_ok=False)
    comparison.to_csv(args.output / 'frozen_test_comparison.csv', index=False)
    predictions.to_csv(args.output / 'frozen_test_predictions.csv', index=False)
    evidence = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'PASS', 'test_recordings': len(y), 'test_participants': int(metadata['sub_id'].nunique()),
        'positive_test_recordings': int(y.sum()), 'positive_test_participants': positive_participants,
        'primary_model': 'rgb_behavior_pca32_lr', 'threshold': threshold,
        'frozen_config': frozen_path.relative_to(args.project).as_posix(),
        'frozen_config_sha256': sha256(frozen_path),
        'model_artifact': model_path.relative_to(args.project).as_posix(), 'model_sha256': sha256(model_path),
        'test_input_validation': validation_path.relative_to(args.project).as_posix(),
        'test_input_validation_sha256': sha256(validation_path),
        'test_file_sha256': validation['output_sha256'],
        'dummy_reference': 'Constant positive prevalence from the same 1510-recording development subset',
        'dummy_positive_prior': prior, 'sklearn_version': sklearn.__version__,
        'official_e6_test_evaluated': True, 'retraining': False, 'threshold_tuning': False,
        'metrics': comparison.to_dict(orient='records'),
        'limitations': [
            'Official test had previously been evaluated for E1; this is not an untouched test set.',
            'All positive test recordings come from one participant when positive_test_participants equals one.',
            'Repeated recordings are correlated; recording-level scores do not establish participant-level generalization.',
            'Candidate selection used exploratory development comparisons, without nested selection validation.',
            'Human-coded behavioral inputs are required; clinical diagnosis or disease-stage assessment is not established.',
        ],
    }
    (args.output / 'frozen_test_evaluation.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print('\nFrozen recording-level test comparison:')
    print(comparison.round(4).to_string(index=False))
    print('Input alignment and frozen artifact checks: PASS')
    print('Saved:', args.output)
    print('No models retrained or thresholds tuned. Frozen source files preserved.')
    if positive_participants == 1:
        print('Interpretation limit: positive test recordings represent one participant.')


if __name__ == '__main__':
    stage_main(__file__, run, SPEC)
