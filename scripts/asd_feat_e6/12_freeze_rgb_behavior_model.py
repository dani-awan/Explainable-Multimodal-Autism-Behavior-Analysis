"""Fit and freeze the selected RGB plus human-coded behavior research candidate."""
from stage_runtime import stage_main

SPEC = {
    'name': '12_freeze_rgb_behavior_model', 'source_cell': None,
    'fields': {
        'input_dir': 'dataset/external/asd_feat/derived/e6_inputs',
        'base_run': 'results/asd_feat/e6_reproduction/20261008_115710_8871053/grouped_cv',
        'ablation_run': 'results/asd_feat/e6_reproduction/06_branch_ablation/20261008T190006_874650Z',
    },
    'requires': [('input_dir', name) for name in [
        'development_combined_X.csv', 'development_metadata.csv',
        'development_y.csv', 'input_schema.json',
    ]] + [('base_run', name) for name in ['experiment_config.json', 'cv_summary.csv']]
    + [('ablation_run', name) for name in [
        'experiment_config.json', 'fold_results.csv', 'full_model_reproduction.csv',
        'without_audio_fold1.joblib',
    ]],
    'packages': ['numpy', 'pandas', 'sklearn', 'joblib', 'threadpoolctl'],
    'writes': True,
}


def run(args):
    from datetime import datetime, timezone
    import hashlib
    import json
    import warnings
    import joblib
    import numpy as np
    import pandas as pd
    import sklearn
    from sklearn.base import clone
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits

    def sha256(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    config_path = args.base_run / 'experiment_config.json'
    ablation_config_path = args.ablation_run / 'experiment_config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    ablation_config = json.loads(ablation_config_path.read_text(encoding='utf-8'))
    if config['test_evaluated'] or ablation_config['test_evaluated']:
        raise ValueError('Expected development-only source experiments.')
    if config != ablation_config['base_config']:
        raise ValueError('Ablation does not belong to the selected base run.')
    if float(config['threshold']) != 0.5:
        raise ValueError('Expected the fixed development threshold of 0.5.')
    for name, expected in config['input_sha256'].items():
        if sha256(args.input_dir / name) != expected:
            raise ValueError('Input hash mismatch: '+name)

    reproduction = pd.read_csv(args.ablation_run / 'full_model_reproduction.csv')
    if len(reproduction) != 5 or not reproduction['status'].eq('PASS').all():
        raise ValueError('Expected successful full-model reproduction for all five folds.')
    base_summary = pd.read_csv(args.base_run / 'cv_summary.csv')
    ablation_folds = pd.read_csv(args.ablation_run / 'fold_results.csv')
    if ablation_folds.duplicated(['fold', 'experiment']).any():
        raise ValueError('Duplicate ablation evaluation rows.')
    candidates = base_summary[['model', 'asd_f1_mean', 'macro_f1_mean']].copy()
    reduced = ablation_folds.groupby('experiment')[['asd_f1', 'macro_f1']].mean().reset_index()
    reduced = reduced.rename(columns={'experiment': 'model', 'asd_f1': 'asd_f1_mean', 'macro_f1': 'macro_f1_mean'})
    candidates = pd.concat([candidates, reduced], ignore_index=True)
    if candidates['model'].duplicated().any() or not np.isfinite(candidates[['asd_f1_mean', 'macro_f1_mean']].to_numpy()).all():
        raise ValueError('Invalid candidate comparison.')
    selected_folds = ablation_folds[ablation_folds['experiment'].eq('without_audio')]
    if sorted(selected_folds['fold'].tolist()) != [1, 2, 3, 4, 5]:
        raise ValueError('Selected candidate requires all five evaluation folds.')
    selected_score = float(selected_folds['asd_f1'].mean())
    if selected_score < float(candidates['asd_f1_mean'].max()) - 1e-12:
        raise ValueError('RGB plus behavior is not the highest mean ASD F1 candidate in these sources. Review selection.')
    candidates['selected'] = candidates['model'].eq('without_audio')
    candidates = candidates.sort_values(['asd_f1_mean', 'model'], ascending=[False, True])

    features = pd.read_csv(args.input_dir / 'development_combined_X.csv')
    metadata = pd.read_csv(args.input_dir / 'development_metadata.csv')
    targets = pd.read_csv(args.input_dir / 'development_y.csv')
    for table in [features, metadata, targets]:
        if table['filename'].isna().any() or table['filename'].duplicated().any():
            raise ValueError('Missing or duplicate recording identifiers.')
        if table['filename'].tolist() != features['filename'].tolist():
            raise ValueError('Recording order differs between model input tables.')
    y = targets['target'].to_numpy()
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError('Expected both binary target classes.')
    if not np.array_equal(y, metadata['target'].to_numpy()):
        raise ValueError('Target tables disagree.')
    if metadata[['sub_id', 'evaluation_fold']].isna().any().any():
        raise ValueError('Missing participant or fold metadata.')
    if not metadata['evaluation_fold'].isin([1, 2, 3, 4, 5]).all():
        raise ValueError('Unexpected development folds.')
    if metadata.groupby('sub_id')['evaluation_fold'].nunique().gt(1).any():
        raise ValueError('Participant appears in multiple evaluation folds.')
    if len(features) != int(config['recordings']) or metadata['sub_id'].nunique() != int(config['participants']):
        raise ValueError('Source experiment support differs from current inputs.')

    model_source = args.ablation_run / 'without_audio_fold1.joblib'
    # Load only the project's trusted bundle and remove its fitted state.
    source_bundle = joblib.load(model_source)
    if set(source_bundle['retained_branches']) != {'rgb', 'behavior'} or float(source_bundle['threshold']) != 0.5:
        raise ValueError('Unexpected selected model bundle.')
    model = clone(source_bundle['model'])
    transformers = model.named_steps['branches'].transformers
    if [name for name, _, _ in transformers] != ['rgb', 'behavior']:
        raise ValueError('Expected RGB followed by behavior, with no audio transformer.')
    groups = {name: config['feature_groups'][name] for name in ['rgb', 'behavior']}
    for name, _, columns in transformers:
        if list(columns) != groups[name]:
            raise ValueError('Transformer feature order differs: '+name)
    columns = groups['rgb'] + groups['behavior']
    if len(groups['rgb']) != 1024 or len(groups['behavior']) != 12:
        raise ValueError('Expected 1024 RGB and 12 human-coded behavioral inputs.')
    X = features[columns]
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError('Non-finite selected features.')
    print('Development recordings:', len(X))
    print('Participants:', metadata['sub_id'].nunique())
    print('Selected candidate: RGB + human-coded behavior')
    print('Mean development-fold ASD F1:', f'{selected_score:.4f}')
    print('Required input features:', len(columns))
    print('Fitting on development data only...', flush=True)
    with threadpool_limits(limits=2), warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(X, y.astype(int))
    if list(model.feature_names_in_) != columns or list(model.classes_) != [0, 1]:
        raise ValueError('Fitted schema or class order differs.')
    preview = X.iloc[:min(5, len(X))]
    original_probabilities = model.predict_proba(preview)
    if not np.isfinite(original_probabilities).all() or not ((original_probabilities >= 0) & (original_probabilities <= 1)).all():
        raise ValueError('Invalid fitted-model inference.')

    args.output.mkdir(parents=True, exist_ok=False)
    model_path = args.output / 'rgb_behavior_pca32_lr_frozen.joblib'
    joblib.dump({
        'model': model, 'input_columns': columns, 'feature_groups': groups,
        'threshold': 0.5, 'target': config['target'],
        'selected_candidate': 'without_audio',
        'scope': 'Full-development research fit; no official test evaluation',
        'requires_human_coded_behavior': True,
    }, model_path)
    restored = joblib.load(model_path)
    restored_probabilities = restored['model'].predict_proba(preview)
    if not np.array_equal(original_probabilities, restored_probabilities):
        raise ValueError('Saved-model round-trip inference differs.')
    sources = [config_path, ablation_config_path, model_source,
               args.base_run / 'cv_summary.csv', args.ablation_run / 'fold_results.csv']
    frozen = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'model_artifact': model_path.relative_to(args.project).as_posix(),
        'model_sha256': sha256(model_path), 'candidate': 'without_audio',
        'branches': ['rgb', 'behavior'], 'input_columns': columns,
        'input_feature_count': len(columns), 'threshold': 0.5,
        'development_recordings': len(X), 'development_participants': int(metadata['sub_id'].nunique()),
        'positive_recordings': int(y.sum()), 'mean_fold_asd_f1': selected_score,
        'selection_basis': 'Highest observed mean ASD F1 among exploratory development CV and ablation candidates',
        'selection_timing': 'After exploratory development comparisons; not nested model-selection validation',
        'input_sha256': config['input_sha256'],
        'source_sha256': {p.relative_to(args.project).as_posix(): sha256(p) for p in sources},
        'sklearn_version': sklearn.__version__, 'round_trip_inference': 'PASS',
        'official_e6_test_evaluated': False,
        'limitations': [
            'Development scores used for selection are not independent estimates for the selected winner.',
            'The E1 official test was previously evaluated and contained positive recordings from one participant.',
            'Human-coded behavioral inputs are required; fully automated inference is not established.',
            'Current-visit research target; no clinical diagnosis, disease-stage or treatment recommendation validation.',
        ],
    }
    candidates.to_csv(args.output / 'development_candidate_comparison.csv', index=False)
    (args.output / 'frozen_model_config.json').write_text(json.dumps(frozen, indent=2), encoding='utf-8')
    print('Saved-model round-trip inference: PASS')
    print('Saved:', model_path)
    print('Saved:', args.output / 'frozen_model_config.json')
    print('Research model frozen. No official test evaluated.')


if __name__ == '__main__':
    stage_main(__file__, run, SPEC)
