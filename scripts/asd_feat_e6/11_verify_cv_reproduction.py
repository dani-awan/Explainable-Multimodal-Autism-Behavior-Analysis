"""Compare a fresh E6 CV run with an existing reference; never fit a model."""
from stage_runtime import stage_main

SPEC = {
    'name': '11_verify_cv_reproduction', 'source_cell': None,
    'fields': {
        'base_run': 'results/asd_feat/e6_grouped_cv/20261007T202805_649935Z',
        'candidate_run': 'results/asd_feat/e6_reproduction/candidate_cv',
    },
    'requires': [(field, name) for field in ['base_run', 'candidate_run']
                 for name in ['experiment_config.json', 'oof_predictions.csv', 'fold_results.csv']],
    'packages': ['numpy', 'pandas'], 'writes': True,
}


def run(args):
    import hashlib
    import json
    import numpy as np
    import pandas as pd

    reference_config = json.loads((args.base_run / 'experiment_config.json').read_text(encoding='utf-8'))
    candidate_config = json.loads((args.candidate_run / 'experiment_config.json').read_text(encoding='utf-8'))
    for key in ['recordings', 'participants', 'threshold', 'pca_components_per_audio_or_rgb_branch',
                'pca_solver', 'random_state', 'logistic_settings', 'feature_groups', 'input_sha256']:
        if reference_config[key] != candidate_config[key]:
            raise ValueError('Configuration or input mismatch: '+key)
    if reference_config['test_evaluated'] or candidate_config['test_evaluated']:
        raise ValueError('Expected development-only E6 runs.')
    source_files = []
    rows = []
    for name, keys in [('oof_predictions.csv', ['filename']), ('fold_results.csv', ['fold', 'model'])]:
        paths = [folder / name for folder in [args.base_run, args.candidate_run]]
        reference, candidate = [pd.read_csv(path) for path in paths]
        source_files += paths
        if reference.columns.tolist() != candidate.columns.tolist():
            raise ValueError('Schema mismatch: '+name)
        if any(table[keys].isna().any().any() or table.duplicated(keys).any() for table in [reference, candidate]):
            raise ValueError('Missing or duplicate comparison keys: '+name)
        reference = reference.set_index(keys).sort_index()
        candidate = candidate.set_index(keys).sort_index()
        if not reference.index.equals(candidate.index):
            raise ValueError('Comparison rows differ: '+name)
        largest = 0.0
        for column in reference:
            if pd.api.types.is_numeric_dtype(reference[column]):
                a, b = reference[column].to_numpy(dtype=float), candidate[column].to_numpy(dtype=float)
                if not np.isfinite(a).all() or not np.isfinite(b).all():
                    raise ValueError('Non-finite comparison values: '+column)
                if column.endswith('_prediction') or column in ['target', 'evaluation_fold', 'visit', 'tp', 'fp', 'fn', 'tn']:
                    matches = np.array_equal(a, b)
                else:
                    matches = np.allclose(a, b, rtol=1e-7, atol=1e-8)
                if not matches:
                    raise ValueError('Reproduction mismatch: '+name+' / '+column)
                largest = max(largest, float(np.max(np.abs(a-b))))
            elif not reference[column].equals(candidate[column]):
                raise ValueError('Metadata mismatch: '+name+' / '+column)
        rows.append({'file': name, 'rows': len(reference), 'status': 'PASS', 'maximum_numeric_difference': largest})
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        'status': 'PASS', 'comparisons': rows,
        'reference_run': args.base_run.relative_to(args.project).as_posix(),
        'candidate_run': args.candidate_run.relative_to(args.project).as_posix(),
        'input_and_configuration_match': True,
        'source_sha256': {p.relative_to(args.project).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        'scope': 'Saved development CV comparison only; no training or official test evaluation',
    }
    (args.output / 'CV_REPRODUCTION_VALIDATION.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False))
    print('CV reproduction: PASS')
    print('Saved:', args.output)


if __name__ == '__main__':
    stage_main(__file__, run, SPEC)
