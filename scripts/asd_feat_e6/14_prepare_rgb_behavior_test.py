"""Pool selected official test RGB arrays and align frozen behavioral inputs."""
from stage_runtime import stage_main

SPEC = {
    'name': '14_prepare_rgb_behavior_test', 'source_cell': None,
    'fields': {
        'audit_run': 'results/asd_feat/e6_reproduction/13_audit_rgb_behavior_test/20261008T193617_993194Z',
        'frozen_run': 'results/asd_feat/e6_reproduction/12_freeze_rgb_behavior_model/20261008T192951_063340Z',
    },
    'requires': [('audit_run', name) for name in ['test_availability_audit.json', 'official_test_annotations.csv', 'official_test_rgb_manifest.csv']]
    + [('frozen_run', 'frozen_model_config.json'), ('project', 'dataset/external/asd_feat/i3d_rgb.tar.gz'),
       ('project', 'dataset/external/asd_feat/metadata.tar.gz')],
    'packages': ['numpy', 'pandas'], 'writes': True,
}


def run(args):
    from datetime import datetime, timezone
    import hashlib
    import io
    import json
    import tarfile
    import numpy as np
    import pandas as pd

    def sha256(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    audit_path = args.audit_run / 'test_availability_audit.json'
    frozen_path = args.frozen_run / 'frozen_model_config.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    frozen = json.loads(frozen_path.read_text(encoding='utf-8'))
    if sha256(frozen_path) != audit['frozen_config_sha256']:
        raise ValueError('Frozen configuration changed after availability audit.')
    model_path = args.project / frozen['model_artifact']
    if sha256(model_path) != frozen['model_sha256'] or frozen['model_sha256'] != audit['model_sha256']:
        raise ValueError('Frozen model artifact changed.')
    if frozen['branches'] != ['rgb', 'behavior'] or frozen['threshold'] != 0.5:
        raise ValueError('Unexpected frozen candidate.')
    metadata_archive = args.project / 'dataset/external/asd_feat/metadata.tar.gz'
    if sha256(metadata_archive) != audit['metadata_archive_sha256']:
        raise ValueError('Metadata archive changed after the audit.')
    if audit['development_test_participant_overlap'] or audit['development_test_recording_overlap']:
        raise ValueError('Availability audit reports development/test overlap.')
    manifest_path = args.audit_run / 'official_test_rgb_manifest.csv'
    annotation_path = args.audit_run / 'official_test_annotations.csv'
    manifest = pd.read_csv(manifest_path)
    annotations = pd.read_csv(annotation_path)
    for name, table in [('manifest', manifest), ('annotations', annotations)]:
        if table['filename'].isna().any() or table['filename'].duplicated().any():
            raise ValueError(name+': missing or duplicate recording IDs.')
    if len(manifest) != audit['test_recordings'] or set(manifest['filename']) != set(annotations['filename']):
        raise ValueError('Audited test recording coverage differs.')
    if not manifest['status'].eq('available').all() or manifest['rgb_archive_member'].isna().any() or manifest['rgb_archive_member'].duplicated().any():
        raise ValueError('Require unique RGB entries for every audited test recording.')
    ids = sorted(manifest['filename'].tolist())
    manifest = manifest.set_index('filename').loc[ids].reset_index()
    annotations = annotations.set_index('filename').loc[ids].reset_index()
    for column in ['sub_id', 'visit', 'target']:
        if not manifest[column].equals(annotations[column]) or manifest[column].isna().any():
            raise ValueError('Manifest/annotation disagreement: '+column)
    if not annotations['target'].isin([0, 1]).all():
        raise ValueError('Non-binary test targets.')
    normalized = annotations['asd_longitudinal'].astype(str).str.lower().str.replace(r'[\s_-]+', '', regex=True)
    expected_target = normalized.map({'asd': 1, 'nonasd': 0, 'notasd': 0})
    if expected_target.isna().any() or not np.array_equal(expected_target.to_numpy(), annotations['target'].to_numpy()):
        raise ValueError('Current-visit outcome and target disagreement.')
    columns = frozen['input_columns']
    rgb_columns = [c for c in columns if c.startswith('rgb_')]
    behavior_columns = [c for c in columns if not c.startswith('rgb_')]
    if rgb_columns != [f'rgb_{i:04d}_mean' for i in range(1024)] or len(behavior_columns) != 12:
        raise ValueError('Unexpected frozen input feature schema.')
    if not np.isfinite(annotations[behavior_columns].to_numpy(dtype=float)).all():
        raise ValueError('Non-finite human-coded test behavior.')
    if int(annotations['target'].sum()) != audit['positive_test_recordings'] or annotations['sub_id'].nunique() != audit['test_participants']:
        raise ValueError('Test support differs from availability audit.')

    lookup = manifest.set_index('rgb_archive_member')['filename'].to_dict()
    sizes = manifest.set_index('rgb_archive_member')['rgb_size_bytes'].to_dict()
    seen = set()
    features = []
    outcomes = []
    args.output.mkdir(parents=True, exist_ok=False)
    print('Selected test recordings:', len(manifest))
    print('Pooling 1024 RGB features per recording; one array at a time...', flush=True)
    archive_path = args.project / 'dataset/external/asd_feat/i3d_rgb.tar.gz'
    with tarfile.open(archive_path, 'r|gz') as archive:
        for member in archive:
            if not member.isfile() or member.name not in lookup:
                continue
            if member.name in seen:
                raise ValueError('Duplicate selected archive member: '+member.name)
            seen.add(member.name)
            filename = lookup[member.name]
            array = None
            try:
                if member.size != int(sizes[member.name]) or member.size <= 0 or member.size > 64 * 1024 * 1024:
                    raise ValueError('Unexpected array file size.')
                with archive.extractfile(member) as source:
                    raw = source.read(64 * 1024 * 1024 + 1)
                if len(raw) != member.size:
                    raise ValueError('Array payload size differs from its header.')
                array = np.load(io.BytesIO(raw), allow_pickle=False)
                del raw
                if array.ndim != 2 or not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
                    raise ValueError('Expected a real numeric two-dimensional RGB array.')
                axes = [i for i, size in enumerate(array.shape) if size == 1024]
                if len(axes) != 1:
                    raise ValueError('Cannot uniquely identify the 1024-feature axis.')
                rgb = array if axes[0] == 0 else array.T
                if rgb.shape[1] == 0 or not np.isfinite(rgb).all():
                    raise ValueError('Empty or non-finite RGB windows.')
                pooled = np.mean(rgb, axis=1, dtype=np.float64)
                if not np.isfinite(pooled).all():
                    raise ValueError('Non-finite pooled features.')
                features.append(dict(zip(['filename', *rgb_columns], [filename, *pooled.tolist()])))
                outcomes.append({'filename': filename, 'status': 'PASS', 'windows': int(rgb.shape[1]), 'error': ''})
            except Exception as error:
                outcomes.append({'filename': filename, 'status': 'FAIL', 'windows': 0, 'error': f'{type(error).__name__}: {error}'})
                print('FAIL:', filename, '|', error, flush=True)
            finally:
                array = None
                rgb = None
            if len(outcomes) % 25 == 0:
                pd.DataFrame(outcomes).to_csv(args.output / 'test_rgb_extraction_report.csv', index=False)
                print(f'Processed {len(outcomes)}/{len(manifest)}', flush=True)
    for member in set(lookup)-seen:
        outcomes.append({'filename': lookup[member], 'status': 'FAIL', 'windows': 0, 'error': 'Audited RGB member missing from archive'})
    outcomes = pd.DataFrame(outcomes)
    outcomes.to_csv(args.output / 'test_rgb_extraction_report.csv', index=False)
    if len(outcomes) != len(manifest) or not outcomes['status'].eq('PASS').all():
        raise ValueError('Test RGB extraction incomplete. Review the saved report; no complete input table was written.')
    rgb_table = pd.DataFrame(features, columns=['filename', *rgb_columns]).set_index('filename').loc[ids]
    behavior_table = annotations.set_index('filename').loc[ids, behavior_columns]
    X = pd.concat([rgb_table, behavior_table], axis=1)
    if X.columns.tolist() != columns or not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError('Frozen feature order or finite-value check failed.')
    metadata = annotations[['filename', 'sub_id', 'visit', 'target']].copy()
    outputs = {
        'test_rgb_behavior_X.csv': X.reset_index(),
        'test_y.csv': annotations[['filename', 'target']],
        'test_metadata.csv': metadata,
        'test_rgb_features.csv': rgb_table.reset_index(),
    }
    hashes = {}
    for name, table in outputs.items():
        path = args.output / name
        table.to_csv(path, index=False)
        hashes[name] = sha256(path)
    report = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'PASS',
        'recordings': len(X), 'participants': int(metadata['sub_id'].nunique()),
        'positive_recordings': int(metadata['target'].sum()),
        'positive_participants': int(metadata.loc[metadata['target'].eq(1), 'sub_id'].nunique()),
        'features': len(columns), 'input_columns': columns,
        'frozen_config': frozen_path.relative_to(args.project).as_posix(),
        'frozen_config_sha256': sha256(frozen_path), 'model_sha256': frozen['model_sha256'],
        'source_sha256': {p.relative_to(args.project).as_posix(): sha256(p) for p in [audit_path, manifest_path, annotation_path]},
        'output_sha256': hashes,
        'aggregation': 'Mean of each I3D RGB feature across released windows, matching development extraction',
        'scope': 'Official test feature preparation only; no fitting, tuning or predictions',
        'test_history': audit['test_history'],
    }
    (args.output / 'test_input_validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Passed:', len(X))
    print('Failed: 0')
    print('Test feature table:', (len(X), len(columns)+1))
    print('Feature/target alignment and finite-value checks: PASS')
    print('Saved:', args.output)
    print('No model fitting, threshold tuning or test predictions performed.')


if __name__ == '__main__':
    stage_main(__file__, run, SPEC)
