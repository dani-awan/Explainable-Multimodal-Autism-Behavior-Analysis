"""Audit official test metadata and released RGB coverage after model freeze."""
from stage_runtime import stage_main

SPEC = {
    'name': '13_audit_rgb_behavior_test', 'source_cell': None,
    'fields': {
        'frozen_run': 'results/asd_feat/e6_reproduction/12_freeze_rgb_behavior_model/20261008T192951_063340Z',
        'input_dir': 'dataset/external/asd_feat/derived/e6_inputs',
    },
    'requires': [('frozen_run', 'frozen_model_config.json'),
                 ('input_dir', 'development_metadata.csv'),
                 ('project', 'dataset/external/asd_feat/metadata.tar.gz'),
                 ('project', 'dataset/external/asd_feat/i3d_rgb.tar.gz')],
    'packages': ['numpy', 'pandas'], 'writes': True,
}


def run(args):
    from datetime import datetime, timezone
    from pathlib import PurePosixPath
    import hashlib
    import json
    import tarfile
    import numpy as np
    import pandas as pd

    config_path = args.frozen_run / 'frozen_model_config.json'
    frozen = json.loads(config_path.read_text(encoding='utf-8'))
    if frozen['candidate'] != 'without_audio' or frozen['branches'] != ['rgb', 'behavior'] or frozen['threshold'] != 0.5:
        raise ValueError('Unexpected frozen research candidate.')
    if frozen['official_e6_test_evaluated']:
        raise ValueError('Frozen configuration already records E6 test evaluation.')
    artifact = args.project / frozen['model_artifact']
    if not artifact.is_file() or hashlib.sha256(artifact.read_bytes()).hexdigest() != frozen['model_sha256']:
        raise ValueError('Frozen model artifact is missing or changed.')
    data = args.project / 'dataset/external/asd_feat'
    metadata_archive = data / 'metadata.tar.gz'
    with tarfile.open(metadata_archive, 'r:gz') as archive:
        matches = [m for m in archive.getmembers() if m.isfile()
                   and PurePosixPath(m.name).name == 'hj_test_G_Aggregated.csv']
        if len(matches) != 1:
            available = [m.name for m in archive.getmembers() if m.isfile() and m.name.endswith('.csv')]
            raise ValueError(f'Official test G table not found uniquely. Available tables: {available}')
        with archive.extractfile(matches[0]) as source:
            test = pd.read_csv(source)
        metadata_member = matches[0].name

    behavior_columns = [c for c in frozen['input_columns'] if not c.startswith('rgb_')]
    if len(behavior_columns) != 12:
        raise ValueError('Expected 12 frozen behavioral features.')
    required = ['filename', 'sub_id', 'visit', 'asd_longitudinal', *behavior_columns]
    if not set(required).issubset(test.columns):
        raise ValueError('Missing test annotation columns: '+str(sorted(set(required)-set(test.columns))))
    if test[['filename', 'sub_id', 'visit', 'asd_longitudinal']].isna().any().any() or test['filename'].duplicated().any():
        raise ValueError('Missing test metadata or duplicate recording IDs.')
    if not np.isfinite(test[behavior_columns].to_numpy(dtype=float)).all():
        raise ValueError('Non-finite human-coded test behavior features.')
    labels = test['asd_longitudinal'].astype(str).str.lower().str.replace(r'[\s_-]+', '', regex=True)
    target = labels.map({'asd': 1, 'nonasd': 0, 'notasd': 0})
    if target.isna().any():
        raise ValueError('Unrecognized test outcome annotations.')
    test = test.copy()
    test['target'] = target.astype(int)
    development = pd.read_csv(args.input_dir / 'development_metadata.csv')
    participant_overlap = set(test['sub_id']) & set(development['sub_id'])
    recording_overlap = set(test['filename']) & set(development['filename'])
    if participant_overlap or recording_overlap:
        raise ValueError('Development/test participant or recording overlap.')

    test['recording_stem'] = test['filename'].map(lambda value: PurePosixPath(str(value)).stem)
    if test['recording_stem'].duplicated().any():
        raise ValueError('Test filenames do not produce unique archive lookup stems.')
    lookup = test.set_index('recording_stem')['filename'].to_dict()
    matched = {}
    entries = 0
    print('Official test recordings:', len(test))
    print('Scanning RGB archive headers; no arrays are loaded...', flush=True)
    rgb_archive = data / 'i3d_rgb.tar.gz'
    # Stream headers only; never extract archive members onto disk.
    with tarfile.open(rgb_archive, 'r|gz') as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith('.npy'):
                continue
            entries += 1
            stem = PurePosixPath(member.name).stem
            if stem in lookup:
                if stem in matched:
                    raise ValueError('Multiple RGB members match a test recording: '+stem)
                if member.size <= 0 or member.size > 64 * 1024 * 1024:
                    raise ValueError('Unexpected selected array file size: '+member.name)
                matched[stem] = {'rgb_archive_member': member.name, 'rgb_size_bytes': member.size}
            if entries % 200 == 0:
                print(f'Scanned {entries} array headers | Matched test recordings: {len(matched)}', flush=True)
    manifest = test[['filename', 'recording_stem', 'sub_id', 'visit', 'target']].copy()
    manifest['rgb_archive_member'] = manifest['recording_stem'].map(lambda stem: matched.get(stem, {}).get('rgb_archive_member', ''))
    manifest['rgb_size_bytes'] = manifest['recording_stem'].map(lambda stem: matched.get(stem, {}).get('rgb_size_bytes', 0))
    manifest['status'] = np.where(manifest['rgb_archive_member'].ne(''), 'available', 'missing_rgb')
    available = manifest[manifest['status'].eq('available')]
    positive_participants = int(test.loc[test['target'].eq(1), 'sub_id'].nunique())
    report = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'test_recordings': len(test), 'test_participants': int(test['sub_id'].nunique()),
        'positive_test_recordings': int(test['target'].sum()),
        'positive_test_participants': positive_participants,
        'matched_rgb_recordings': len(available), 'missing_rgb_recordings': len(test)-len(available),
        'matched_positive_recordings': int(available['target'].sum()),
        'matched_positive_participants': int(available.loc[available['target'].eq(1), 'sub_id'].nunique()),
        'development_test_participant_overlap': len(participant_overlap),
        'development_test_recording_overlap': len(recording_overlap),
        'rgb_array_headers_scanned': entries, 'behavior_feature_count': len(behavior_columns),
        'frozen_config': config_path.relative_to(args.project).as_posix(),
        'frozen_config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'model_sha256': frozen['model_sha256'],
        'metadata_archive_member': metadata_member,
        'metadata_archive_sha256': hashlib.sha256(metadata_archive.read_bytes()).hexdigest(),
        'scope': 'Test availability and annotation audit only; no arrays loaded or model predictions generated',
        'test_history': 'Official test was previously evaluated in E1; it is not an untouched test set.',
        'positive_participant_warning': 'One positive participant cannot establish generalization across positive participants.' if positive_participants == 1 else None,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    test.to_csv(args.output / 'official_test_annotations.csv', index=False)
    manifest.to_csv(args.output / 'official_test_rgb_manifest.csv', index=False)
    (args.output / 'test_availability_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Test participants:', report['test_participants'])
    print('Positive test recordings:', report['positive_test_recordings'])
    print('Positive test participants:', positive_participants)
    print('Available RGB recordings:', len(available), '/', len(test))
    print('Missing RGB recordings:', report['missing_rgb_recordings'])
    print('Development/test participant overlap: 0')
    print('Saved:', args.output)
    print('Audit complete. No arrays loaded, models fitted or predictions generated.')


if __name__ == '__main__':
    stage_main(__file__, run, SPEC)
