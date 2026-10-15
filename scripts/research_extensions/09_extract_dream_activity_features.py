"""Extract masked body-relative session descriptors for exploratory activity recognition."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

JOINTS = ['head', 'sholder_left', 'sholder_right', 'elbow_left', 'elbow_right',
          'wrist_left', 'wrist_right', 'hand_left', 'hand_right']
COLUMNS = [f'{joint}_{axis}_{stat}' for joint in JOINTS for axis in ['x', 'y', 'z'] for stat in ['mean', 'std']]


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def coordinates(skeleton, joint):
    import numpy as np
    values = skeleton.get(joint)
    require(isinstance(values, dict), 'Missing skeleton joint: ' + joint)
    axes = [values.get(axis) for axis in ['x', 'y', 'z']]
    require(all(isinstance(axis, list) for axis in axes) and len({len(axis) for axis in axes}) == 1,
            'Invalid XYZ arrays: ' + joint)
    require(all(value is None or (isinstance(value, (float, int)) and not isinstance(value, bool) and np.isfinite(value))
                for axis in axes for value in axis), 'Unexpected nonnumeric or infinite coordinate: ' + joint)
    return np.column_stack([np.asarray([np.nan if value is None else value for value in axis], dtype=float) for axis in axes])


def descriptors(data):
    import numpy as np
    skeleton = data['skeleton']
    center = coordinates(skeleton, 'sholder_center')
    require(len(center) > 0, 'No coordinate frames.')
    center_valid = np.isfinite(center).all(axis=1)
    any_valid = np.zeros(len(center), dtype=bool)
    row, coverage = {}, {}
    for joint in JOINTS:
        values = coordinates(skeleton, joint)
        require(len(values) == len(center), 'Joint/centre frame-count disagreement.')
        valid = center_valid & np.isfinite(values).all(axis=1)
        any_valid |= valid
        coverage[joint] = int(valid.sum())
        relative = values[valid] - center[valid]
        require(np.isfinite(relative).all(), 'Non-finite relative coordinates.')
        for axis, name in enumerate(['x', 'y', 'z']):
            row[f'{joint}_{name}_mean'] = float(relative[:, axis].mean()) if len(relative) >= 2 else np.nan
            row[f'{joint}_{name}_std'] = float(relative[:, axis].std(ddof=0)) if len(relative) >= 2 else np.nan
    require(int(any_valid.sum()) >= 32, 'Fewer than 32 frames with centre plus a complete joint XYZ.')
    require(np.isfinite(np.asarray(list(row.values()))).sum() >= 6, 'Insufficient finite descriptors.')
    coverage.update(original_frames=len(center), center_valid_frames=int(center_valid.sum()),
                    frames_with_relative_joint=int(any_valid.sum()))
    return row, coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--cohort-run', type=Path, required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    import numpy as np
    import pandas as pd
    root = args.project.resolve()
    selection_path = args.cohort_run / 'provisional_activity_cohort.csv'
    review_path = args.cohort_run / 'review_evidence.json'
    review = json.loads(review_path.read_text(encoding='utf-8'))
    cohort = pd.read_csv(selection_path, keep_default_na=False)
    require(review['status'] == 'REVIEW_DOCUMENTED' and len(cohort) == review['provisional_activity_cohort']['sessions'], 'Cohort evidence mismatch.')
    require(cohort['member'].is_unique and cohort['member'].notna().all(), 'Duplicate or missing selected members.')
    require(cohort['task_ability'].isin(['TT', 'IM', 'JA']).all(), 'Unknown activity target.')
    archive = root / 'dataset/external/dream/DREAMdataset.zip'
    require(archive.is_file(), 'DREAM archive missing.')
    print(f'Activity-feature preflight: PASS | {len(cohort)} selected sessions | {len(COLUMNS)} descriptors', flush=True)
    if not args.run:
        print('Inspection only. Use --run for feature extraction.')
        return
    print('Verifying source archive identity...', flush=True)
    require(sha(archive) == review['sources']['archive_sha256'], 'DREAM archive identity changed.')
    source_hashes = {'cohort': sha(selection_path), 'review': sha(review_path), 'script': sha(Path(__file__))}
    output = root / 'results/research_extensions/dream_activity_features' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output.mkdir(parents=True, exist_ok=False)
    rows, metadata, coverage_rows, failures = [], [], [], []
    print('Output:', output, flush=True)
    with zipfile.ZipFile(archive) as bundle:
        require(len({entry.filename for entry in bundle.infolist()}) == len(bundle.infolist()), 'Duplicate archive paths.')
        for index, record in enumerate(cohort.to_dict('records'), start=1):
            try:
                member = bundle.getinfo(record['member'])
                require(not member.is_dir() and member.file_size <= 64*1024**2, 'Invalid or oversized selected member.')
                with bundle.open(member) as source:
                    data = json.load(source)
                require(str(data['participant']['id']) == str(record['participant_id']), 'Participant disagreement.')
                require(data['task']['ability'].strip() == record['task_ability'], 'Activity target disagreement.')
                values, coverage = descriptors(data)
                rows.append({'member': record['member'], **values})
                metadata.append({'member': record['member'], 'participant_id': record['participant_id'], 'target_activity': record['task_ability']})
                coverage_rows.append({'member': record['member'], **coverage})
                del data
            except (ValueError, TypeError, KeyError, UnicodeError, zipfile.BadZipFile) as error:
                failures.append({'member': record['member'], 'reason': f'{type(error).__name__}: {error}'})
            if index % 50 == 0 or index == len(cohort):
                print(f'Processed {index}/{len(cohort)} | Passed: {len(rows)} | Excluded: {len(failures)}', flush=True)
    require(bool(rows), 'No usable feature sessions; failure evidence needs review.')
    features = pd.DataFrame(rows, columns=['member', *COLUMNS])
    meta = pd.DataFrame(metadata)
    require(features['member'].tolist() == meta['member'].tolist(), 'Feature/target alignment failed.')
    array = features[COLUMNS].to_numpy(dtype=float)
    require(not np.isinf(array).any(), 'Infinite feature value.')
    require(sha(selection_path) == source_hashes['cohort'] and sha(review_path) == source_hashes['review'], 'Source metadata changed during extraction.')
    features.to_csv(output / 'activity_X.csv', index=False)
    meta.to_csv(output / 'activity_metadata.csv', index=False)
    pd.DataFrame(coverage_rows).to_csv(output / 'coordinate_coverage.csv', index=False)
    pd.DataFrame(failures, columns=['member', 'reason']).to_csv(output / 'excluded_sessions.csv', index=False)
    grouped = meta.groupby('target_activity').agg(sessions=('member', 'size'), participants=('participant_id', 'nunique')).reset_index()
    grouped.to_csv(output / 'activity_support.csv', index=False)
    report = {'status': 'FEATURE_EXTRACTION_COMPLETE_WITH_EXCLUSIONS' if failures else 'FEATURE_EXTRACTION_COMPLETE',
        'selected_sessions': len(cohort), 'feature_sessions': len(features), 'excluded_sessions': len(failures),
        'participants': int(meta['participant_id'].nunique()), 'features': len(COLUMNS),
        'feature_columns': COLUMNS, 'activity_support': grouped.to_dict('records'),
        'missing_feature_values': int(np.isnan(array).sum()),
        'entirely_missing_columns': features[COLUMNS].columns[features[COLUMNS].isna().all()].tolist(),
        'archive_sha256': review['sources']['archive_sha256'], 'source_hashes': source_hashes,
        'feature_sha256': sha(output / 'activity_X.csv'), 'metadata_sha256': sha(output / 'activity_metadata.csv'),
        'scope': 'Exploratory therapy-activity descriptors, not autism diagnosis or severity',
        'method': 'Mean and population SD of 9 joint XYZ positions relative to shoulder centre, only on complete joint-plus-centre frames; at least two observations per descriptor.',
        'eligibility': 'At least 32 frames with any complete relative joint; minimum six finite descriptors.',
        'missing_policy': 'Unavailable descriptors remain NaN; any later imputer must fit on training participants only.',
        'excluded_predictors': ['participant ID', 'therapy condition', 'ADOS', 'age', 'gender', 'task difficulty', 'session duration', 'coordinate coverage'],
        'limitations': ['Relative coordinates reduce global translation but are not body-size or rotation normalized.',
            'Finite coordinates do not establish tracking accuracy; missingness and recording setup may confound activity recognition.',
            'No frozen classifier was evaluated and no model trained.'],
        'completed_at_utc': datetime.now(timezone.utc).isoformat()}
    (output / 'feature_validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('\nActivity support:\n' + grouped.to_string(index=False), flush=True)
    print('Missing descriptors retained:', report['missing_feature_values'])
    print('Entirely missing columns:', report['entirely_missing_columns'])
    print('Feature/metadata alignment: PASS')
    print('Saved:', output)
    print('No sessions extracted to disk, missing values filled or models trained.')


if __name__ == '__main__':
    main()
