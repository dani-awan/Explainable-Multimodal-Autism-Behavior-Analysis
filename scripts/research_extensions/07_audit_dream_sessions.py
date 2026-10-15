"""Audit every DREAM session one at a time; no extraction, imputation or training."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import zipfile


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def audit_channel(values):
    import numpy as np
    if not isinstance(values, list):
        return {'length': 0, 'null_values': 0, 'nonnumeric_values': 0,
                'nonfinite_numeric_values': 0, 'finite_values': 0, 'invalid_array': 1}, None
    numbers = np.asarray([float(value) if numeric(value) else np.nan for value in values], dtype=float)
    nulls = sum(value is None for value in values)
    wrong = sum(value is not None and not numeric(value) for value in values)
    finite = np.isfinite(numbers)
    bad_numeric = sum(numeric(value) and not math.isfinite(value) for value in values)
    return {'length': len(values), 'null_values': nulls, 'nonnumeric_values': wrong,
            'nonfinite_numeric_values': bad_numeric, 'finite_values': int(finite.sum()),
            'invalid_array': 0}, finite


def audit_session(data):
    import numpy as np
    require(isinstance(data, dict), 'Session is not a JSON object.')
    participant = data.get('participant')
    require(isinstance(participant, dict), 'Participant metadata is missing.')
    pid = participant.get('id')
    require(isinstance(pid, (int, str)) and not isinstance(pid, bool) and str(pid).strip(), 'Participant ID is missing or invalid.')
    skeleton = data.get('skeleton')
    require(isinstance(skeleton, dict) and skeleton, 'Skeleton dictionary missing.')
    channels = {}
    triplets = []
    for joint, coordinates in skeleton.items():
        if not isinstance(coordinates, dict):
            channels[f'skeleton.{joint}'] = audit_channel(None)[0]
            continue
        masks = []
        for axis in ['x', 'y', 'z']:
            stats, mask = audit_channel(coordinates.get(axis))
            channels[f'skeleton.{joint}.{axis}'] = stats
            masks.append(mask)
        if all(mask is not None for mask in masks) and len({len(mask) for mask in masks}) == 1:
            triplets.append(np.logical_and.reduce(masks))
        if 'confidence' in coordinates:
            channels[f'confidence.{joint}'] = audit_channel(coordinates['confidence'])[0]
    gaze_triplets = []
    for group in ['eye_gaze', 'head_gaze']:
        values = data.get(group)
        masks = []
        for axis in ['rx', 'ry', 'rz']:
            stats, mask = audit_channel(values.get(axis) if isinstance(values, dict) else None)
            channels[f'{group}.{axis}'] = stats
            masks.append(mask)
        if all(mask is not None for mask in masks) and len({len(mask) for mask in masks}) == 1:
            gaze_triplets.append(np.logical_and.reduce(masks))
    core = {name: stats for name, stats in channels.items() if not name.startswith('confidence.')}
    lengths = [stats['length'] for stats in core.values()]
    finite = sum(stats['finite_values'] for stats in core.values())
    nulls = sum(stats['null_values'] for stats in core.values())
    wrong = sum(stats['nonnumeric_values'] for stats in core.values())
    nonfinite = sum(stats['nonfinite_numeric_values'] for stats in core.values())
    invalid = sum(stats['invalid_array'] for stats in core.values())
    flags = []
    if max(lengths, default=0) == 0:
        flags.append('EMPTY_COORDINATE_ARRAYS')
    elif finite == 0:
        flags.append('NO_FINITE_COORDINATES')
    if len(set(lengths)) > 1:
        flags.append('COORDINATE_LENGTH_MISMATCH')
    if invalid:
        flags.append('MISSING_OR_INVALID_ARRAYS')
    if nulls:
        flags.append('NULL_COORDINATES')
    if wrong:
        flags.append('NONNUMERIC_COORDINATES')
    if nonfinite:
        flags.append('NONFINITE_NUMERIC_COORDINATES')
    frame_rate = data.get('frame_rate')
    valid_rate = numeric(frame_rate) and math.isfinite(frame_rate) and frame_rate > 0
    if not valid_rate:
        flags.append('INVALID_FRAME_RATE')
    aligned_joints = len(set(lengths)) == 1 and not invalid and max(lengths, default=0) > 0
    complete_joint_frames = 0
    if aligned_joints and triplets:
        complete_joint_frames = int(np.logical_or.reduce(triplets).sum())
    if max(lengths, default=0) == 0:
        status = 'EMPTY'
    elif finite == 0:
        status = 'NO_FINITE_COORDINATES'
    elif invalid or len(set(lengths)) > 1 or wrong or nonfinite or not valid_rate:
        status = 'STRUCTURAL_OR_NUMERIC_FINDINGS'
    elif nulls:
        status = 'PARTIAL_MISSING'
    else:
        status = 'COMPLETE_NUMERIC'
    time_value = data.get('time')
    row = {'participant_id': str(pid), 'condition': str(data.get('condition', 'MISSING')),
           'task': str(data.get('task', 'MISSING')), 'quality_status': status,
           'flags': ';'.join(flags), 'maximum_coordinate_length': max(lengths, default=0),
           'finite_coordinate_values': finite, 'null_coordinate_values': nulls,
           'nonnumeric_coordinate_values': wrong, 'nonfinite_numeric_values': nonfinite,
           'invalid_arrays': invalid, 'coordinate_lengths_aligned': aligned_joints,
           'frames_with_any_complete_joint_xyz': complete_joint_frames,
           'finite_joint_xyz_triplets': sum(int(mask.sum()) for mask in triplets),
           'finite_gaze_triplets': sum(int(mask.sum()) for mask in gaze_triplets),
           'frame_rate': float(frame_rate) if valid_rate else '',
           'time_structure': type(time_value).__name__,
           'time_list_length': len(time_value) if isinstance(time_value, list) else '',
           'session_id': str(data.get('$id', ''))}
    return row, channels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--structure-run', type=Path, required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    import numpy as np
    root = args.project.resolve()
    evidence_path = args.structure_run / 'structure_audit.json'
    evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
    require(evidence['status'] == 'INVENTORY_PASS', 'Completed structure inventory required.')
    archive = root / 'dataset/external/dream/DREAMdataset.zip'
    require(archive.is_file(), 'Nested archive missing.')
    print('DREAM full-session audit preflight: PASS', flush=True)
    if not args.run:
        print('Inspection only. Use --run to audit all JSON sessions.')
        return
    print('Checking nested archive identity...', flush=True)
    require(sha(archive) == evidence['source_identity']['nested_sha256'], 'Nested archive changed.')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = root / 'results/research_extensions/dream_quality' / stamp
    output.mkdir(parents=True, exist_ok=False)
    rows, failures = [], []
    totals = defaultdict(Counter)
    print('Output:', output, flush=True)
    with zipfile.ZipFile(archive) as bundle:
        entries = [entry for entry in bundle.infolist() if not entry.is_dir() and entry.filename.lower().endswith('.json')]
        require(len(entries) == evidence['json_entries'], 'JSON inventory changed.')
        require(len({entry.filename for entry in entries}) == len(entries), 'Duplicate archive paths.')
        for index, entry in enumerate(entries, start=1):
            try:
                require(entry.file_size <= 64*1024**2, 'Session exceeds 64 MiB parser limit.')
                with bundle.open(entry) as source:
                    data = json.load(source)
                row, channels = audit_session(data)
                row['member'] = entry.filename
                rows.append(row)
                for name, stats in channels.items():
                    totals[name].update(stats)
                del data
            except (ValueError, TypeError, KeyError, UnicodeError, zipfile.BadZipFile) as error:
                failures.append({'member': entry.filename, 'error': f'{type(error).__name__}: {error}'})
            if index % 50 == 0 or index == len(entries):
                print(f'Audited {index}/{len(entries)} | Parsed: {len(rows)} | Failed: {len(failures)}', flush=True)
    def write_csv(name, values):
        if values:
            with (output / name).open('w', newline='', encoding='utf-8') as source:
                writer = csv.DictWriter(source, fieldnames=list(values[0]))
                writer.writeheader(); writer.writerows(values)
        else:
            (output / name).write_text('', encoding='utf-8')
    write_csv('session_quality.csv', rows)
    write_csv('failed_sessions.csv', failures)
    write_csv('channel_quality.csv', [{'channel': name, **dict(stats)} for name, stats in sorted(totals.items())])
    participants = Counter(row['participant_id'] for row in rows)
    write_csv('participant_coverage.csv', [{'participant_id': key, 'sessions': value} for key, value in sorted(participants.items())])
    statuses = Counter(row['quality_status'] for row in rows)
    flags = Counter(flag for row in rows for flag in row['flags'].split(';') if flag)
    lengths = [row['maximum_coordinate_length'] for row in rows]
    session_ids = [row['session_id'] for row in rows if row['session_id']]
    duplicate_ids = {key: count for key, count in Counter(session_ids).items() if count > 1}
    (output / 'duplicate_session_ids.json').write_text(json.dumps(duplicate_ids, indent=2), encoding='utf-8')
    report = {'status': 'AUDIT_COMPLETE_WITH_FINDINGS' if failures or flags or duplicate_ids else 'AUDIT_COMPLETE',
        'archive_json_entries': len(entries), 'parsed_sessions': len(rows), 'failed_sessions': len(failures),
        'participants_in_parsed_sessions': len(participants), 'quality_status_counts': dict(statuses),
        'flag_counts': dict(flags), 'condition_counts': dict(Counter(row['condition'] for row in rows)),
        'task_counts': dict(Counter(row['task'] for row in rows)),
        'time_structure_counts': dict(Counter(row['time_structure'] for row in rows)),
        'sessions_with_finite_coordinates': sum(row['finite_coordinate_values'] > 0 for row in rows),
        'sessions_with_any_complete_joint_xyz': sum(row['finite_joint_xyz_triplets'] > 0 for row in rows),
        'duplicate_session_id_values': len(duplicate_ids),
        'coordinate_length_min': min(lengths, default=0), 'coordinate_length_max': max(lengths, default=0),
        'coordinate_length_median': float(np.median(lengths)) if lengths else None,
        'coordinate_null_values': sum(row['null_coordinate_values'] for row in rows),
        'coordinate_finite_values': sum(row['finite_coordinate_values'] for row in rows),
        'archive_sha256': evidence['source_identity']['nested_sha256'],
        'structure_evidence_sha256': sha(evidence_path), 'script_sha256': sha(Path(__file__)),
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'All-session structural and numeric coverage audit; no cleaning or modeling',
        'limitations': ['Missing coordinates were counted, not imputed as zero.',
            'Confidence channels were audited separately; finite XYZ does not prove tracking accuracy.',
            'Time structure recorded; timestamp units/order and frame alignment not yet validated.',
            'Therapy condition is not an autism-negative class. No diagnostic performance estimated.']}
    (output / 'quality_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('\nDREAM audit summary:', flush=True)
    for key in ['archive_json_entries', 'parsed_sessions', 'failed_sessions', 'participants_in_parsed_sessions',
                'quality_status_counts', 'condition_counts', 'task_counts', 'time_structure_counts',
                'sessions_with_finite_coordinates', 'sessions_with_any_complete_joint_xyz', 'duplicate_session_id_values']:
        print(f'{key}: {report[key]}', flush=True)
    print('Flag counts:', dict(flags), flush=True)
    print('Saved:', output, flush=True)
    print('Audit complete. Findings require review; no files extracted, values filled or models trained.')


if __name__ == '__main__':
    main()
