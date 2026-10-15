"""Unpack the verified nested DREAM ZIP only, then inspect headers and JSON samples."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
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


def structure(value, depth=0):
    if isinstance(value, dict):
        if depth >= 4:
            return {'type': 'object', 'keys': list(value)}
        return {str(key): structure(item, depth+1) for key, item in value.items()}
    if isinstance(value, list):
        result = {'type': 'array', 'length': len(value)}
        if value:
            result['element_type'] = type(value[0]).__name__
            if isinstance(value[0], list):
                result['first_element_length'] = len(value[0])
            elif isinstance(value[0], dict):
                result['first_element_keys'] = list(value[0])
        return result
    return {'type': type(value).__name__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--download-audit', type=Path, required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    root = args.project.resolve()
    audit = json.loads(args.download_audit.read_text(encoding='utf-8'))
    require(audit['status'] == 'PASS' and audit['publisher_checksum_verified'] is True and
            audit['zip_crc'] == 'PASS', 'Completed publisher/ZIP audit required.')
    archive = (root / audit['archive']).resolve()
    require(archive.is_relative_to(root) and archive.is_file(), 'Invalid archive path.')
    expected = audit['source']['sha256']
    require(archive.stat().st_size == audit['source']['size'], 'Outer archive size changed.')
    print('Completed download evidence: PASS', flush=True)
    if not args.run:
        print('Inspection only. Use --run to copy the nested archive and inspect samples.')
        return
    print('Verifying archive against publisher hash before unpacking...', flush=True)
    require(sha(archive) == expected, 'Outer archive publisher hash mismatch.')
    folder = root / 'dataset/external/dream'
    destination = folder / 'DREAMdataset.zip'
    identity_path = folder / 'DREAMdataset.zip.source.json'
    with zipfile.ZipFile(archive) as outer:
        matches = [entry for entry in outer.infolist() if entry.filename == 'DREAMdataset.zip']
        require(len(matches) == 1 and not matches[0].is_dir(), 'Nested DREAMdataset.zip not unique.')
        entry = matches[0]
        if destination.exists():
            require(identity_path.is_file(), 'Existing nested ZIP has no source identity; not overwritten.')
            identity = json.loads(identity_path.read_text(encoding='utf-8'))
            require(identity['outer_publisher_sha256'] == expected and identity['outer_member_crc32'] == entry.CRC
                    and destination.stat().st_size == entry.file_size and sha(destination) == identity['nested_sha256'],
                    'Existing nested ZIP identity mismatch; not overwritten.')
            print('Existing nested ZIP verification: PASS', flush=True)
        else:
            require(shutil.disk_usage(folder).free >= entry.file_size + 1024**3, 'Insufficient free F-drive space.')
            partial = folder / 'DREAMdataset.zip.unpacking.part'
            copied = 0
            update = time.monotonic()
            digest = hashlib.sha256()
            with outer.open(entry) as source, partial.open('wb') as output:
                for block in iter(lambda: source.read(1024**2), b''):
                    output.write(block)
                    digest.update(block)
                    copied += len(block)
                    if time.monotonic()-update >= 15:
                        print(f'Nested ZIP copy: {100*copied/entry.file_size:.1f}%', flush=True)
                        update = time.monotonic()
            require(copied == entry.file_size and zipfile.is_zipfile(partial), 'Nested ZIP copy invalid.')
            partial.rename(destination)
            identity = {'outer_publisher_sha256': expected, 'outer_member': entry.filename,
                        'outer_member_crc32': entry.CRC, 'nested_sha256': digest.hexdigest(),
                        'nested_bytes': entry.file_size}
            identity_path.write_text(json.dumps(identity, indent=2), encoding='utf-8')
            print('Nested ZIP copy and outer-member CRC: PASS', flush=True)
        with outer.open('data_example.json') as source:
            require(outer.getinfo('data_example.json').file_size <= 32*1024**2, 'Example exceeds limit.')
            example = json.load(source)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = root / 'results/research_extensions/dream_structure' / stamp
    output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(destination) as inner:
        entries = [entry for entry in inner.infolist() if not entry.is_dir()]
        require(len({entry.filename for entry in entries}) == len(entries), 'Duplicate nested paths.')
        inventory = [{'member': entry.filename, 'bytes': entry.file_size,
                      'compressed_bytes': entry.compress_size, 'crc32': entry.CRC} for entry in entries]
        json_entries = [entry for entry in entries if entry.filename.lower().endswith('.json')]
        samples = []
        for entry in sorted(json_entries, key=lambda item: (item.file_size, item.filename)):
            if entry.file_size > 32*1024**2:
                continue
            with inner.open(entry) as source:
                data = json.load(source)
            if not isinstance(data, dict) or not {'participant', 'skeleton'}.issubset(data):
                continue
            samples.append({'member': entry.filename, 'bytes': entry.file_size,
                'top_level_keys': list(data), 'participant_fields': list(data['participant']) if isinstance(data['participant'], dict) else [],
                'frame_rate': data.get('frame_rate'), 'structure': structure(data)})
            if len(samples) >= 3:
                break
    (output / 'nested_archive_inventory.json').write_text(json.dumps(inventory, indent=2), encoding='utf-8')
    (output / 'session_sample_structure.json').write_text(json.dumps(samples, indent=2), encoding='utf-8')
    (output / 'example_structure.json').write_text(json.dumps(structure(example), indent=2), encoding='utf-8')
    report = {'status': 'INVENTORY_PASS', 'source_identity': identity,
        'nested_regular_files': len(entries), 'json_entries': len(json_entries),
        'extensions': dict(Counter(Path(entry.filename).suffix.lower() for entry in entries)),
        'total_uncompressed_bytes': sum(entry.file_size for entry in entries),
        'largest_member_bytes': max((entry.file_size for entry in entries), default=0),
        'session_samples_parsed': len(samples), 'sample_limit_bytes': 32*1024**2,
        'scope': 'Nested archive inventory and at most three small session samples; not a full cohort quality audit',
        'session_files_extracted': False, 'models_trained': False,
        'created_at_utc': datetime.now(timezone.utc).isoformat()}
    (output / 'structure_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Nested regular files:', report['nested_regular_files'])
    print('JSON entries:', report['json_entries'])
    print('Extensions:', report['extensions'])
    print('Uncompressed size GiB:', round(report['total_uncompressed_bytes']/1024**3, 3))
    print('Largest file MiB:', round(report['largest_member_bytes']/1024**2, 2))
    print('Sample sessions parsed:', len(samples))
    for index, sample in enumerate(samples, start=1):
        print(f"Sample {index} keys:", sample['top_level_keys'])
        print('Participant fields:', sample['participant_fields'])
        print('Frame rate:', sample['frame_rate'])
        for key in ['skeleton', 'eye_gaze', 'head_gaze']:
            print(key, json.dumps(sample['structure'].get(key, {})))
    print('Saved:', output)
    print('No session files extracted or models trained. Full participant/quality audit remains.')


if __name__ == '__main__':
    main()
