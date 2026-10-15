"""Prepare 32 ordered RGB bins from development arrays, without fitting models."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import tarfile


def require(value, message):
    if not value:
        raise ValueError(message)


def pool_bins(windows, bins=32):
    import numpy as np
    require(windows.ndim == 2 and windows.shape[0] >= bins,
            f'Need at least {bins} windows: {windows.shape}')
    chunks = np.array_split(windows, bins, axis=0)
    counts = np.asarray([len(chunk) for chunk in chunks], dtype=int)
    sequence = np.stack([chunk.mean(axis=0, dtype=np.float64) for chunk in chunks])
    require(np.allclose(np.average(sequence, axis=0, weights=counts),
        windows.mean(axis=0, dtype=np.float64), rtol=1e-7, atol=1e-9),
        'Contiguous-bin conservation failed.')
    return sequence.astype(np.float32), counts


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    import numpy as np
    import pandas as pd
    root = args.project.resolve()
    data = root / 'dataset/external/asd_feat'
    inputs = data / 'derived/e6_inputs'
    archive_path = data / 'i3d_rgb.tar.gz'
    manifest_path = data / 'derived/rgb/successful_development_manifest.csv'
    paths = [archive_path, manifest_path, inputs / 'development_metadata.csv',
             inputs / 'development_y.csv', inputs / 'development_combined_X.csv',
             inputs / 'input_schema.json']
    for path in paths:
        require(path.is_file(), f'Missing input: {path}')
    hashes = {str(p.relative_to(root)): sha(p) for p in paths[1:]}
    archive_stat = archive_path.stat()
    schema = json.loads(paths[-1].read_text(encoding='utf-8-sig'))
    rgb_columns = schema['feature_groups']['rgb']
    require(len(rgb_columns) == 1024, 'Unexpected RGB features.')
    require(sha(inputs / 'development_combined_X.csv') ==
        schema['output_feature_sha256']['development_combined_X.csv'], 'Combined feature hash mismatch.')
    metadata = pd.read_csv(inputs / 'development_metadata.csv')
    targets = pd.read_csv(inputs / 'development_y.csv')
    features = pd.read_csv(inputs / 'development_combined_X.csv')
    manifest = pd.read_csv(manifest_path)
    for name, table in [('metadata', metadata), ('targets', targets),
                         ('features', features), ('manifest', manifest)]:
        require(table['filename'].notna().all() and table['filename'].is_unique,
                f'{name}: missing or duplicate filenames.')
    ids = metadata['filename'].tolist()
    require(len(ids) == 1510 and metadata['sub_id'].nunique() == 259, 'Unexpected development cohort.')
    require(set(targets['filename']) == set(ids) and set(features['filename']) == set(ids),
            'Development recording sets differ.')
    require(set(ids).issubset(set(manifest['filename'])), 'Manifest coverage missing.')
    manifest = manifest.set_index('filename').loc[ids].reset_index()
    targets = targets.set_index('filename').loc[ids]
    features = features.set_index('filename').loc[ids]
    require(targets['target'].isin([0, 1]).all() and
            np.array_equal(targets['target'].to_numpy(), metadata['target'].to_numpy()), 'Target disagreement.')
    require(metadata['sub_id'].notna().all() and
            metadata['evaluation_fold'].isin([1, 2, 3, 4, 5]).all() and
            metadata.groupby('sub_id')['evaluation_fold'].nunique().eq(1).all(), 'Participant fold checks failed.')
    for column in ['sub_id', 'evaluation_fold', 'target', 'visit']:
        require(column in manifest.columns and manifest[column].notna().all(), f'Missing manifest field: {column}')
        require(manifest[column].equals(metadata[column]), f'Manifest metadata disagreement: {column}')
    require(manifest['rgb_archive_member'].notna().all() and manifest['rgb_archive_member'].is_unique,
            'Missing or duplicate archive mappings.')
    require(np.isfinite(features[rgb_columns].to_numpy(dtype=float)).all(), 'Non-finite RGB reference.')
    print('Sequence preparation preflight: PASS | 1510 development recordings', flush=True)
    print('32 contiguous bins; original array row order retained.', flush=True)
    if not args.run:
        print('No arrays loaded or new outputs written. Use --run to prepare sequences.')
        return
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = root / 'results/research_extensions/rgb_sequences' / stamp
    output.mkdir(parents=True, exist_ok=False)
    sequence_path = output / 'rgb_sequences.npy'
    sequences = np.lib.format.open_memmap(sequence_path, mode='w+', dtype=np.float32,
        shape=(len(ids), 32, 1024))
    counts = np.zeros((len(ids), 32), dtype=np.int32)
    lookup = {member: index for index, member in enumerate(manifest['rgb_archive_member'])}
    seen, audit = set(), []
    print('Output:', output, flush=True)
    with tarfile.open(archive_path, 'r|gz') as archive:
        for member in archive:
            if member.name not in lookup:
                continue
            require(member.isfile() and member.name not in seen, f'Invalid or repeated member: {member.name}')
            require(0 < member.size <= 64 * 1024**2, 'RGB member exceeds size limit.')
            seen.add(member.name)
            index = lookup[member.name]
            with archive.extractfile(member) as source:
                payload = source.read(64 * 1024**2 + 1)
            require(len(payload) == member.size, 'Archive payload size mismatch.')
            array = np.load(io.BytesIO(payload), allow_pickle=False)
            require(array.ndim == 2 and array.dtype.kind in 'fiu', 'Expected numeric 2D RGB array.')
            axes = [axis for axis, size in enumerate(array.shape) if size == 1024]
            require(len(axes) == 1, f'Ambiguous RGB feature axis: {array.shape}')
            windows = array if axes[0] == 1 else array.T
            require(np.isfinite(windows).all(), 'Non-finite RGB sequence.')
            mean = windows.mean(axis=0, dtype=np.float64)
            expected = features.iloc[index][rgb_columns].to_numpy(dtype=float)
            difference = float(np.max(np.abs(mean - expected)))
            require(np.allclose(mean, expected, rtol=1e-6, atol=1e-7),
                    f'Array does not reproduce prepared RGB features: {ids[index]} ({difference})')
            sequence, bin_counts = pool_bins(windows)
            require(np.isfinite(sequence).all(), 'Non-finite pooled bins.')
            sequences[index] = sequence
            counts[index] = bin_counts
            audit.append({'filename': ids[index], 'original_windows': len(windows),
                          'maximum_pooled_reference_difference': difference, 'status': 'PASS'})
            if len(seen) % 100 == 0:
                sequences.flush()
                print(f'Prepared {len(seen)}/1510', flush=True)
    require(len(seen) == len(ids), 'Selected archive coverage incomplete.')
    sequences.flush()
    del sequences
    np.save(output / 'bin_window_counts.npy', counts, allow_pickle=False)
    metadata.to_csv(output / 'sequence_metadata.csv', index=False)
    pd.DataFrame(audit).to_csv(output / 'array_validation.csv', index=False)
    require(all(sha(root / name) == value for name, value in hashes.items()), 'An input changed.')
    current_stat = archive_path.stat()
    require((archive_stat.st_size, archive_stat.st_mtime_ns) ==
        (current_stat.st_size, current_stat.st_mtime_ns), 'Archive changed during preparation.')
    report = {'status': 'PASS', 'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
        'recordings': len(ids), 'participants': 259, 'positive_recordings': int(metadata['target'].sum()),
        'sequence_shape': [len(ids), 32, 1024], 'dtype': 'float32',
        'pooling': '32 contiguous, non-overlapping bins covering all stored RGB windows',
        'time_interpretation': 'Stored window order preserved; no timestamp or audio synchronization claim.',
        'archive': str(archive_path.relative_to(root)), 'archive_size_bytes': archive_stat.st_size,
        'input_sha256': hashes, 'sequence_sha256': sha(sequence_path),
        'counts_sha256': sha(output / 'bin_window_counts.npy'),
        'metadata_sha256': sha(output / 'sequence_metadata.csv'),
        'script_sha256': sha(Path(__file__)), 'official_test_processed': False,
        'minimum_original_windows': min(row['original_windows'] for row in audit),
        'maximum_reference_difference': max(row['maximum_pooled_reference_difference'] for row in audit)}
    (output / 'sequence_validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Sequence preparation and pooled-feature reproduction: PASS', flush=True)
    print('Shape:', report['sequence_shape'], '| Minimum windows:', report['minimum_original_windows'])
    print('Saved:', output)
    print('No models fitted and no official test arrays processed.')


if __name__ == '__main__':
    main()
