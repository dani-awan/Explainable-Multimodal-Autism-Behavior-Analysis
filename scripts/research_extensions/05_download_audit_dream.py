"""Resume direct DREAM release download; verify publisher SHA-256 and ZIP CRC."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import http.client
import json
from pathlib import Path
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

CATALOGUE = 'https://researchdata.se/en/catalogue/dataset/snd1156-1/1'
DIRECT = 'https://api.researchdata.se/dataset/snd1156-1/1/file/data/DREAM2020_SND1156-001-V1.0.zip'
NAME = 'DREAM2020_SND1156-001-V1.0.zip'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def find_distribution(node):
    if isinstance(node, dict):
        if node.get('contentUrl') == DIRECT:
            yield node
        for value in node.values():
            yield from find_distribution(value)
    elif isinstance(node, list):
        for value in node:
            yield from find_distribution(value)


def publisher_metadata():
    raw = None
    last_error = None
    attempts = [
        (CATALOGUE, 'AV-ASD-Research-Audit/1.0'),
        (CATALOGUE, 'Mozilla/5.0'),
        ('https://researchdata.se/en/catalogue/dataset/snd1156-1', 'Mozilla/5.0'),
        ('https://researchdata.se/sv/catalogue/dataset/snd1156-1/1', 'Mozilla/5.0'),
    ]
    for index, (page_url, agent) in enumerate(attempts, start=1):
        request = urllib.request.Request(page_url, headers={
            'User-Agent': agent, 'Accept': 'text/html',
            'Accept-Language': 'en-US,en;q=0.9'})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read(8 * 1024**2 + 1)
            print('Publisher metadata fetched:', page_url, flush=True)
            break
        except (urllib.error.URLError, http.client.IncompleteRead,
                TimeoutError, OSError) as error:
            last_error = error
            print(f'Catalogue request {index}/4 failed: {error}', flush=True)
            if index < len(attempts):
                time.sleep(3)
    if raw is None:
        raise RuntimeError('Publisher metadata unavailable; download not started. Last error: ' + str(last_error))
    require(len(raw) <= 8 * 1024**2, 'Catalogue page exceeds size limit.')
    page = raw.decode('utf-8')
    blocks = re.findall(r'<script\b[^>]*type=[\"\']application/ld\+json[\"\'][^>]*>(.*?)</script>', page, re.S | re.I)
    matches = []
    for block in blocks:
        matches.extend(find_distribution(json.loads(block)))
    unique = {json.dumps(item, sort_keys=True) for item in matches}
    require(len(unique) == 1, 'Cannot find unique official direct-file distribution.')
    entry = json.loads(next(iter(unique)))
    checksum = str(entry.get('sha256', '')).lower()
    require(re.fullmatch(r'[0-9a-f]{64}', checksum) is not None, 'Publisher SHA-256 missing or invalid.')
    size = int(entry['contentSize'])
    require(size == 3261899695, 'Publisher release size changed; review before downloading.')
    return {'url': DIRECT, 'size': size, 'sha256': checksum,
            'catalogue': CATALOGUE, 'catalogue_html_sha256': hashlib.sha256(raw).hexdigest()}


def download(partial, metadata, identity_path):
    expected = metadata['size']
    offset = partial.stat().st_size if partial.exists() else 0
    require(offset <= expected, 'Partial file exceeds publisher size.')
    identity = {}
    if identity_path.exists():
        identity = json.loads(identity_path.read_text(encoding='utf-8'))
        require(identity['publisher_sha256'] == metadata['sha256'] and identity['url'] == DIRECT,
                'Partial download belongs to a different release.')
    elif offset:
        raise ValueError('Partial download lacks provenance sidecar; inspect it before resuming.')
    last_progress = time.monotonic()
    errors = 0
    while offset < expected:
        headers = {'User-Agent': 'AV-ASD-Research-Audit/2.0', 'Accept-Encoding': 'identity',
                   'Range': f'bytes={offset}-{min(offset + 8 * 1024**2, expected) - 1}'}
        validator = identity.get('etag') or identity.get('last_modified')
        if offset and validator:
            headers['If-Range'] = validator
        try:
            request = urllib.request.Request(DIRECT, headers=headers)
            with urllib.request.urlopen(request, timeout=90) as response:
                status = response.getcode()
                if status == 206:
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                    require(match and int(match[1]) == offset and int(match[3]) == expected,
                            'Unexpected server resume range.')
                    response_end = int(match[2]) + 1
                    require(offset < response_end <= min(offset + 8 * 1024**2, expected), 'Invalid response range end.')
                else:
                    require(status == 200 and offset == 0,
                            'Direct server did not honor resume; partial preserved. Do not append a full response.')
                    response_end = expected
                content_type = response.headers.get('Content-Type', '').lower()
                require('text/html' not in content_type and 'application/json' not in content_type,
                        'Server returned an error page instead of archive bytes.')
                etag = response.headers.get('ETag')
                modified = response.headers.get('Last-Modified')
                if identity.get('etag') and etag:
                    require(identity['etag'] == etag, 'Remote file ETag changed.')
                identity = {'url': DIRECT, 'publisher_sha256': metadata['sha256'],
                            'etag': etag, 'last_modified': modified}
                identity_path.write_text(json.dumps(identity, indent=2), encoding='utf-8')
                print(f'Connection: HTTP {status} | Resume offset: {offset/1024**2:.1f} MiB', flush=True)
                with partial.open('ab' if offset else 'wb') as output:
                    while offset < response_end:
                        try:
                            block = response.read(min(64 * 1024, response_end-offset))
                        except http.client.IncompleteRead as error:
                            if error.partial:
                                require(offset + len(error.partial) <= response_end, 'Incomplete block exceeds response range.')
                                output.write(error.partial)
                                offset += len(error.partial)
                                output.flush()
                            raise
                        if not block:
                            raise ConnectionError('Connection ended before expected response bytes.')
                        output.write(block)
                        offset += len(block)
                        if time.monotonic() - last_progress >= 15:
                            print(f'{100*offset/expected:.1f}% | {offset/1024**2:.1f} / {expected/1024**2:.1f} MiB', flush=True)
                            last_progress = time.monotonic()
                require(offset == response_end, 'Response size mismatch.')
            errors = 0
        except urllib.error.HTTPError as error:
            if error.code not in [408, 429, 500, 502, 503, 504]:
                raise
            errors += 1
            if errors > 6:
                raise
            print(f'Temporary HTTP {error.code}; retry {errors}/6. Partial preserved.', flush=True)
            time.sleep(min(2**errors, 15))
        except (urllib.error.URLError, http.client.IncompleteRead, ConnectionError, TimeoutError, OSError) as error:
            offset = partial.stat().st_size if partial.exists() else 0
            errors += 1
            if errors > 6:
                raise
            print(f'Connection interrupted ({type(error).__name__}); retry {errors}/6 from {offset/1024**2:.1f} MiB.', flush=True)
            time.sleep(min(2**errors, 15))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    root = args.project.resolve()
    folder = root / 'dataset/external/dream'
    print('Official direct file:', DIRECT, flush=True)
    print('Old streamed-bundle partial is separate and will not be appended.', flush=True)
    if not args.run:
        print('Inspection only. Use --run to download and audit.')
        return
    metadata = publisher_metadata()
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / NAME
    partial = folder / (NAME + '.part')
    sidecar = folder / (NAME + '.download_identity.json')
    metadata_path = folder / 'direct_release_metadata.json'
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    remaining = metadata['size'] - (partial.stat().st_size if partial.exists() else 0)
    require(shutil.disk_usage(folder).free >= max(remaining, 0) + 1024**3 or archive.exists(),
            'Insufficient free space for remaining download and reserve.')
    print('Publisher size MiB:', round(metadata['size']/1024**2, 1), flush=True)
    if not archive.exists():
        download(partial, metadata, sidecar)
        print('Checking publisher SHA-256...', flush=True)
        require(partial.stat().st_size == metadata['size'] and sha(partial) == metadata['sha256'],
                'Publisher checksum mismatch. Partial preserved; do not use this file.')
        partial.rename(archive)
    else:
        require(archive.stat().st_size == metadata['size'] and sha(archive) == metadata['sha256'],
                'Existing direct archive does not match publisher checksum.')
    print('Publisher SHA-256: PASS', flush=True)
    rows = []
    last_progress = time.monotonic()
    with zipfile.ZipFile(archive) as bundle:
        for index, member in enumerate(bundle.infolist(), start=1):
            if member.is_dir():
                continue
            with bundle.open(member) as source:
                while source.read(1024**2):
                    if time.monotonic() - last_progress >= 15:
                        print(f'CRC audit: {index}/{len(bundle.infolist())} entries', flush=True)
                        last_progress = time.monotonic()
            rows.append({'member': member.filename, 'size': member.file_size,
                         'compressed_size': member.compress_size, 'crc32': member.CRC})
    require(bool(rows), 'Archive has no regular files.')
    output = root / 'results/research_extensions/dream_inventory' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'archive_inventory.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    report = {'status': 'PASS', 'source': metadata, 'release_doi': '10.5878/17p8-6k13',
        'archive': str(archive.relative_to(root)), 'publisher_checksum_verified': True,
        'zip_crc': 'PASS', 'regular_entries': len(rows),
        'extensions': dict(Counter(Path(row['member']).suffix.lower() for row in rows)),
        'uncompressed_bytes': sum(row['size'] for row in rows),
        'scope': 'Direct release download, publisher checksum and ZIP inventory only',
        'license_observed': 'CC BY-NC-SA 4.0',
        'inspected_at_utc': datetime.now(timezone.utc).isoformat()}
    (output / 'download_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('ZIP CRC: PASS', flush=True)
    print('Regular entries:', len(rows))
    print('Extensions:', report['extensions'])
    print('Uncompressed size GiB:', round(report['uncompressed_bytes']/1024**3, 3))
    print('First archive paths:', [row['member'] for row in rows[:5]])
    print('Saved:', output)
    print('No files extracted or models trained.')


if __name__ == '__main__':
    main()
