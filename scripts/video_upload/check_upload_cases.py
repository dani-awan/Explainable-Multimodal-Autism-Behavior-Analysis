"""Check actual media extraction using synthetic fixtures; no model training."""
from pathlib import Path
from datetime import datetime, timezone
import json
import subprocess
import tempfile
from features import extract_video, FFMPEG

ROOT = Path(__file__).resolve().parents[2]


def generate(path, duration, audio):
    command = [FFMPEG, '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
        '-f', 'lavfi', '-i', f'color=c=blue:s=64x64:r=10:d={duration}']
    if audio:
        command += ['-f', 'lavfi', '-i', f'anullsrc=r=16000:cl=mono',
                    '-t', str(duration), '-c:a', 'aac']
    command += ['-c:v', 'mpeg4', '-threads', '1', str(path)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=90)
    if result.returncode:
        raise RuntimeError(result.stderr)


def main():
    if not FFMPEG:
        raise RuntimeError('FFmpeg not found on PATH.')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = ROOT / 'results/video_upload_cases' / stamp
    output.mkdir(parents=True, exist_ok=False)
    results = []
    cases = [('silent_audio', 2, True, None),
             ('short_clip', 0.6, True, None),
             ('missing_audio', 2, False, 'RuntimeError'),
             ('too_long', 121, True, 'ValueError'),
             ('corrupt_mp4', None, False, 'ValueError')]
    with tempfile.TemporaryDirectory(prefix='fixtures_', dir=output) as folder:
        for name, duration, audio, expected_error in cases:
            path = Path(folder) / (name + '.mp4')
            if duration is None:
                path.write_bytes(b'This is not a video file.')
            else:
                generate(path, duration, audio)
            try:
                values, details = extract_video(path)
                passed = expected_error is None and len(values) == 58 and details['all_zero_audio']
                if name == 'short_clip':
                    passed = passed and details['sampled_frames'] == 2 and details['decoder'] == 'FFmpeg fallback'
                result = {'case': name, 'status': 'PASS' if passed else 'FAIL',
                          'outcome': 'accepted', 'quality': details}
            except Exception as error:
                passed = expected_error == type(error).__name__
                result = {'case': name, 'status': 'PASS' if passed else 'FAIL',
                          'outcome': 'rejected', 'error_type': type(error).__name__, 'error': str(error)}
            results.append(result)
            print(f"{result['status']}: {name} | {result['outcome']}")
    passed = all(r['status'] == 'PASS' for r in results)
    report = {'status': 'PASS' if passed else 'FAIL', 'cases': results,
              'scope': 'Synthetic extraction checks only. No clinical validation, training, or changes to existing datasets.'}
    target = output / 'upload_case_validation.json'
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Saved:', target)
    print('Passed:', sum(r['status']=='PASS' for r in results), '/', len(results))
    if not passed:
        raise SystemExit('Some cases failed. Review the saved report.')


if __name__ == '__main__':
    main()
