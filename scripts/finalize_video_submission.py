"""Inspect the video extension and update submission documentation on --run."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
from zipfile import ZipFile

AV_MODEL = 'models/baselines/combined_logistic_regression_tuned.joblib'
AV_HASH = '2a2636966be5f977367569a73db72c7dbdbd90f2e05e7240d014ee443de39e3e'
ASD_MODEL = 'results/asd_feat/e6_reproduction/12_freeze_rgb_behavior_model/20261008T192951_063340Z/rgb_behavior_pca32_lr_frozen.joblib'
ASD_HASH = '2c8e9d9c126e9a7a8134eb9792550254de3747847b96aeaca1e396bae57327d4'
REQUIRED = ['README.md', 'docs/FINAL_DEMO_GUIDE.md', 'docs/FINAL_SUBMISSION_MANIFEST.json', 'docs/FINAL_SUBMISSION_CHECKLIST.md', 'docs/FINAL_RESEARCH_REPORT.docx', 'docs/FINAL_RESEARCH_REPORT.pdf', 'docs/FINAL_RESEARCH_REPORT.md', 'docs/NETSOL_AV_ASD_Project_Brief.pdf', 'presentations/FINAL_RESEARCH_PRESENTATION.pptx', 'docs/VIDEO_UPLOAD_FINAL_STATUS.md', 'dashboard/pages/9_Video_Behavior_Analysis.py', 'scripts/video_upload/features.py', 'scripts/video_upload/report_details.py', 'scripts/video_upload/verify_video_features.py', 'scripts/video_upload/check_upload_cases.py', 'scripts/video_upload/verify_explanations.py', AV_MODEL, ASD_MODEL]
EVIDENCE = ['results/video_upload_validation/20261011T194344_005289Z/validation.json', 'results/video_upload_cases/20261011T195412_015353Z/upload_case_validation.json', 'results/video_upload_explanations/20261011T202257_847333Z/validation.json']

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--run', action='store_true', help='Back up and update documentation and the manifest.')
    args = parser.parse_args()
    root = args.project.resolve()
    for name in REQUIRED + EVIDENCE:
        require((root / name).is_file() and (root / name).stat().st_size > 0, f'Missing or empty: {name}')
    for name in REQUIRED:
        if name.endswith('.py'):
            compile((root / name).read_text(encoding='utf-8-sig'), name, 'exec')
    require(sha(root / AV_MODEL) == AV_HASH, 'AV-ASD artifact hash mismatch.')
    require(sha(root / ASD_MODEL) == ASD_HASH, 'ASD-FEAT artifact hash mismatch.')
    a, b, c = [json.loads((root / name).read_text(encoding='utf-8-sig')) for name in EVIDENCE]
    require(a['status'] == 'PASS' and a['model_sha256'] == AV_HASH and len(a['checks']) == 5 and all(x['status'] == 'PASS' for x in a['checks']), 'Extraction evidence failed.')
    require(b['status'] == 'PASS' and len(b['cases']) == 5 and all(x['status'] == 'PASS' for x in b['cases']), 'Upload-case evidence failed.')
    require(c['status'] == 'PASS' and c['recordings'] == 107 and c['label_reconstruction_checks'] == 1070 and math.isfinite(c['maximum_score_difference']) and 0 <= c['maximum_score_difference'] < 1e-8, 'Explanation evidence failed.')
    with ZipFile(root / 'docs/FINAL_RESEARCH_REPORT.docx') as z:
        text = ''.join(ET.fromstring(z.read('word/document.xml')).itertext())
        require('Muhammad Danial, Hamza Rasheed and Muhammad Freeh' in text, 'Corrected report team order missing.')
    with ZipFile(root / 'presentations/FINAL_RESEARCH_PRESENTATION.pptx') as z:
        text = ''.join(ET.fromstring(z.read('ppt/slides/slide1.xml')).itertext())
        require('Muhammad Danial   Hamza Rasheed   Muhammad Freeh' in text, 'Corrected presentation team order missing.')
    old_manifest = json.loads((root / 'docs/FINAL_SUBMISSION_MANIFEST.json').read_text(encoding='utf-8-sig'))
    names = sorted(set([x['path'] for x in old_manifest['files']] + [x for x in REQUIRED if x not in [AV_MODEL, ASD_MODEL]] + EVIDENCE + ['scripts/finalize_video_submission.py']))
    names = [name for name in names if name != 'docs/FINAL_SUBMISSION_MANIFEST.json']
    for name in names:
        require((root / name).is_file(), f'Manifest file missing: {name}')
    print('PASS: required files and video Python syntax')
    print('PASS: both frozen model hashes')
    print('PASS: extraction 5/5, upload cases 5/5, explanation checks 1070/1070')
    print('PASS: corrected report and presentation team order')
    if not args.run:
        print('Inspection only. Add --run to update submission documentation.')
        return
    replacements = {
        'Hamza Rasheed, Muhammad Danial and Muhammad Freeh': 'Muhammad Danial, Hamza Rasheed and Muhammad Freeh',
        'Hamza Rasheed, Muhammad Danial, and Muhammad Freeh': 'Muhammad Danial, Hamza Rasheed, and Muhammad Freeh',
        'Hamza Rasheed | Muhammad Danial | Muhammad Freeh': 'Muhammad Danial | Hamza Rasheed | Muhammad Freeh',
    }
    edits = {}
    for name in ['README.md', 'docs/FINAL_DEMO_GUIDE.md']:
        s = (root / name).read_text(encoding='utf-8-sig')
        for old, new in replacements.items():
            s = s.replace(old, new)
        if name == 'README.md':
            s = s.replace('The report and presentation require an update for the new MP4 extension.', 'The report and presentation include the validated AV-ASD MP4 extension.')
        else:
            s = s.replace('The research report and presentation still require this extension update.', 'The research report and presentation include this extension.')
        edits[name] = s
    name = 'docs/FINAL_SUBMISSION_CHECKLIST.md'
    s = (root / name).read_text(encoding='utf-8-sig')
    s = s.replace('Independent participant validation, calibration and verified raw-media extraction are future research. They are not implemented capabilities of this feature-based demonstration.', 'Independent participant validation, calibration, automated ASD-FEAT raw-media extraction and temporal modeling remain future research. The separate AV-ASD MP4 feature-extraction workflow is implemented and has passed the recorded software checks.')
    if '## AV-ASD MP4 extension completion' not in s:
        s += '\n\n## AV-ASD MP4 extension completion\n\n- MP4 analysis, summary, exact linear explanations and HTML/JSON export implemented.\n- Existing-video reproduction 5/5; upload cases 5/5; explanation checks 1070/1070.\n- Corrected Word/PDF report and 14-slide presentation installed.\n- Team order: Muhammad Danial, Hamza Rasheed and Muhammad Freeh.\n- New-video accuracy and clinical reliability are not established.\n- GitHub review and publishing of this extension remain separate tasks.\n'
    edits[name] = s
    backup = root / 'documentation_backups' / datetime.now(timezone.utc).strftime('final_checks_%Y%m%dT%H%M%S_%fZ')
    for name in [*edits, 'docs/FINAL_SUBMISSION_MANIFEST.json']:
        target = backup / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)
    for name, s in edits.items():
        (root / name).write_text(s, encoding='utf-8')
        print('Updated:', name)
    manifest = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Installed research submission and AV-ASD video extension files',
        'experiments_retrained': False,
        'files': [{'path': name, 'bytes': (root / name).stat().st_size, 'sha256': sha(root / name)} for name in names],
        'frozen_artifacts': [{'path': AV_MODEL, 'sha256': AV_HASH}, {'path': ASD_MODEL, 'sha256': ASD_HASH}],
        'limits': 'File hashes, syntax and recorded evidence only; no fresh inference or clinical validation.'}
    (root / 'docs/FINAL_SUBMISSION_MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    for item in manifest['files']:
        require(sha(root / item['path']) == item['sha256'], 'Manifest mismatch: ' + item['path'])
    print('Updated: docs/FINAL_SUBMISSION_MANIFEST.json | current hashes verified')
    print('Backups:', backup)
    print('No models loaded, fitted or changed. No predictions regenerated.')
    print('Video extension submission checks: PASS')

if __name__ == '__main__':
    main()
