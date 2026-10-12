"""Compare fresh video features and inference against saved test artifacts."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from features import extract_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.project.resolve()
    artifact = root / 'models/baselines/combined_logistic_regression_tuned.joblib'
    bundle = joblib.load(artifact)
    columns = list(bundle['feature_columns'])
    labels = list(bundle['target_columns'])
    if len(columns) != 58 or len(labels) != 10:
        raise ValueError('Expected the AV-ASD 58-feature, 10-label baseline.')
    X = pd.read_csv(root / 'dataset/model_inputs/test_combined_X.csv')
    if X.columns.tolist() != ['Video_ID', *columns] or X.Video_ID.duplicated().any():
        raise ValueError('Unexpected input schema or duplicate IDs.')
    saved = pd.read_csv(root / 'results/baselines/combined_lr_test_probabilities.csv').set_index('Video_ID')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = root / 'results/video_upload_validation' / stamp
    output.mkdir(parents=True, exist_ok=False)
    checks = []
    # Use a fixed first-five selection, independent of outcomes and scores.
    for _, row in X.head(5).iterrows():
        clip_id = str(row['Video_ID'])
        try:
            values, details = extract_video(root / 'dataset/clips' / (clip_id + '.mp4'))
            if set(values) != set(columns):
                raise ValueError('Extracted feature names differ from the bundle.')
            actual = np.array([values[c] for c in columns], dtype=float)
            expected = row[columns].to_numpy(dtype=float)
            close = np.isclose(actual, expected, rtol=1e-5, atol=1e-6)
            if not close.all():
                bad = [columns[i] for i in np.flatnonzero(~close)]
                raise ValueError('Feature mismatch: ' + ', '.join(bad))
            frame = pd.DataFrame([values], columns=columns)
            probabilities = np.asarray(bundle['model'].predict_proba(frame), dtype=float)[0]
            old_probabilities = saved.loc[clip_id, labels].to_numpy(dtype=float)
            if not np.allclose(probabilities, old_probabilities, rtol=1e-5, atol=1e-6):
                raise ValueError('Saved probabilities do not reproduce within tolerance.')
            threshold = float(bundle['threshold'])
            if not np.array_equal(probabilities >= threshold, old_probabilities >= threshold):
                raise ValueError('Thresholded predictions differ.')
            checks.append({'Video_ID': clip_id, 'status': 'PASS', **details,
                'max_feature_difference': float(np.max(np.abs(actual-expected))),
                'max_probability_difference': float(np.max(np.abs(probabilities-old_probabilities)))})
            print('PASS:', clip_id, '| 58 features | probabilities and predictions match')
        except Exception as error:
            checks.append({'Video_ID': clip_id, 'status': 'FAIL', 'error': str(error)})
            print('FAIL:', clip_id, '|', error)
    passed = len(checks) == 5 and all(c['status']=='PASS' for c in checks)
    report = {'status': 'PASS' if passed else 'FAIL', 'checks': checks,
        'model_sha256': hashlib.sha256(artifact.read_bytes()).hexdigest(),
        'feature_rtol': 1e-5, 'feature_atol': 1e-6,
        'scope': 'Five existing test clips; extraction and frozen inference only. No training or dataset updates.'}
    (output / 'validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Saved:', output / 'validation.json')
    if not passed:
        raise SystemExit('Validation failed. Review mismatches before enabling uploads.')
    print('Video extraction pilot: PASS (5/5)')


if __name__ == '__main__':
    main()
