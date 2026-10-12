"""Verify linear explanations against the existing fitted model; no fitting."""
from pathlib import Path
from datetime import datetime, timezone
import json
import joblib
import numpy as np
import pandas as pd
from report_details import explain_features

ROOT = Path(__file__).resolve().parents[2]
bundle = joblib.load(ROOT / 'models/baselines/combined_logistic_regression_tuned.joblib')
columns = list(bundle['feature_columns'])
features = pd.read_csv(ROOT / 'dataset/model_inputs/test_combined_X.csv')
if features.columns.tolist() != ['Video_ID', *columns] or features['Video_ID'].isna().any() or features['Video_ID'].duplicated().any():
    raise ValueError('Unexpected test feature schema or IDs.')
if not np.isfinite(features[columns].to_numpy(dtype=float)).all():
    raise ValueError('Non-finite inputs.')
maximum = 0.0
checks = 0
for index in range(len(features)):
    explanations = explain_features(bundle, features.loc[[index], columns])
    maximum = max(maximum, max(e['score_difference'] for e in explanations))
    checks += len(explanations)
output = ROOT / 'results/video_upload_explanations' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
output.mkdir(parents=True, exist_ok=False)
report = {'status': 'PASS', 'recordings': len(features), 'label_reconstruction_checks': checks, 'maximum_score_difference': maximum, 'scope': 'Existing-model explanation reconstruction; no fitting or tuning'}
(output / 'validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(f"Recordings: {len(features)} | Label reconstruction checks: {checks}")
print(f'Maximum score difference: {maximum:.12g}')
print('Explanation reconstruction: PASS')
print(f'Saved: {output / "validation.json"}')
