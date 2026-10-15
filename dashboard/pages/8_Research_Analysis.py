"""Feature-based research inference with a verified frozen local artifact."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import html
import io
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / 'results/asd_feat/e6_reproduction/12_freeze_rgb_behavior_model/20261008T192951_063340Z'
TEST = ROOT / 'results/asd_feat/e6_reproduction/14_prepare_rgb_behavior_test/20261008T194036_697250Z'


def load_frozen(folder):
    config_path = folder / 'frozen_model_config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    artifact = (ROOT / config['model_artifact']).resolve()
    if not artifact.is_relative_to(ROOT.resolve()) or not artifact.is_file():
        raise ValueError('Frozen model path is outside the project or missing.')
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != config['model_sha256']:
        raise ValueError('Frozen model checksum differs from its configuration.')
    # Load only the trusted local model, never a user-uploaded model bundle.
    bundle = joblib.load(artifact)
    columns = list(bundle['input_columns'])
    groups = bundle['feature_groups']
    if columns != config['input_columns'] or columns != groups['rgb'] + groups['behavior']:
        raise ValueError('Frozen input schema disagreement.')
    if len(groups['rgb']) != 1024 or len(groups['behavior']) != 12:
        raise ValueError('Expected 1024 RGB features and 12 human-coded behavior values.')
    if float(bundle['threshold']) != 0.5 or float(config['threshold']) != 0.5:
        raise ValueError('Unexpected frozen threshold.')
    model = bundle['model']
    if list(model.feature_names_in_) != columns or list(model.classes_) != [0, 1]:
        raise ValueError('Unexpected fitted feature or class order.')
    return config, bundle


def validate_input(table, columns):
    if table.columns.duplicated().any():
        raise ValueError('Duplicate column names.')
    if len(table) != 1:
        raise ValueError('Upload exactly one recording per analysis.')
    if set(table.columns) != set(columns) or len(table.columns) != len(columns):
        missing = sorted(set(columns) - set(table.columns))
        extra = sorted(set(table.columns) - set(columns))
        raise ValueError(f'Schema mismatch. Missing: {missing[:5]}; extra: {extra[:5]}.')
    ordered = table.loc[:, columns].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(ordered.to_numpy(dtype=float)).all():
        raise ValueError('All 1036 values must be finite numeric values.')
    return ordered


def explain_prediction(bundle, inputs):
    model = bundle['model']
    probability = float(model.predict_proba(inputs)[0, 1])
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError('Invalid model output.')
    transformed = np.asarray(model[:-1].transform(inputs), dtype=float)
    classifier = model.named_steps['classifier']
    contributions = transformed[0] * classifier.coef_[0]
    intercept = float(classifier.intercept_[0])
    logit = float(model.decision_function(inputs)[0])
    if not np.isclose(intercept + contributions.sum(), logit, atol=1e-8, rtol=1e-7):
        raise ValueError('Explanation does not reconstruct the fitted score.')
    branch_transformer = model.named_steps['branches']
    rows = []
    for branch in ['rgb', 'behavior']:
        section = branch_transformer.output_indices_[branch]
        rows.append({'Branch': branch, 'Logit contribution': float(contributions[section].sum())})
    return probability, intercept, logit, pd.DataFrame(rows)


def build_report(source, config, probability, intercept, logit, contributions):
    report = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'Feature-based research demonstration; not a clinical assessment',
        'input_source': source,
        'model': 'rgb_behavior_pca32_lr',
        'model_sha256': config['model_sha256'],
        'model_score': probability,
        'model_score_percent': probability * 100,
        'clinical_diagnosis': 'Not established by this research model',
        'severity_or_stage': 'Not assessed',
        'score_interpretation': 'Uncalibrated model output; not a validated autism likelihood',
        'threshold': 0.5,
        'threshold_result': 'Above threshold' if probability >= 0.5 else 'Below threshold',
        'intercept': intercept,
        'decision_logit': logit,
        'branch_logit_contributions': contributions.to_dict(orient='records'),
        'input_features': 1036,
        'behavior_source': 'Human-coded values supplied with the compatible feature row',
        'limits': [
            'Score is an uncalibrated model output, not the probability that a child has autism.',
            'All positive official test recordings represented one participant.',
            'No clinical diagnosis, disease-stage assignment or treatment recommendations are provided.',
            'Raw video, photo and audio extraction are not implemented on this page.',
            'Uploaded feature provenance cannot be verified from numeric values alone.',
            'Explanations are fitted-score contributions, not causes or observed symptoms.',
        ],
    }
    escape = lambda value: html.escape(str(value))
    limits = ''.join(f'<li>{escape(item)}</li>' for item in report['limits'])
    rows = ''.join(f"<tr><td>{escape(row['Branch'])}</td><td>{row['Logit contribution']:.6f}</td></tr>" for row in report['branch_logit_contributions'])
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Research Analysis Report</title><style>
body{{font:16px Arial,sans-serif;max-width:850px;margin:40px auto;padding:0 24px;color:#172b40;line-height:1.5}}
h1,h2{{color:#172b40}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ddd;padding:10px;text-align:left}}
@media print{{body{{margin:0;max-width:none}}}}
</style><h1>Research Analysis Report</h1>
<p>Feature-based research demonstration. Not a clinical assessment.</p>
<p>Created: {escape(report['created_at_utc'])}<br>Input: {escape(source)}</p>
<h2>Analysis result</h2>
<p><strong>Clinical diagnosis:</strong> Not established by this research model.<br>
<strong>Model score:</strong> {probability * 100:.2f}% ({probability:.6f} on a 0 to 1 scale)<br>
<strong>Frozen threshold:</strong> 50% (0.5)<br>
<strong>Research prediction:</strong> {escape(report['threshold_result'])}<br>
<strong>Autism severity or stage:</strong> Not assessed.</p>
<p>The percentage is an uncalibrated model score, not a validated percentage chance of autism. Neither threshold outcome confirms or rules out autism.</p>
<h2>Fitted score explanation</h2><table><tr><th>Branch</th><th>Logit contribution</th></tr>{rows}</table>
<p>Intercept: {intercept:.6f}; reconstructed decision logit: {logit:.6f}.</p>
<h2>Input and validation</h2><p>1036 finite numeric values: 1024 pooled I3D RGB features and 12 human-coded behavior values. Feature order follows the frozen model. Local artifact checksum verified.</p>
<h2>Limitations</h2><ul>{limits}</ul>
<h2>Research conclusion</h2>
<p>According to this frozen model, the supplied features produced a score of <strong>{probability * 100:.2f}%</strong>, which is <strong>{escape(report['threshold_result']).lower()}</strong> at the fixed 50% threshold. This is a research classification result. It does not establish whether a child has autism or how severe it is.</p>
<p>Model SHA256: {escape(config['model_sha256'])}</p></html>'''
    return report, document


st.title('Research Analysis')
st.caption('RGB plus human-coded behavior • Frozen threshold 0.5')
st.warning('Research demonstration only. This page does not determine whether a child has autism, assign a stage or recommend treatment.')
st.info('Use an existing recording or one compatible feature CSV. Raw photo, audio and video analysis will require a separately verified extractor.')
try:
    config, bundle = load_frozen(FROZEN)
except Exception as error:
    st.error(f'Cannot load the verified local model: {error}')
    st.stop()

columns = bundle['input_columns']
mode = st.radio('Input source', ['Existing test recording demonstration', 'Compatible feature CSV'], horizontal=True)
inputs = None
source = ''

if mode == 'Existing test recording demonstration':
    st.caption('These recordings were already evaluated. This is a replay of research inference, not a new test.')
    try:
        validation = json.loads((TEST / 'test_input_validation.json').read_text(encoding='utf-8'))
        expected = validation['output_sha256']['test_rgb_behavior_X.csv']
        if hashlib.sha256((TEST / 'test_rgb_behavior_X.csv').read_bytes()).hexdigest() != expected:
            raise ValueError('Prepared test feature checksum differs.')
        if validation['model_sha256'] != config['model_sha256']:
            raise ValueError('Prepared test inputs belong to a different model.')
        table = pd.read_csv(TEST / 'test_rgb_behavior_X.csv')
        if table['filename'].isna().any() or table['filename'].duplicated().any():
            raise ValueError('Missing or duplicate recording identifiers.')
        if table.columns.tolist() != ['filename', *columns]:
            raise ValueError('Prepared test feature order differs from the frozen model.')
        index = st.selectbox('Recording', range(len(table)), format_func=lambda value: f'Recording {value + 1}')
        inputs = validate_input(table.iloc[[index]].drop(columns='filename'), columns)
        source = f'Existing official test recording {index + 1}; already evaluated'
    except Exception as error:
        st.error(f'Prepared demonstration inputs are unavailable or invalid: {error}')
else:
    st.caption('Features must use the released I3D representation and the same pooling and behavior coding conventions. Matching column names alone does not prove compatibility.')
    schema = {'columns': columns, 'feature_groups': bundle['feature_groups'], 'requirements': 'One row, no identifier column, 1036 finite numeric values. Released I3D RGB features mean-pooled over windows plus the 12 human-coded behavior values.'}
    st.download_button('Download input schema', json.dumps(schema, indent=2), 'research_input_schema.json', 'application/json')
    uploaded = st.file_uploader('One-recording feature CSV', type=['csv'])
    verified = st.checkbox('I have checked the feature extraction and behavior coding against the training source.')
    if uploaded and verified:
        try:
            payload = uploaded.getvalue()
            if len(payload) > 2 * 1024 * 1024:
                raise ValueError('Feature CSV exceeds the 2 MiB limit.')
            # Check the original header before pandas can rename duplicate columns.
            import csv
            header = next(csv.reader(io.StringIO(payload.decode('utf-8-sig'))))
            if len(header) != len(set(header)):
                raise ValueError('Duplicate CSV header names.')
            inputs = validate_input(pd.read_csv(io.BytesIO(payload)), columns)
            source = 'User-supplied compatible feature CSV; provenance declared by uploader'
        except Exception as error:
            st.error(f'Input rejected: {error}')

# Never reuse a result after the input source or uploaded file changes.
fingerprint = hashlib.sha256(inputs.to_csv(index=False).encode()).hexdigest() if inputs is not None else None
if st.session_state.get('research_input_fingerprint') != fingerprint:
    st.session_state.pop('research_analysis_result', None)
    st.session_state['research_input_fingerprint'] = fingerprint
if inputs is not None and st.button('Analyze research features', type='primary'):
    try:
        probability, intercept, logit, contributions = explain_prediction(bundle, inputs)
        report, document = build_report(source, config, probability, intercept, logit, contributions)
        st.session_state['research_analysis_result'] = (report, document, contributions)
    except Exception as error:
        st.error(f'Analysis stopped: {error}')

if 'research_analysis_result' in st.session_state:
    report, document, contributions = st.session_state['research_analysis_result']
    probability = report['model_score']
    st.metric('Model score — uncalibrated', f'{probability * 100:.2f}%')
    st.caption(f'Raw score: {probability:.6f}; frozen threshold: 50% (0.5). The percentage is not a validated autism likelihood.')
    st.write('Threshold result:', report['threshold_result'])
    st.caption('Neither threshold outcome is a clinical conclusion.')
    st.subheader('Contributions to the fitted decision score')
    st.dataframe(contributions, hide_index=True)
    st.caption('Positive contributions raise the logit; negative contributions lower it. These are not observed symptoms or causal findings.')
    st.download_button('Download readable report', document, 'Research_Analysis_Report.html', 'text/html')
    st.download_button('Download structured report', json.dumps(report, indent=2), 'Research_Analysis_Report.json', 'application/json')
    st.caption('Use the direct PDF download below. Reports contain no personal name or recording identifier. Inputs are not written to project files by this page.')

# REPORT_WORKSPACE_INTEGRATION
# Formatting existing results only; no additional model inference.
import sys as _workspace_sys
_workspace_sys.path.insert(0, str(ROOT / 'scripts/report_workspace'))
from theme import apply_theme as _workspace_theme
from downloads import pdf_download as _workspace_download
_workspace_theme(st)
_workspace_report = (st.session_state.get('research_analysis_result') or [None])[0]
if _workspace_report:
    _workspace_download(st, _workspace_report, 'professional_pdf_features')
# END_REPORT_WORKSPACE_INTEGRATION
