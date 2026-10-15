"""Local MP4 analysis using the validated AV-ASD feature baseline."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import html
import json
import sys
import tempfile
import threading

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/video_upload'))
from features import extract_video
from report_details import explain_features, summarize_results

st.set_page_config(page_title='Video Behavior Analysis', layout='wide')
st.title('Video Behavior Analysis — AV-ASD')
st.write('Upload a short MP4 to explore ten behavioral-label scores using the existing audio and visual baseline.')
st.info('Research analysis only. These scores do not diagnose autism, measure severity, or represent validated chances of autism. This page uses the AV-ASD baseline, separately from the ASD-FEAT RGB/behavior model.')
st.caption('MP4 with an audio track • maximum 120 seconds, 7200 frames and 200 MiB. A silent audio track is accepted and flagged; an absent audio track is rejected.')

@st.cache_resource
def extraction_lock():
    return threading.Lock()


def load_validated_bundle():
    artifact = ROOT / 'models/baselines/combined_logistic_regression_tuned.joblib'
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    reports = sorted((ROOT / 'results/video_upload_validation').glob('*/validation.json'), reverse=True)
    valid = False
    for path in reports:
        report = json.loads(path.read_text(encoding='utf-8'))
        checks = report.get('checks', [])
        if report.get('status') == 'PASS' and report.get('model_sha256') == digest and len(checks) == 5 and all(c.get('status') == 'PASS' for c in checks):
            valid = True
            break
    if not valid:
        raise ValueError('Run the video extraction verification script successfully for this model first.')
    bundle = joblib.load(artifact)
    if len(bundle['feature_columns']) != 58 or len(bundle['target_columns']) != 10:
        raise ValueError('Unexpected AV-ASD model schema.')
    if not np.isclose(float(bundle['threshold']), 0.3):
        raise ValueError('Expected the frozen AV-ASD threshold of 0.3.')
    return bundle, digest

def clear_previous_report():
    st.session_state.pop('av_asd_video_report', None)

upload = st.file_uploader('Choose MP4 video', type=['mp4'],
    key='av_asd_video_upload', on_change=clear_previous_report)

with st.form('video_analysis'):
    submit = st.form_submit_button('Analyze video')

if submit:
    st.session_state.pop('av_asd_video_report', None)
    if upload is None:
        st.error('Choose a video first.')
    elif upload.size > 200 * 1024**2:
        st.error('The video exceeds the 200 MiB limit.')
    else:
        try:
            with st.spinner('Extracting features and running the frozen baseline...'):
                bundle, model_digest = load_validated_bundle()
                scratch = ROOT / 'results/video_upload_temp'
                scratch.mkdir(parents=True, exist_ok=True)
                # Temporary uploads are removed after success or failure.
                with tempfile.TemporaryDirectory(prefix='upload_', dir=scratch) as folder:
                    video_path = Path(folder) / 'clip.mp4'
                    video_path.write_bytes(upload.getvalue())
                    video_digest = hashlib.sha256(video_path.read_bytes()).hexdigest()
                    with extraction_lock():
                        values, details = extract_video(video_path)
                columns = list(bundle['feature_columns'])
                labels = list(bundle['target_columns'])
                if set(values) != set(columns):
                    raise ValueError('Extracted feature names differ from the saved model.')
                frame = pd.DataFrame([values], columns=columns)
                scores = np.asarray(bundle['model'].predict_proba(frame), dtype=float)
                if scores.shape != (1, 10) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
                    raise ValueError('Invalid model output.')
                threshold = float(bundle['threshold'])
                result = [{'label': label, 'model_score': float(score),
                    'threshold_result': 'Above threshold' if score >= threshold else 'Below threshold'}
                    for label, score in zip(labels, scores[0])]
                explanations = explain_features(bundle, frame)
                summary = summarize_results(result, threshold)
                st.session_state['av_asd_video_report'] = {
                    'generated_at_utc': datetime.now(timezone.utc).isoformat(),
                    'model': 'AV-ASD combined Logistic Regression',
                    'model_sha256': model_digest, 'video_sha256': video_digest,
                    'threshold': threshold, 'features': 58,
                    'quality': details, 'labels': result,
                    'summary': summary, 'explanations': explanations,
                    'limitations': 'Uncalibrated behavioral-label scores; not autism diagnosis, severity, or clinical probability. New-video performance has not been independently validated. Appearance and audio features may reflect background, camera, or recording conditions.'}
        except Exception as error:
            st.error(f'Analysis failed: {error}')

report = st.session_state.get('av_asd_video_report')
if report:
    st.subheader('Recording summary')
    st.write(report.get('summary', summarize_results(report['labels'], report['threshold'])))
    st.subheader('Behavioral-label results')
    st.caption(f"Frozen threshold: {report['threshold']:.2f}. Above threshold is a model output, not confirmation that a behavior occurred.")
    table = pd.DataFrame(report['labels']).sort_values('model_score', ascending=False)
    st.dataframe(table, hide_index=True, width='stretch')
    if report.get('explanations'):
        st.subheader('Why the model assigned these scores')
        st.caption('Contributions explain the fitted model, not the cause of a behavior. Positive values raise the logit; negative values lower it. They are not percentages. The top ten rows omit other contributions and the intercept.')
        for explanation in sorted(report['explanations'], key=lambda e: -e['reconstructed_score']):
            with st.expander(explanation['label']):
                st.dataframe(pd.DataFrame(explanation['top_features']), hide_index=True, width='stretch')
                st.write(f"Audio contribution: {explanation['audio_logit_contribution']:.4f} | Visual contribution: {explanation['visual_logit_contribution']:.4f} | Intercept: {explanation['intercept']:.4f}")
                st.caption(f"Score reconstruction: {explanation['reconstruction']}")
    st.subheader('Input quality')
    st.json(report['quality'])
    if report['quality']['all_zero_audio']:
        st.warning('The audio track is entirely silent. Interpret the combined scores cautiously.')
    st.write(report['limitations'])
    payload = json.dumps(report, indent=2)
    st.download_button('Download JSON report', payload, 'video_behavior_report.json', 'application/json')
    rows = ''.join('<tr><td>'+html.escape(r['label'])+'</td><td>'+f"{r['model_score']:.4f}"+'</td><td>'+html.escape(r['threshold_result'])+'</td></tr>' for r in report['labels'])
    explanation_html = ''
    for e in report.get('explanations', []):
        feature_rows = ''.join('<tr><td>'+html.escape(r['feature'])+'</td><td>'+f"{r['logit_contribution']:.4f}"+'</td><td>'+html.escape(r['direction'])+'</td></tr>' for r in e['top_features'])
        explanation_html += '<h3>'+html.escape(e['label'])+'</h3><table border="1"><tr><th>Feature</th><th>Logit contribution</th><th>Direction</th></tr>'+feature_rows+'</table>'
    document = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Video behavior research report</title><body><h1>AV-ASD Video Behavior Analysis</h1><p>'+html.escape(report['limitations'])+'</p><h2>Recording summary</h2><p>'+html.escape(report.get('summary', summarize_results(report['labels'], report['threshold'])))+'</p><p>Frozen threshold: '+str(report['threshold'])+'</p><table border="1"><tr><th>Label</th><th>Model score</th><th>Threshold result</th></tr>'+rows+'</table><h2>Model feature contributions</h2><p>Logit contributions explain the model, not behavioral causes. Top ten features omit the intercept and remaining features; values are not percentages.</p>'+explanation_html+'<h2>Input quality and provenance</h2><pre>'+html.escape(payload)+'</pre></body></html>'
    st.download_button('Download HTML report', document, 'video_behavior_report.html', 'text/html')

# REPORT_WORKSPACE_INTEGRATION
# Formatting existing results only; no additional model inference.
import sys as _workspace_sys
_workspace_sys.path.insert(0, str(ROOT / 'scripts/report_workspace'))
from theme import apply_theme as _workspace_theme
from downloads import pdf_download as _workspace_download
_workspace_theme(st)
_workspace_report = st.session_state.get('av_asd_video_report')
if _workspace_report:
    _workspace_download(st, _workspace_report, 'professional_pdf_video')
# END_REPORT_WORKSPACE_INTEGRATION
