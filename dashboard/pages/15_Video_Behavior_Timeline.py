"""Exploratory window scores with source-video playback and fitted explanations."""
from pathlib import Path
import hashlib,json,sys,tempfile,inspect
import streamlit as st
import pandas as pd
import altair as alt
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/video_timeline'))
sys.path.insert(0,str(ROOT/'scripts/report_workspace'))
from timeline_core import analyze,load_bundle,rows,selected_report,LIMIT
from report_pdf import build_pdf
from theme import apply_theme,hero
st.set_page_config(page_title='Video Behavior Timeline',page_icon='🎞️',layout='wide')
apply_theme(st)
hero(st,'Video behavior timeline','Explore how the frozen AV-ASD model scores successive video windows. Select a segment to inspect its playback, quality checks and fitted explanation.')
st.info(LIMIT)
st.caption('MP4 with audio · up to 120 seconds / 7200 frames / 200 MiB · temporary media stored under the project on F: and removed after extraction')
upload=st.file_uploader('Choose an MP4',type=['mp4'],key='timeline_upload')
window=st.select_slider('Requested window length',options=[5,10,15],value=10,format_func=lambda v:f'{v} seconds')
fingerprint=hashlib.sha256(upload.getvalue()).hexdigest()+':'+str(window) if upload else None
if st.session_state.get('timeline_fingerprint')!=fingerprint:
    st.session_state.pop('timeline_result',None)
    st.session_state.pop('timeline_selected',None)
    st.session_state.pop('timeline_last_chart',None)
    st.session_state['timeline_fingerprint']=fingerprint
if st.button('Build exploratory timeline',type='primary'):
    st.session_state.pop('timeline_result',None)
    if upload is None:st.error('Choose a video first.')
    elif upload.size>200*1024**2:st.error('Video exceeds 200 MiB.')
    else:
        try:
            progress=st.progress(0,text='Checking the frozen model and video...')
            bundle=load_bundle()
            temp=ROOT/'results/video_timeline_temp';temp.mkdir(parents=True,exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='timeline_',dir=temp) as folder:
                source=Path(folder)/'clip.mp4';source.write_bytes(upload.getvalue())
                report=analyze(source,bundle,window,lambda n,total:progress.progress(n/total,text=f'Processed window {n}/{total}'),temp_root=folder)
            st.session_state['timeline_result']=report;progress.empty()
        except Exception as error:st.error('Timeline stopped: '+str(error))
report=st.session_state.get('timeline_result')
if report and upload:
    frame=pd.DataFrame(rows(report))
    a,b,c=st.columns(3);a.metric('Windows',len(report['segments']));b.metric('Video duration',f"{report['duration_seconds']:.1f} s");c.metric('Frozen threshold',f"{report['threshold']:.2f}")
    st.subheader('Window score map')
    pick=alt.selection_point(name='segment_pick',fields=['segment_index'],toggle=False)
    chart=alt.Chart(frame).mark_rect(cursor='pointer').encode(x=alt.X('start_seconds:Q',title='Video time in seconds',scale=alt.Scale(domain=[0,report['duration_seconds']])),x2='end_seconds:Q',y=alt.Y('label:N',title=None,sort=list(dict.fromkeys(frame['label'])),axis=alt.Axis(labelLimit=280)),color=alt.Color('model_score:Q',title='Model score',scale=alt.Scale(domain=[0,1],scheme='tealblues')),tooltip=[alt.Tooltip('start_seconds:Q',title='Start',format='.2f'),alt.Tooltip('end_seconds:Q',title='End',format='.2f'),'label:N',alt.Tooltip('model_score:Q',title='Score',format='.3f'),'threshold_result:N']).add_params(pick).properties(height=340)
    chart_width={'width':'stretch'} if 'width' in inspect.signature(st.altair_chart).parameters else {'use_container_width':True}
    event=st.altair_chart(chart,on_select='rerun',key='timeline_chart',**chart_width)
    chosen=event.get('selection',{}).get('segment_pick',[]) if event else []
    chart_index=int(chosen[0]['segment_index']) if chosen else None
    if chart_index is not None and chart_index!=st.session_state.get('timeline_last_chart'):
        st.session_state['timeline_selected']=chart_index;st.session_state['timeline_last_chart']=chart_index
    if st.session_state.get('timeline_selected') not in range(len(report['segments'])):st.session_state['timeline_selected']=0
    index=st.selectbox('Selected window',range(len(report['segments'])),format_func=lambda n:f"Window {n+1}: {report['segments'][n]['start_seconds']:.2f} - {report['segments'][n]['end_seconds']:.2f} seconds",key='timeline_selected')
    segment=report['segments'][index]
    st.caption('Click a heatmap segment or use the selector. Windows use nominal frame timing, not verified event boundaries. A final remainder under 2 seconds is merged into the preceding window.')
    left,right=st.columns([1.1,1],gap='large')
    with left:
        st.video(upload.getvalue(),start_time=segment['start_seconds'],end_time=segment['end_seconds'])
        st.caption(f"Selected interval: {segment['start_seconds']:.2f} - {segment['end_seconds']:.2f} seconds. Browser seeking can be approximate.")
        with st.expander('Window input quality'):st.json(segment['quality'])
        if segment['quality'].get('all_zero_audio'):st.warning('This window has silent audio.')
    with right:
        st.subheader('Selected-window outputs')
        st.dataframe(pd.DataFrame(segment['labels']).sort_values('model_score',ascending=False),hide_index=True,width='stretch')
        st.caption('Above threshold means a model output exceeded 0.30; it is not confirmation of a behavior.')
    st.subheader('Explanation for a label')
    choices=[e['label'] for e in segment['explanations']]
    label=st.selectbox('Label',choices,key='timeline_explanation_label')
    e=next(e for e in segment['explanations'] if e['label']==label)
    st.write(f"Audio contribution: {e['audio_logit_contribution']:.4f} · Visual contribution: {e['visual_logit_contribution']:.4f} · Intercept: {e['intercept']:.4f}")
    st.dataframe(pd.DataFrame(e['top_features']),hide_index=True,width='stretch')
    st.caption('Contributions reconstruct the fitted logit; they are not percentages, clinical symptoms or causal effects. Top rows omit other features.')
    st.subheader('Export the research evidence')
    x,y,z=st.columns(3)
    x.download_button('Timeline CSV',frame.to_csv(index=False),'video_timeline_scores.csv','text/csv')
    y.download_button('Timeline JSON',json.dumps(report,indent=2),'video_timeline_report.json','application/json')
    z.download_button('Selected-window PDF',build_pdf(selected_report(report,index)),'selected_window_report.pdf','application/pdf')
    st.caption('Timeline predictions are new exploratory outputs in this session. Original dataset predictions and frozen models are preserved.')
