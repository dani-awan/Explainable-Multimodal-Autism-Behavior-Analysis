from pathlib import Path
import sys
import streamlit as st
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/report_workspace'))
from theme import apply_theme,hero
apply_theme(st)
hero(st,'Multimodal behavior research','A local research workspace for video analysis, prepared-feature models and therapy-activity experiments. Inspect evidence, export reports and ask questions about the results.')
st.caption('Muhammad Danial · Hamza Rasheed · Muhammad Fareed')
cols=st.columns(3,gap='large')
cards=[('AV-ASD','Video behavior analysis','Explore ten behavioral-label scores using audio and handcrafted visual features.','pages/9_Video_Behavior_Analysis.py','Open video analysis'),('ASD-FEAT','Prepared-feature research','Analyze compatible RGB and human-coded behavior inputs with the preserved frozen model.','pages/8_Research_Analysis.py','Open feature analysis'),('REPORTS','Reports and document assistant','Create a PDF, ask source-grounded questions and compare two exported reports.','pages/13_Report_Workspace.py','Open report workspace')]
for col,(tag,title,body,path,label) in zip(cols,cards):
    with col:
        st.markdown(f'<div class="research-card"><div class="tag">{tag}</div><h3>{title}</h3><p>{body}</p></div>',unsafe_allow_html=True)
        st.page_link(path,label=label)
st.subheader('Explore a video over time')
st.write('Inspect fixed-window scores, select a segment for playback and export its model explanation. Segment-level predictive accuracy is not established.')
st.page_link('pages/15_Video_Behavior_Timeline.py',label='Open video timeline')
st.subheader('Three distinct research tasks')
st.dataframe([{'Track':'AV-ASD','Target':'Ten video behavior labels','Documented support':'527 usable clips'},{'Track':'ASD-FEAT','Target':'Current-visit annotations','Documented support':'1,510 aligned development recordings'},{'Track':'DREAM','Target':'Imitation, joint attention and turn taking','Documented support':'2,615 selected sessions / 56 participants'}],hide_index=True,width='stretch')
st.info('Model outputs support research inspection. They do not establish clinical autism diagnosis, stage, treatment or a validated chance of autism. The three tasks have different endpoints and their scores must stay separate.')
st.subheader('A focused demonstration')
st.write('1. Run an existing feature demonstration or a supported short MP4 analysis.\n\n2. Download a professional PDF and review its score, quality and explanation.\n\n3. Open Reports and assistant to ask questions or compare compatible reports.\n\n4. Use Research Evidence to show participant grouping, model comparisons and limitations.')
