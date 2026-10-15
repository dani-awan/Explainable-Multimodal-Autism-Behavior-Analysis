"""A new project-themed entry point; the original dashboard/app.py is preserved."""
from pathlib import Path
import streamlit as st
ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='Multimodal Research Workspace',page_icon='🔬',layout='wide')
st.sidebar.markdown('### MULTIMODAL\nResearch workspace')
core=[('14_Research_Home.py','Research home'),('9_Video_Behavior_Analysis.py','Video analysis'),('15_Video_Behavior_Timeline.py','Video timeline'),('8_Research_Analysis.py','Feature analysis'),('13_Report_Workspace.py','Reports and assistant')]
experiments=[('5_ASD_FEAT_E6_Experiments.py','Grouped experiments'),('6_E6_Model_Explanations.py','Model explanations'),('7_E6_Frozen_Test.py','Frozen test results'),('10_Research_Extensions.py','XGBoost and temporal models'),('11_DREAM_Data_Quality.py','DREAM data review'),('12_DREAM_Activity_Results.py','DREAM activity results')]
def pages(items):return [st.Page('pages/'+file,title=title,default=file=='14_Research_Home.py') for file,title in items if (ROOT/'pages'/file).is_file()]
groups={'WORKSPACE':pages(core),'RESEARCH EVIDENCE':pages(experiments)}
navigation=st.navigation({k:v for k,v in groups.items() if v})
st.sidebar.caption('Muhammad Danial\n\nHamza Rasheed\n\nMuhammad Fareed')
navigation.run()
