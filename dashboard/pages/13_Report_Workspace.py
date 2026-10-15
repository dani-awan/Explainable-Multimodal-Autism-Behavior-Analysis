from pathlib import Path
import hashlib,json,sys
import streamlit as st
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/report_workspace'))
from report_pdf import build_pdf,validate_report,TEAM
from pdf_chat import read_pdf,answer
from report_facts import facts,compare
from theme import apply_theme,hero
st.set_page_config(page_title='Reports and Assistant',page_icon='📄',layout='wide')
apply_theme(st)
hero(st,'Reports and document assistant','Turn an existing analysis into a readable PDF. Ask source-grounded questions, inspect the evidence and compare two exported research reports.')
st.caption('Muhammad Danial · Hamza Rasheed · Muhammad Fareed · Local processing')
one,two,three=st.tabs(['Create a report','Ask your report','Compare reports'])
with one:
    left,right=st.columns([1.5,1],gap='large')
    with left:
        st.subheader('Prepare a research PDF')
        options=['Upload exported report JSON']
        if st.session_state.get('av_asd_video_report'):options.append('Current video analysis')
        if st.session_state.get('research_analysis_result'):options.append('Current prepared-feature analysis')
        choice=st.selectbox('Analysis source',options)
        report=None
        if choice=='Current video analysis':report=st.session_state['av_asd_video_report']
        elif choice=='Current prepared-feature analysis':report=st.session_state['research_analysis_result'][0]
        else:
            upload=st.file_uploader('Exported video or feature-analysis JSON',type=['json'],key='workspace_json_v2')
            if upload:
                try:
                    if upload.size>2*1024**2:raise ValueError('JSON exceeds 2 MiB.')
                    report=json.loads(upload.getvalue())
                    if not isinstance(report,dict):raise ValueError('JSON must contain a report object.')
                    report=dict(report,document_origin='User-supplied JSON; content not independently authenticated')
                except Exception as e:st.error('Report rejected: '+str(e))
        if report is not None:
            try:
                st.success('Compatible schema: '+validate_report(report))
                st.download_button('Download professional PDF',build_pdf(report),'research_analysis_report.pdf','application/pdf',type='primary')
                with st.expander('Inspect the source report'):st.json(report)
                st.caption('Existing scores are formatted into a PDF; this operation does not run a model. Uploaded JSON content is not independently authenticated.')
            except Exception as e:st.error('Cannot generate PDF: '+str(e))
    with right:
        st.markdown('<div class="research-card"><div class="tag">RESEARCH REPORT</div><h3>Evidence in a readable format</h3><p>Model scores, frozen threshold, available quality checks, fitted explanations and provenance. Every report preserves interpretation limits.</p></div>',unsafe_allow_html=True)
        st.info('Use Video Analysis or Feature Analysis first. You can also upload its exported JSON here.')
with two:
    left,right=st.columns([2,1])
    with left:uploaded=st.file_uploader('Upload a text-readable PDF · 15 MiB / 80 pages',type=['pdf'],key='workspace_pdf_v2')
    with right:language=st.selectbox('Answer language',['English','Roman Urdu'])
    fingerprint=hashlib.sha256(uploaded.getvalue()).hexdigest() if uploaded else None
    if st.session_state.get('workspace_pdf_hash_v2')!=fingerprint:
        for key in ['workspace_pages_v2','workspace_messages_v2','workspace_pdf_error_v2']:st.session_state.pop(key,None)
        st.session_state['workspace_pdf_hash_v2']=fingerprint
        if uploaded:
            try:st.session_state['workspace_pages_v2']=read_pdf(uploaded.getvalue())
            except Exception as e:st.session_state['workspace_pdf_error_v2']=str(e)
    if st.session_state.get('workspace_pdf_error_v2'):st.error(st.session_state['workspace_pdf_error_v2'])
    pages=st.session_state.get('workspace_pages_v2')
    if pages:
        fs=facts(pages)
        a,b,c=st.columns(3);a.metric('PDF pages',len(pages));b.metric('Readable pages',sum(bool(p['text']) for p in pages));c.metric('Processing','Local')
        if 'score_percent' in fs:st.caption('Extracted model score: '+fs['score_percent']['value']+'% · Uncalibrated report output')
        st.caption('Quick questions or type your own. Answers refer to this PDF; source text is available under each response.')
        prompts=['Report overview','Model score','Frozen threshold','Limitations']
        quick=None
        for col,prompt in zip(st.columns(4),prompts):
            if col.button(prompt,key='faq_'+prompt):quick={'Report overview':'Ye kia ha','Model score':'Model score kitna hai?','Frozen threshold':'What is the threshold?','Limitations':'Explain the limitations'}[prompt]
        if st.button('Clear conversation',key='clear_chat_v2'):st.session_state['workspace_messages_v2']=[]
        messages=st.session_state.setdefault('workspace_messages_v2',[])
        def display(result):
            st.write(result['answer'])
            for source in result['sources']:
                with st.expander('Evidence · PDF page '+str(source['page'])):st.text(source['text'])
        for message in messages:
            with st.chat_message(message['role']):
                if message['role']=='user':st.write(message['content'])
                else:display(message['content'])
        question=st.chat_input('Ask about the model, score, threshold or limitations',key='workspace_question_v2') or quick
        if question:
            with st.chat_message('user'):st.write(question)
            result=answer(pages,question,language)
            with st.chat_message('assistant'):display(result)
            messages.extend([{'role':'user','content':question},{'role':'assistant','content':result}])
            if len(messages)>40:del messages[:-40]
        if messages:st.download_button('Download conversation',json.dumps({'pdf_sha256':fingerprint,'messages':messages},indent=2),'report_conversation.json','application/json')
    else:st.info('Upload a report to start. Scanned image-only PDFs need OCR first.')
    st.caption('Known report fields receive direct sourced answers. Other questions use matching excerpts. This is a local document assistant, not a generative language model. Changing the PDF clears its conversation.')
with three:
    st.subheader('Compare two report documents')
    st.caption('See extracted fields side by side. A score difference is shown only when task, model name, artifact hash and threshold match. It is not a clinical improvement measure.')
    a,b=st.columns(2)
    file_a=a.file_uploader('Report A PDF',type=['pdf'],key='compare_a_v2')
    file_b=b.file_uploader('Report B PDF',type=['pdf'],key='compare_b_v2')
    if file_a and file_b:
        try:
            result=compare(read_pdf(file_a.getvalue()),read_pdf(file_b.getvalue()))
            st.dataframe(result['rows'],hide_index=True,width='stretch')
            if result['compatible_fields']:
                st.metric('Reported score difference · B minus A',f"{result['difference_percentage_points']:+.2f} percentage points")
                st.caption('Compatibility here is based on document text; it is not independent verification of the model or input provenance.')
            else:st.info('Required fields are missing or differ. No numeric score difference is asserted. AV-ASD multi-label outputs are not one overall autism score.')
            for col,key in zip([a,b],['facts_a','facts_b']):
                with col:
                    with st.expander('Inspect extracted field evidence'):
                        for name,value in result[key].items():st.write(name+' · page '+str(value['page']));st.text(value['text'])
            st.download_button('Download comparison JSON',json.dumps(result,indent=2),'report_comparison.json','application/json')
        except Exception as e:st.error('Comparison unavailable: '+str(e))
    st.caption('Uploaded PDFs stay in the active session and are not sent to an external API or written to project storage.')
