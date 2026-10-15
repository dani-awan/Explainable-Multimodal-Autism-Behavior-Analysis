"""Check PDF exports, sourced retrieval and page syntax; no model inference."""
from pathlib import Path
import json
from pypdf import PdfReader,PdfWriter
import io
from report_pdf import build_pdf,validate_report
from pdf_chat import read_pdf,answer
ROOT=Path(__file__).resolve().parents[2]
def rejected(function):
    try:function()
    except (ValueError,TypeError,KeyError):return
    raise AssertionError('Invalid input was accepted.')
def main():
    sample={'model':'rgb_behavior_pca32_lr','created_at_utc':'Software validation example','model_score':0.1661,'threshold':0.5,'input_features':1036,'limits':['Model output is uncalibrated. This does not establish autism diagnosis.'],'branch_logit_contributions':[{'Branch':'rgb','Logit contribution':-0.740099},{'Branch':'behavior','Logit contribution':0.969165}],'intercept':-1.842596,'decision_logit':-1.613530}
    payload=build_pdf(sample);pages=read_pdf(payload);text='\n'.join(p['text'] for p in pages)
    assert '16.61%' in text and 'Muhammad Fareed' in text and 'Below threshold' in text
    assert 'Muhammad Freeh' not in text
    result=answer(pages,'model score kitna hai?');assert result['sources']
    for s in result['sources']:assert s['text'] in pages[s['page']-1]['text']
    assert not answer(pages,'volcano tectonic eruption')['sources']
    assert answer(pages,'Summarize this report')['sources']
    rejected(lambda:validate_report(dict(sample,model_score=float('nan'))))
    rejected(lambda:validate_report(dict(sample,model_score=1.1)))
    rejected(lambda:read_pdf(b'not pdf'))
    blank=PdfWriter();blank.add_blank_page(width=500,height=700);b=io.BytesIO();blank.write(b);rejected(lambda:read_pdf(b.getvalue()))
    video={'model':'AV-ASD combined Logistic Regression','threshold':0.3,'labels':[{'label':'Eye contact avoidance','model_score':0.31,'threshold_result':'Above threshold'},{'label':'Background','model_score':0.2,'threshold_result':'Below threshold'}],'quality':{'all_zero_audio':True},'limitations':'Arbitrary new-video accuracy is unverified.'}
    vt='\n'.join(x['text'] for x in read_pdf(build_pdf(video)));assert '31.00%' in vt and 'Above threshold' in vt
    rejected(lambda:validate_report(dict(video,labels=video['labels']*2)))
    for name in ['8_Research_Analysis.py','9_Video_Behavior_Analysis.py','13_Report_Workspace.py']:
        path=ROOT/'dashboard/pages'/name
        compile(path.read_text(encoding='utf-8-sig'),name,'exec')
    print('PASS: video/feature PDF content, score validation, page-source integrity, unknown question, summary, corrupt/scanned PDF rejection and dashboard syntax')
    print('Software checks only; browser upload and chat review remain necessary. No models loaded or predictions generated.')
if __name__=='__main__':main()
