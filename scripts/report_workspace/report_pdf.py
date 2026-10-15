"""Create a research PDF from an existing report; never fit or run a model."""
import io
import math
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether

TEAM='Muhammad Danial, Hamza Rasheed, Muhammad Fareed'
LIMIT='Research output only. Model scores are uncalibrated and do not establish autism diagnosis, severity or treatment. Threshold outcomes are not confirmed observations.'
def numeric(v):
    if isinstance(v,bool):raise ValueError('Boolean score is invalid.')
    n=float(v)
    if not math.isfinite(n):raise ValueError('Numeric value must be finite.')
    return n
def validate_report(report):
    if not isinstance(report,dict):raise ValueError('Report must be a JSON object.')
    threshold=numeric(report.get('threshold'))
    if not 0 <= threshold <= 1:raise ValueError('Threshold must be in [0,1].')
    if 'labels' in report:
        labels=report['labels']
        if not isinstance(labels,list) or not 1 <= len(labels) <= 30:raise ValueError('Invalid behavioral label list.')
        names=[]
        for row in labels:
            if not isinstance(row,dict) or not isinstance(row.get('label'),str) or not row['label'].strip():raise ValueError('Invalid label.')
            s=numeric(row.get('model_score'))
            if not 0 <= s <= 1:raise ValueError('Score must be in [0,1].')
            names.append(row['label'])
        if len(names)!=len(set(names)):raise ValueError('Duplicate labels.')
        return 'AV-ASD behavioral labels'
    if report.get('model')!='rgb_behavior_pca32_lr':raise ValueError('Unsupported report schema. Use the exported video or Research Analysis JSON.')
    s=numeric(report.get('model_score'))
    if not 0 <= s <= 1:raise ValueError('Score must be in [0,1].')
    return 'ASD-FEAT prepared features'
def build_pdf(report):
    task=validate_report(report)
    font=Path(__file__).parent/'assets/DejaVuSans.ttf'
    if 'ResearchSans' not in pdfmetrics.getRegisteredFontNames():pdfmetrics.registerFont(TTFont('ResearchSans',str(font)))
    navy=colors.HexColor('#112B43');teal=colors.HexColor('#087F8C');gray=colors.HexColor('#536574')
    styles=getSampleStyleSheet()
    for name in ['Normal','BodyText','Heading1','Heading2','Title']:
        styles[name].fontName='ResearchSans';styles[name].textColor=navy
    styles['BodyText'].fontSize=10;styles['BodyText'].leading=14;styles['BodyText'].spaceAfter=6
    styles['Title'].fontSize=26;styles['Title'].leading=32;styles['Title'].spaceAfter=10;styles['Title'].alignment=TA_LEFT
    styles['Heading1'].fontSize=14;styles['Heading1'].leading=19;styles['Heading1'].spaceBefore=12;styles['Heading1'].spaceAfter=7
    small=ParagraphStyle('Small',parent=styles['BodyText'],fontSize=8,leading=11,textColor=gray)
    def para(v,style='BodyText'):return Paragraph(escape(str(v)),styles[style] if isinstance(style,str) else style)
    def table(headers,rows,widths):
        header=ParagraphStyle('TableHeader',parent=small,textColor=colors.white)
        data=[[para(x,header) for x in headers]]+[[para(x,small) for x in row] for row in rows]
        t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F0F5F8')]),('GRID',(0,0),(-1,-1),.35,colors.HexColor('#D6E0E7')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8)]))

        return t
    stream=io.BytesIO();doc=SimpleDocTemplate(stream,pagesize=A4,leftMargin=44,rightMargin=44,topMargin=53,bottomMargin=48,title='Research analysis report',author=TEAM)
    story=[para('RESEARCH WORKSPACE',small),Spacer(1,12),para('Research analysis report','Title'),para(task),para('Team: '+TEAM,small),Spacer(1,10),para(LIMIT)]
    stamp=report.get('generated_at_utc',report.get('created_at_utc','Not provided'))
    story += [para('Analysis overview','Heading1'),para('Report timestamp: '+str(stamp),small),para('Model: '+str(report.get('model','Not provided'))),para('Frozen threshold: '+f"{float(report['threshold']):.2f}")]
    from research_prediction import prediction
    result=prediction(report)
    if result:
        story += [para('ASD-related research prediction','Heading1'),
                  para('Model classification: '+result['classification']),
                  para(result['scope']),para(result['interpretation']),
                  para(result['evidence_limit'],small),
                  para('Recommended next step','Heading1'),para(result['next_step']),
                  para('Assessment guidance: https://www.cdc.gov/autism/diagnosis/index.html',small)]
    if task.startswith('AV-ASD'):
        rows=[]
        for r in sorted(report['labels'],key=lambda x:-float(x['model_score'])):
            score=float(r['model_score']);rows.append([r['label'],f'{100*score:.2f}%', 'Above threshold' if score>=float(report['threshold']) else 'Below threshold'])
        story += [para('Behavioral label scores','Heading1'),table(['Label','Model score','Threshold output'],rows,[260,78,169]),para('Labels are scored independently. Their scores do not add up to 100%.',small)]
    else:
        score=float(report['model_score'])
        story += [table(['Measure','Research result'],[['Uncalibrated model score',f'{100*score:.2f}%'],['Threshold output','Above threshold' if score>=float(report['threshold']) else 'Below threshold'],['Clinical diagnosis','Not established'],['Severity or stage','Not assessed']],[220,287])]
    if report.get('summary'):story += [para('Recording summary','Heading1'),para(report['summary'])]
    quality=report.get('quality',{})
    if quality:
        story += [KeepTogether([para('Input quality','Heading1'),table(['Check','Recorded value'],[[k,str(v)] for k,v in quality.items()],[225,282])])]
    branches=report.get('branch_logit_contributions',[])
    if branches:
        story += [para('Explanation of the fitted score','Heading1'),table(['Branch','Logit contribution'],[[r['Branch'],f"{numeric(r['Logit contribution']):.6f}"] for r in branches],[225,282]),para('Intercept: '+str(report.get('intercept','Not provided'))+'. Decision logit: '+str(report.get('decision_logit','Not provided'))+'. Contributions are not percentages or symptom measurements.',small)]
    for e in report.get('explanations',[]):
        story += [para('Fitted explanation for '+str(e['label']),'Heading1'),para('Audio logit contribution: '+str(e.get('audio_logit_contribution'))+'; visual: '+str(e.get('visual_logit_contribution'))+'; intercept: '+str(e.get('intercept')),small)]
        rows=[[r['feature'],f"{numeric(r['logit_contribution']):.5f}",r.get('direction','')] for r in e.get('top_features',[])[:10]]
        if rows:story.append(table(['Feature','Logit contribution','Direction'],rows,[280,100,127]))
        story.append(para('Top features omit other contributions. Contributions explain the fitted model, not behavioral causes.',small))
    limits=report.get('limits',[])
    if not limits:limits=[report.get('limitations',LIMIT)]
    story += [para('Interpretation and limitations','Heading1')]
    for value in limits:story.append(para(value))
    story += [para('Input and model provenance','Heading1')]
    for k in ['document_origin','input_source','input_features','features','behavior_source','model_sha256','video_sha256']:
        if k in report:story.append(para(k.replace('_',' ').capitalize()+': '+str(report[k]),small))
    story.append(para('Prepared for research demonstration. No clinical diagnosis or treatment recommendation is issued by this report.',small))
    def footer(canvas,document):
        canvas.setStrokeColor(teal);canvas.line(44,34,A4[0]-44,34);canvas.setFont('ResearchSans',8);canvas.setFillColor(gray);canvas.drawString(44,22,'AV-ASD research workspace | Local report');canvas.drawRightString(A4[0]-44,22,f'Page {document.page}')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    return stream.getvalue()
