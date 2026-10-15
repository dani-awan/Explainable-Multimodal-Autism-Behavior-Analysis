"""Local extractive report Q&A with exact page sources; no remote API."""
import io
import re
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ALIASES={'kitna':'score value results','persent':'score percent','percent':'score percentage','percentage':'score percentage','kia':'','kya':'','hai':'','ha':'','iss':'','is':'','mein':'','main':'','ka':'','ki':'','ke':'','aur':'','or':'','batao':'','batay':'','natija':'results conclusion','nateeja':'results conclusion','khulasa':'summary','samjhao':'explanation','stage':'severity stage','chance':'probability score uncalibrated','audio':'audio','video':'video','accuracy':'accuracy','model':'model','limitations':'limitations interpretation','dignosis':'diagnosis','diagnose':'diagnosis','autism':'autism diagnosis'}
def read_pdf(payload):
    if not isinstance(payload,bytes) or not payload.startswith(b'%PDF-'):raise ValueError('Choose a valid PDF file.')
    if len(payload)>15*1024**2:raise ValueError('PDF exceeds 15 MiB.')
    reader=PdfReader(io.BytesIO(payload),strict=False)
    if reader.is_encrypted:raise ValueError('Password-protected PDF is unsupported. Upload an unlocked copy.')
    if not 1<=len(reader.pages)<=80:raise ValueError('PDF must contain 1 to 80 pages.')
    pages=[];total=0
    for i,p in enumerate(reader.pages,1):
        contents=p.get_contents()
        if contents and len(contents.get_data())>8*1024**2:raise ValueError('PDF page is too complex for local extraction.')
        text=(p.extract_text() or '').strip();total+=len(text)
        if total>500000:raise ValueError('PDF text exceeds the local limit.')
        pages.append({'page':i,'text':text})
    if total<50:raise ValueError('No readable text found. Scanned PDFs need OCR before upload.')
    return pages
def chunks(pages):
    out=[]
    for page in pages:
        lines=page['text'].splitlines()
        for i in range(0,len(lines),8):
            text='\n'.join(lines[max(0,i-2):i+8]).strip()
            if text:out.append({'page':page['page'],'text':text})
    return out
def answer(pages,question):
    if not isinstance(question,str) or not question.strip():raise ValueError('Enter a question.')
    if len(question)>1000:raise ValueError('Question exceeds 1,000 characters.')
    q=question.lower();terms=re.findall(r'\w+',q)
    query=' '.join(ALIASES.get(t,t) for t in terms)
    items=chunks(pages)
    generic=' '.join(terms) in {'ye kia ha','ye kya hai','ye kia hai','ye kya ha','ye kiya ha','ye report kia ha','ye report kya hai','what is this','what is this report','is report ke bare mein batao','report samjhao'}
    if generic or any(w in terms for w in ['summarize','summary','khulasa','overview']):
        refs=[]
        for pg in pages:
            if pg['text']:refs.append({'page':pg['page'],'text':pg['text'][:900]})
            if len(refs)==3:break
        return {'answer':'Report overview: here are opening excerpts from up to three pages. Read the full report for the complete findings.','sources':refs,'mode':'Extractive overview'}
    if not query.strip():return {'answer':'Please include a report topic, such as model score, threshold, precision or limitations.','sources':[],'mode':'No match'}
    vector=TfidfVectorizer(lowercase=True,ngram_range=(1,2),token_pattern=r'(?u)\b\w+\b')
    corpus=vector.fit_transform([r['text'] for r in items]+[query]);scores=cosine_similarity(corpus[-1],corpus[:-1]).ravel()
    order=scores.argsort()[::-1];refs=[]
    for idx in order:
        if scores[idx]<0.07:break
        if any(x['page']==items[idx]['page'] and x['text']==items[idx]['text'] for x in refs):continue
        refs.append(items[idx])
        if len(refs)==3:break
    if not refs:return {'answer':'I could not find a sufficiently matching passage in this PDF. Try wording from the report. I cannot establish an answer from information that is absent.','sources':[],'mode':'No match'}
    return {'answer':'The uploaded report contains these relevant passages. These are exact excerpts, not an independent clinical interpretation.','sources':refs,'mode':'Local report search'}

_search_answer=answer

def answer(pages,question,language='English'):
    """Known field questions receive direct sourced answers; others use excerpts."""
    from report_facts import facts
    if not isinstance(question,str) or not question.strip():raise ValueError('Enter a question.')
    if len(question)>1000:raise ValueError('Question exceeds 1,000 characters.')
    q=' '.join(re.findall(r'\w+',question.lower()))
    roman=language=='Roman Urdu'
    f=facts(pages)
    def result(text,keys):
        refs=[]
        for k in keys:
            if k in f:
                refs.append({'page':f[k]['page'],'text':f[k]['text']})
        return {'answer':text,'sources':refs,'mode':'Sourced report fields'}
    generic=q in {'ye kia ha','ye kya hai','ye kia hai','ye kya ha','ye kiya ha','ye report kia ha','ye report kya hai','what is this','what is this report','is report ke bare mein batao','report samjhao'} or any(w in q.split() for w in ['summary','summarize','overview','khulasa'])
    if generic and 'task' in f:
        pieces=[('Ye '+f['task']['value']+' ki research report hai.') if roman else ('This report concerns '+f['task']['value']+'.')]
        if 'model' in f:pieces.append('Model: '+f['model']['value']+'.')
        if 'score_percent' in f:pieces.append('Uncalibrated model score: '+f['score_percent']['value']+'%.')
        if 'threshold' in f:pieces.append('Frozen threshold: '+f['threshold']['value']+'.')
        if 'diagnosis' in f:pieces.append('Is report mein clinical diagnosis establish nahi ki gayi.' if roman else 'Clinical diagnosis is listed as not established.')
        return result(' '.join(pieces),['task','model','score_percent','threshold','diagnosis'])
    prediction_question=any(w in q.split() for w in ['prediction','predict','predicted','pridict','pridect','predicet','classification','positive','negative','natija','nateeja'])
    if prediction_question and 'research_classification' in f:
        value=f['research_classification']['value']
        if roman:
            text='PDF ke mutabiq research model ki classification '+value+' hai.'
            if 'score_percent' in f:text+=' Model score '+f['score_percent']['value']+'% hai.'
            if 'threshold' in f:text+=' Fixed threshold '+f['threshold']['value']+' hai.'
            text+=' Positive autism confirm nahi karta aur Negative autism rule out nahi karta. Ye experimental research output hai, clinical diagnosis ya autism ka validated chance nahi.'
        else:
            text='The PDF reports a research model classification of '+value+'.'
            if 'score_percent' in f:text+=' The model score is '+f['score_percent']['value']+'%.'
            if 'threshold' in f:text+=' The frozen threshold is '+f['threshold']['value']+'.'
            text+=' Positive does not confirm autism; Negative does not rule it out. This is an experimental research output, not a diagnosis or a validated chance of autism.'
        reply=result(text,['research_classification','score_percent','threshold','diagnosis'])
        for pg in pages:
            match=re.search(r'Experimental research classification,.*?chance of\s+autism\.',pg['text'],re.S)
            if match:reply['sources'].append({'page':pg['page'],'text':match.group(0)})
            match=re.search(r'If there are concerns about development or behavior,.*?regardless of this model result\.',pg['text'],re.S)
            if match:
                reply['answer']+=(' Development ya behavior ka concern ho to pediatrician ya qualified autism specialist se assessment karwayein, result jo bhi ho.' if roman else ' If development or behavior is a concern, seek assessment from a pediatrician or qualified autism specialist regardless of this result.')
                reply['sources'].append({'page':pg['page'],'text':match.group(0)})
        return reply
    if any(w in q.split() for w in ['autism','diagnosis','diagnose','dignosis','positive','negative']) and 'diagnosis' in f:
        return result('Report mein clinical diagnosis "Not established" likhi hai. Model ka score autism hai ya nahi ka faisla nahi karta.' if roman else 'The report lists clinical diagnosis as "Not established". Its model score does not determine whether a child has autism.', ['diagnosis'])
    if any(w in q.split() for w in ['stage','severity']) and 'stage' in f:
        return result('Report ke mutabiq severity ya stage "Not assessed" hai.' if roman else 'The report lists severity or stage as "Not assessed".', ['stage'])
    if 'threshold' in q and 'threshold' in f:
        s=('Report mein frozen threshold '+f['threshold']['value']+' hai.') if roman else ('The frozen threshold reported in this PDF is '+f['threshold']['value']+'.')
        return result(s,['threshold'])
    if any(w in q.split() for w in ['score','percent','percentage','persent']) and 'score_percent' in f:
        s=('Report mein model score '+f['score_percent']['value']+'% hai. Ye uncalibrated output hai, autism ka validated chance nahi.') if roman else ('The reported model score is '+f['score_percent']['value']+'%. It is an uncalibrated output, not a validated chance of autism.')
        keys=['score_percent']
        if 'threshold_result' in f:s+=' '+f['threshold_result']['value']+'.';keys.append('threshold_result')
        return result(s,keys)
    if 'model' in q and 'model' in f and not any(x in q for x in ['accuracy','precision','recall','auc','f1','feature']):
        return result(('Report mein model: ' if roman else 'The model named in the report is ')+f['model']['value']+'.',['model'])
    r=_search_answer(pages,question)
    if roman:
        r['answer']='PDF mein ye relevant passages mile hain. Neeche page references ke saath original text hai.' if r['sources'] else 'Is sawal ka matching jawab PDF mein nahi mila. Report ka koi topic, score ya threshold poochhein; main absent information se jawab nahi banaunga.'
    return r
