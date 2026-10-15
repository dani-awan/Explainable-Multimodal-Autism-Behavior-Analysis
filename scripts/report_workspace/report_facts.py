"""Extract unambiguous fields from our exported reports with exact page evidence."""
import re
def facts(pages):
    patterns={
    'research_classification':r'Model classification:\s*(Positive|Negative)\b',
    'model':r'(?m)^Model:\s*([^\n]+)$',
    'threshold':r'(?m)^Frozen threshold:\s*([0-9]+(?:\.[0-9]+)?)\s*$',
    'score_percent':r'Uncalibrated model score\s*\n([0-9]+(?:\.[0-9]+)?)%',
    'threshold_result':r'Threshold output\s*\n(Above threshold|Below threshold)',
    'model_sha256':r'(?m)^Model sha256:\s*([a-fA-F0-9]{64})\s*$',
    'diagnosis':r'Clinical diagnosis\s*\n(Not established)',
    'stage':r'Severity or stage\s*\n(Not assessed)'}
    result={}
    for name,pattern in patterns.items():
        matches=[]
        for pg in pages:
            for m in re.finditer(pattern,pg['text']):matches.append({'value':m.group(1),'page':pg['page'],'text':m.group(0)})
        if matches and len({x['value'] for x in matches})==1:result[name]=matches[0]
    for key,maximum in [('score_percent',100),('threshold',1)]:
        if key in result and not 0<=float(result[key]['value'])<=maximum:result.pop(key)
    tasks=[]
    for pg in pages:
        for task in ['ASD-FEAT prepared features','AV-ASD behavioral labels']:
            if task in pg['text']:tasks.append({'value':task,'page':pg['page'],'text':task})
    if tasks and len({x['value'] for x in tasks})==1:result['task']=tasks[0]
    if result.get('task',{}).get('value')!='ASD-FEAT prepared features':
        # A multi-label table is not one overall autism score.
        for k in ['score_percent','threshold_result','diagnosis','stage','research_classification']:result.pop(k,None)
    return result
def compare(pages_a,pages_b):
    a,b=facts(pages_a),facts(pages_b)
    rows=[{'Field':name.replace('_',' ').capitalize(),'Report A':a.get(name,{}).get('value','Not found uniquely'),'Report B':b.get(name,{}).get('value','Not found uniquely')} for name in ['task','model','threshold','score_percent','threshold_result','model_sha256']]
    required=['task','model','threshold','model_sha256','score_percent']
    compatible=all(k in a and k in b for k in required) and all(a[k]['value']==b[k]['value'] for k in required[:-1])
    delta=round(float(b['score_percent']['value'])-float(a['score_percent']['value']),4) if compatible else None
    return {'rows':rows,'difference_percentage_points':delta,'compatible_fields':compatible,'facts_a':a,'facts_b':b}
