"""Exploratory fixed-window inference using the existing frozen AV-ASD baseline."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,math,sys,tempfile,subprocess,threading
import cv2
import numpy as np
import pandas as pd
import joblib

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/video_upload'))
import features
from report_details import explain_features
MODEL_SHA='2a2636966be5f977367569a73db72c7dbdbd90f2e05e7240d014ee443de39e3e'
LOCK=threading.RLock()
LIMIT='Exploratory fixed-window model scores, not confirmed behaviors or event onset times. The classifier was trained on clips; segment-level predictive accuracy has not been established. Scores are not autism diagnosis, severity or clinical probability.'
def plan_windows(duration,window_seconds=10):
    duration=float(duration);window_seconds=float(window_seconds)
    if not math.isfinite(duration) or not 0<duration<=120:raise ValueError('Duration must be within 0 to 120 seconds.')
    if window_seconds not in [5,10,15]:raise ValueError('Choose a 5, 10 or 15 second window.')
    points=list(np.arange(0,duration,window_seconds));spans=[(float(s),min(float(s+window_seconds),duration)) for s in points]
    if len(spans)>1 and spans[-1][1]-spans[-1][0]<2:
        spans[-2]=(spans[-2][0],duration);spans.pop()
    return spans
def inspect_video(path):
    path=Path(path)
    if not path.is_file() or path.suffix.lower()!='.mp4' or path.stat().st_size>200*1024**2:raise ValueError('Use an MP4 file up to 200 MiB.')
    cap=cv2.VideoCapture(str(path))
    try:
        fps=float(cap.get(cv2.CAP_PROP_FPS));count=float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or not np.isfinite([fps,count]).all() or min(fps,count)<=0:raise ValueError('Invalid video metadata.')
        if count>7200 or count/fps>120:raise ValueError('Video limit: 120 seconds and 7200 frames.')
        return {'fps':fps,'frame_count':int(count),'duration_seconds':count/fps}
    finally:cap.release()
def load_bundle(root=ROOT):
    path=Path(root)/'models/baselines/combined_logistic_regression_tuned.joblib'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=MODEL_SHA:raise ValueError('Frozen AV-ASD model checksum differs. Stop and review the artifact.')
    bundle=joblib.load(path)
    if len(bundle['feature_columns'])!=58 or len(bundle['target_columns'])!=10 or float(bundle['threshold'])!=0.3:raise ValueError('Unexpected frozen model schema or threshold.')
    return bundle
def visual_segment(path,start,end,meta):
    fps=meta['fps'];count=meta['frame_count'];first=max(0,int(round(start*fps)));last=min(count-1,int(math.ceil(end*fps))-1)
    targets=np.unique(np.rint((start+np.arange(0,end-start,1.0))*fps).astype(int));targets=targets[(targets>=first)&(targets<=last)]
    fallback=len(targets)<2
    if fallback:targets=np.unique([first,last])
    if len(targets)<2:raise ValueError('Segment contains fewer than two usable frames.')
    cap=cv2.VideoCapture(str(path));rows=[];changes=[];previous=None
    try:
        for frame_no in targets:
            if not cap.set(cv2.CAP_PROP_POS_FRAMES,int(frame_no)):raise ValueError('Frame seeking failed for a timeline segment.')
            ok,frame=cap.read()
            if not ok or frame is None:raise ValueError('Cannot decode a timeline frame.')
            h,w=frame.shape[:2];scale=224/max(h,w);resized=cv2.resize(frame,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_AREA)
            rgb=cv2.cvtColor(resized,cv2.COLOR_BGR2RGB);gray=cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY);g=gray.astype(np.float32)/255.0
            rows.append([*rgb.mean(axis=(0,1))/255.0,float(g.mean()),float(g.std()),float((cv2.Canny(gray,100,200)>0).mean())])
            if previous is not None:changes.append(float(np.abs(g-previous).mean()))
            previous=g
    finally:cap.release()
    values=np.asarray(rows);result={}
    for j,name in enumerate(['red','green','blue','brightness','contrast','edge_density']):
        result['visual_'+name+'_mean']=float(values[:,j].mean());result['visual_'+name+'_std']=float(values[:,j].std())
    result['visual_temporal_change_mean']=float(np.mean(changes));result['visual_temporal_change_std']=float(np.std(changes))
    return result,{'sampled_frames':len(targets),'short_segment_fallback':fallback,'first_sample_frame':int(targets[0]),'last_sample_frame':int(targets[-1])}
def extract_segment(path,start,end,meta=None,temp_root=None):
    path=Path(path).resolve();meta=meta or inspect_video(path)
    if not (0<=start<end<=meta['duration_seconds']+1e-6):raise ValueError('Invalid segment interval.')
    temp_root=Path(temp_root or ROOT/'results/video_timeline_temp');temp_root.mkdir(parents=True,exist_ok=True)
    with LOCK:
        visual,quality=visual_segment(path,start,end,meta)
        if not features.FFMPEG:raise RuntimeError('FFmpeg is not on PATH.')
        with tempfile.TemporaryDirectory(prefix='segment_',dir=temp_root) as folder:
            cropped=Path(folder)/'segment.wav';normalized=Path(folder)/'features.wav'
            command=[features.FFMPEG,'-nostdin','-hide_banner','-loglevel','error','-y','-threads','1','-i',str(path),'-ss',f'{start:.9f}','-t',f'{end-start:.9f}','-map','0:a:0','-vn','-ac','1','-ar','16000','-c:a','pcm_s16le',str(cropped)]
            if start==0 and abs(end-meta['duration_seconds'])<1e-6:
                audio,silent=features.extract_audio(path,normalized)
            else:
                run=subprocess.run(command,capture_output=True,text=True,timeout=120)
                if run.returncode:raise ValueError('Audio extraction failed; a usable audio track is required. '+run.stderr[-250:])
                audio,silent=features.extract_audio(cropped,normalized)
        result={**audio,**visual}
        if len(result)!=58 or not np.isfinite(list(result.values())).all():raise ValueError('Expected 58 finite segment features.')
        quality.update({'duration_seconds':end-start,'all_zero_audio':silent,'decoder':'OpenCV source-frame seeking; FFmpeg PCM audio trim'})
        return result,quality
def analyze(path,bundle,window_seconds=10,progress=None,temp_root=None):
    meta=inspect_video(path);spans=plan_windows(meta['duration_seconds'],window_seconds);segments=[]
    for i,(start,end) in enumerate(spans):
        values,quality=extract_segment(path,start,end,meta,temp_root)
        if set(values)!=set(bundle['feature_columns']):raise ValueError('Segment feature schema differs from the frozen model.')
        frame=pd.DataFrame([values],columns=bundle['feature_columns'])
        probabilities=np.asarray(bundle['model'].predict_proba(frame),dtype=float)
        if probabilities.shape!=(1,10) or not np.isfinite(probabilities).all() or ((probabilities<0)|(probabilities>1)).any():raise ValueError('Invalid segment model scores.')
        labels=[{'label':str(name),'model_score':float(p),'threshold_result':'Above threshold' if p>=float(bundle['threshold']) else 'Below threshold'} for name,p in zip(bundle['target_columns'],probabilities[0])]
        segments.append({'segment_index':i,'start_seconds':start,'end_seconds':end,'labels':labels,'quality':quality,'explanations':explain_features(bundle,frame)})
        if progress:progress(i+1,len(spans))
    return {'generated_at_utc':datetime.now(timezone.utc).isoformat(),'model':'AV-ASD combined Logistic Regression','model_sha256':MODEL_SHA,'video_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'threshold':float(bundle['threshold']),'requested_window_seconds':window_seconds,'duration_seconds':meta['duration_seconds'],'segments':segments,'timing':'Nominal frame-based windows; approximate playback timing, not annotated event boundaries. Last remainder below 2 seconds is merged into the preceding window.','limitations':LIMIT}
def rows(report):
    return [{'segment_index':s['segment_index'],'start_seconds':s['start_seconds'],'end_seconds':s['end_seconds'],'label':r['label'],'model_score':r['model_score'],'threshold_result':r['threshold_result']} for s in report['segments'] for r in s['labels']]
def selected_report(report,index):
    s=report['segments'][index]
    return {'generated_at_utc':report['generated_at_utc'],'model':report['model'],'model_sha256':report['model_sha256'],'video_sha256':report['video_sha256'],'threshold':report['threshold'],'features':58,'quality':dict(s['quality'],segment_start_seconds=s['start_seconds'],segment_end_seconds=s['end_seconds']),'labels':s['labels'],'explanations':s['explanations'],'summary':f"Exploratory segment from {s['start_seconds']:.2f} to {s['end_seconds']:.2f} seconds. These scores concern this window, not the complete recording.",'limitations':LIMIT}
