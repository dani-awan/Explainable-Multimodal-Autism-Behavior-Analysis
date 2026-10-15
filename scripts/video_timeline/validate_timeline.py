"""Synthetic timeline software checks plus trusted frozen-model inference checks."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,tempfile,subprocess
import os
_cache=Path(__file__).resolve().parents[2]/'runtime_cache/numba'
_cache.mkdir(parents=True,exist_ok=True)
os.environ['NUMBA_CACHE_DIR']=str(_cache)
import numpy as np
from timeline_core import plan_windows,inspect_video,extract_segment,analyze,load_bundle,rows,MODEL_SHA,ROOT
import features
def main():
    a=argparse.ArgumentParser();a.add_argument('--run',action='store_true');args=a.parse_args()
    for duration in [.3,1,5,6,11,19.9,120]:
        for length in [5,10,15]:
            spans=plan_windows(duration,length)
            assert spans[0][0]==0 and spans[-1][1]==duration
            assert all(s<e for s,e in spans)
            assert all(spans[j][1]==spans[j+1][0] for j in range(len(spans)-1))
    print('PASS: window planning and coverage')
    if not args.run:print('Inspection only. Use --run for synthetic extraction and frozen inference checks.');return
    if not features.FFMPEG:raise RuntimeError('FFmpeg is required.')
    output=ROOT/'results/video_timeline_validation'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');output.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix='timeline_test_',dir=output) as folder:
        clip=Path(folder)/'synthetic.mp4'
        command=[features.FFMPEG,'-nostdin','-hide_banner','-loglevel','error','-y','-f','lavfi','-i','color=c=blue:s=160x120:r=10:d=5','-f','lavfi','-i','sine=frequency=440:sample_rate=16000:duration=5','-f','lavfi','-i','color=c=red:s=160x120:r=10:d=5','-f','lavfi','-i','sine=frequency=880:sample_rate=16000:duration=5','-filter_complex','[0:v][1:a][2:v][3:a]concat=n=2:v=1:a=1[v][a]','-map','[v]','-map','[a]','-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac','-threads','1',str(clip)]
        run=subprocess.run(command,capture_output=True,text=True,timeout=120)
        if run.returncode:raise RuntimeError(run.stderr)
        meta=inspect_video(clip)
        first,_=extract_segment(clip,0,5,meta,folder);second,_=extract_segment(clip,5,10,meta,folder)
        assert first['visual_blue_mean']>first['visual_red_mean']+.7
        assert second['visual_red_mean']>second['visual_blue_mean']+.7
        assert len(first)==len(second)==58 and np.isfinite(list(first.values())).all() and np.isfinite(list(second.values())).all()
        whole,_=extract_segment(clip,0,meta['duration_seconds'],meta,folder)
        expected,_=features.extract_video(clip)
        difference=max(abs(whole[k]-expected[k]) for k in expected)
        if difference>1e-6:raise ValueError('Full-window extraction differs from original: '+str(difference))
        bundle=load_bundle();report=analyze(clip,bundle,5,temp_root=folder)
        assert len(report['segments'])==2 and len(rows(report))==20
        for s in report['segments']:
            assert all(e['reconstruction']=='PASS' for e in s['explanations'])
        model_path=ROOT/'models/baselines/combined_logistic_regression_tuned.joblib'
        assert hashlib.sha256(model_path.read_bytes()).hexdigest()==MODEL_SHA
    evidence={'status':'PASS','checks':['Window coverage','Known blue/red scene interval mapping','58 finite features per window','Full-window original-extractor agreement','20 frozen-model label outputs','20 explanation reconstructions','Frozen model unchanged'],'full_window_max_feature_difference':difference,'model_sha256':MODEL_SHA,'scope':'Synthetic software and numerical checks only; segment-level predictive accuracy is not established'}
    (output/'validation.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print('PASS: synthetic extraction, frozen inference and explanation checks')
    print('Saved: '+str(output/'validation.json'))
    print('No model training or official test prediction regeneration. Synthetic checks do not establish behavior recognition accuracy.')
if __name__=='__main__':main()
