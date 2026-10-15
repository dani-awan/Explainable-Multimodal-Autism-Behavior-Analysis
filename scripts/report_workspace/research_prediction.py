"""Interpret an existing frozen ASD-FEAT score; never infer from PDF prose."""
import math
MODEL_SHA='2c8e9d9c126e9a7a8134eb9792550254de3747847b96aeaca1e396bae57327d4'
GUIDANCE='If there are concerns about development or behavior, contact a pediatrician or qualified autism specialist for a developmental assessment, regardless of this model result.'
LIMIT='Experimental research classification, not a clinical screening tool or diagnosis. Positive does not confirm autism; Negative does not rule it out. The score is not a validated percentage chance of autism.'
def prediction(report):
    if 'labels' in report or report.get('model')!='rgb_behavior_pca32_lr':return None
    if report.get('model_sha256')!=MODEL_SHA:return None
    if report.get('input_features')!=1036:return None
    score=report.get('model_score'); threshold=report.get('threshold')
    if isinstance(score,bool) or isinstance(threshold,bool):raise ValueError('Invalid score or threshold.')
    score=float(score);threshold=float(threshold)
    if not math.isfinite(score) or not 0<=score<=1 or threshold!=0.5:raise ValueError('Invalid frozen-model score or threshold.')
    return {'classification':'Positive' if score>=threshold else 'Negative','model_score':score,'threshold':threshold,'endpoint':'ASD-FEAT current-visit research annotation','scope':'One compatible recording; not a diagnosis of the child','interpretation':LIMIT,'next_step':GUIDANCE,'evidence_limit':'All 12 positive official-test recordings came from one participant. Population-level clinical reliability is not established.','provenance':'Interpretation of supplied score fields; PDF or JSON content alone cannot authenticate inference.'}
