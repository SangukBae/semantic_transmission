#!/usr/bin/env python3
"""Frozen statistics for the event-duration and selective-STA experiment."""
import argparse
from collections import Counter
from pathlib import Path
import time

import numpy as np

from semantic_transmission.artifacts import sha256
from semantic_transmission.automatic_validation import auc
from semantic_transmission.metric_v3_formal import read,save
from semantic_transmission.metric_v4_pipeline import verify,rows,METRICS
from semantic_transmission.sta_selective import predict,CLASSES


def finite(x):return x is not None and np.isfinite(x)

def interval(values):
    if not values:return None
    rng=np.random.default_rng(20260914);x=np.asarray(values,float)
    means=x[rng.integers(len(x),size=(1000,len(x)))].mean(1)
    return np.quantile(means,[.025,.975]).tolist()


def rate(group,metric,threshold):
    valid=[r for r in group if finite(r['scores'].get(metric))]
    detected=[r for r in valid if threshold is not None and r['scores'][metric]>threshold]
    sources=sorted({r['source_id'] for r in group})
    by_source=[sum(r['source_id']==s for r in detected)/sum(r['source_id']==s for r in group) for s in sources]
    return {'cases':len(group),'measured':len(valid),'coverage':len(valid)/len(group) if group else None,
            'detected':len(detected),'rate':len(detected)/len(group) if group else None,'source_95ci':interval(by_source)}


def area(group,control,metric):
    return auc([r['scores'][metric] for r in group if finite(r['scores'].get(metric))],
               [r['scores'][metric] for r in control if finite(r['scores'].get(metric))])


def detection(group,control,metric,threshold):
    a=area(group,control,metric);p=rate(group,metric,threshold);n=rate(control,metric,threshold)
    return {'auc':a,'positive':p,'normal':n,'threshold':threshold,
            'detection_gate':bool(a is not None and a>=.9 and p['rate']>=.8 and n['rate']<=.1 and min(p['coverage'],n['coverage'])>=.95)}


def paired_auc(group,controls,candidate,baseline):
    group=[r for r in group if all(finite(r['scores'].get(n)) for n in (candidate,baseline))]
    controls=[r for r in controls if all(finite(r['scores'].get(n)) for n in (candidate,baseline))]
    differences=[]
    for source in sorted({r['source_id'] for r in group}):
        p=[r for r in group if r['source_id']==source];n=[r for r in controls if r['source_id']==source]
        a,b=area(p,n,candidate),area(p,n,baseline)
        if a is not None and b is not None:differences.append(a-b)
    return {'paired_sources':len(differences),'mean_source_AUC_difference':float(np.mean(differences)) if differences else None,
            'source_95ci':interval(differences)}


def target_agreement(group,target):
    measured=[];all_rows=[]
    for r in group:
        candidates=[e for e in r['scores']['external_audit']['source_events'] if e['object_id']==r['target_object']
                    and e['type']==r['target_event'] and abs(e['time_s']-r['target_frame']/8.)<1e-8]
        assert len(candidates)==1
        observed=candidates[0]['observed'];key='ghost_auc' if target=='ghost' else 'delay_penalty'
        value=observed.get(key) if observed else None
        item={'case_id':r['case_id'],'source_id':r['source_id'],'kind':r['kind'],'severity':r['severity'],
              'oracle':r['oracle'],'observed':observed,'measured':finite(value)}
        all_rows.append(item)
        if finite(value):measured.append(item)
    errors=[abs(r['observed'][key]-r['oracle'][key]) for r in measured]
    by_source=[np.mean([abs(r['observed'][key]-r['oracle'][key]) for r in measured if r['source_id']==s])
               for s in sorted({r['source_id'] for r in measured})]
    result={'cases':len(group),'measured':len(measured),'target_event_coverage':len(measured)/len(group),
            'mean_absolute_error':float(np.mean(errors)) if errors else None,'MAE_source_95ci':interval(by_source),
            'rows':all_rows,'scope':'External target-event and true-identity correspondence; coverage includes missed source events'}
    if target=='delay':
        observable_delays=[r for r in measured if r['oracle']['delay_s'] is not None and r['observed']['delay_s'] is not None]
        delay_errors=[abs(r['observed']['delay_s']-r['oracle']['delay_s']) for r in observable_delays]
        tp=fp=fn=0
        for r in all_rows:
            actual=r['oracle']['delay_status']=='deadline_miss'
            predicted=r['observed'] is not None and r['observed']['delay_status']=='deadline_miss'
            tp+=actual and predicted;fp+=not actual and predicted;fn+=actual and not predicted
        result.update(observed_delay_pairs=len(delay_errors),observed_delay_MAE_s=float(np.mean(delay_errors)) if delay_errors else None,
                      deadline_miss={'tp':tp,'fp':fp,'fn':fn,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None})
    return result


def classification(rows,predictions):
    def summarize(subset):
        recalls={c:sum(p['prediction']==c for r,p in subset if r['label']==c)/sum(r['label']==c for r,p in subset)
                 for c in CLASSES if any(r['label']==c for r,p in subset)}
        valid=[(r,p) for r,p in subset if p['prediction']!='abstain']
        normal=[p for r,p in subset if r['label']=='normal']
        return {'class_recall':recalls,'macro_accuracy':float(np.mean(list(recalls.values()))) if recalls else None,
                'fault_macro_accuracy':float(np.mean([v for k,v in recalls.items() if k!='normal'])) if len(recalls)>1 else None,
                'cases':len(subset),'coverage':len(valid)/len(subset) if subset else None,
                'accuracy_measured_only':sum(p['prediction']==r['label'] for r,p in valid)/len(valid) if valid else None,
                'normal_false_alarm_rate':sum(p['prediction'] not in ('normal','abstain') for p in normal)/len(normal) if normal else None,
                'normal_abstention_rate':sum(p['prediction']=='abstain' for p in normal)/len(normal) if normal else None,
                'confusion':{c:dict(Counter(p['prediction'] for r,p in subset if r['label']==c)) for c in CLASSES}}
    pairs=list(zip(rows,predictions));result=summarize(pairs)
    # Eligibility can remove different classes from different source videos.
    # Resample whole source confusion matrices and recompute pooled class recall.
    sources=sorted({r['source_id'] for r in rows});matrix=np.zeros((len(sources),5,6))
    for r,p in pairs:
        matrix[sources.index(r['source_id']),CLASSES.index(r['label']),
               CLASSES.index(p['prediction']) if p['prediction'] in CLASSES else 5]+=1
    rng=np.random.default_rng(20260914)
    drawn=matrix[rng.integers(len(sources),size=(1000,len(sources)))].sum(1)
    denominator=drawn.sum(2);diagonal=np.diagonal(drawn[:,:,:5],axis1=1,axis2=2)
    recall=np.divide(diagonal,denominator,out=np.full_like(diagonal,np.nan),where=denominator>0)
    samples={'macro_accuracy':np.nanmean(recall,axis=1),'fault_macro_accuracy':np.nanmean(recall[:,1:],axis=1),
             'coverage':1-drawn[:,:,5].sum(1)/drawn.sum((1,2)),
             'normal_false_alarm_rate':drawn[:,0,1:5].sum(1)/denominator[:,0],
             'normal_abstention_rate':drawn[:,0,5]/denominator[:,0]}
    for name,values in samples.items():
        result[name+'_source_95ci']=np.nanquantile(values,[.025,.975]).tolist()
    result['source_reports']={s:summarize([(r,p) for r,p in pairs if r['source_id']==s]) for s in sorted({r['source_id'] for r in rows})}
    result['gate']=bool(result['macro_accuracy']>=.8 and result['fault_macro_accuracy']>=.8 and result['coverage']>=.95
                        and result['normal_false_alarm_rate']<=.1)
    result['rejection_reasons']=dict(Counter(p['reason'] for p in predictions if p['reason']))
    result['bootstrap_scope']='Source-cluster resampling with pooled class recall recomputed; per-source descriptive means can differ'
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--declare',action='store_true');args=ap.parse_args()
    root=args.output;verify(root);declaration=root/'statistics_declaration.json'
    if args.declare:
        if declaration.exists() or (root/'scores').exists():raise ValueError('Declare before scores')
        save(declaration,{'created_unix':time.time(),'script_sha256':sha256(Path(__file__)),'heldout_scores_observed':0});return
    verify(root,True)
    if read(declaration)['script_sha256']!=sha256(Path(__file__)):raise ValueError('Statistics changed')
    calibration=read(root/'calibration.json');merged=[]
    for row in rows(root/'cases.jsonl'):
        scored={}
        for part in ('visual','pixel'):
            record=read(root/'scores'/part/(row['case_id']+'.json'))
            assert record['calibration_sha256']==sha256(root/'calibration.json')
            scored.update(record['scores'])
        merged.append({**row,'scores':scored})
    events=[r for r in merged if r['corpus']=='event'];controls=[r for r in events if r['target']=='control']
    results={}
    for target,candidate in (('ghost','ghost_max'),('delay','delay_max')):
        group=[r for r in events if r['target']==target]
        stats={n:detection(group,controls,n,calibration['thresholds'][n]) for n in METRICS}
        individual={kind:detection([r for r in group if r['kind']==kind],controls,candidate,calibration['thresholds'][candidate]) for kind in sorted({r['kind'] for r in group})}
        numerical=target_agreement(group,target)
        agreement=(numerical['mean_absolute_error'] is not None and numerical['mean_absolute_error']<=.25 if target=='ghost' else
                   numerical['observed_delay_MAE_s'] is not None and numerical['observed_delay_MAE_s']<=.25 and numerical['deadline_miss']['f1'] is not None and numerical['deadline_miss']['f1']>=.8)
        results[target]={'candidate':candidate,'metrics':stats,'by_kind':individual,'numerical_agreement':numerical,
                        'by_family':{family:detection([r for r in group if r['family']==family],[r for r in controls if r['family']==family],candidate,calibration['thresholds'][candidate]) for family in sorted({r['family'] for r in group})},
                        'short':detection([r for r in group if r['severity']==.25],controls,candidate,calibration['thresholds'][candidate]),
                        'paired_comparisons':{n:paired_auc(group,controls,candidate,n) for n in METRICS if n not in ('ghost_max','delay_max')},
                        'adoption_gate':bool(all(v['detection_gate'] for v in individual.values()) and numerical['target_event_coverage']>=.8 and agreement)}
    selected=[r for r in merged if r['corpus']=='sta' and r['eligible']]
    classifiers={};prediction_rows=[]
    for key in ('sta','stage'):
        m=calibration['models'][key];predictions=[predict(r['scores'][key],m['fit'],m['rejection'],r['scores'].get('observable',True)) for r in selected]
        classifiers[key]=classification(selected,predictions)
        for row,pred in zip(selected,predictions):prediction_rows.append({'case_id':row['case_id'],'source_id':row['source_id'],'label':row['label'],'classifier':key,**pred})
        classifiers[key+'_without_rejection']=classification(selected,[{**p,'prediction':p['raw_prediction'] or 'abstain','reason':None} for p in predictions])
    technically_missing=[r for r in merged if r['corpus']=='unobservable']
    m=calibration['models']['sta']
    technical=[predict(r['scores']['sta'],m['fit'],m['rejection'],False) for r in technically_missing]
    classifiers['technical_fixture']={'cases':len(technical),'abstained':sum(p['prediction']=='abstain' for p in technical),
        'scope':'missing-input routing fixtures; not evidence of semantic uncertainty detection'}
    # Paired classifier comparison uses each video's class-balanced accuracy.
    paired=[classifiers['sta']['source_reports'][s]['macro_accuracy']-classifiers['stage']['source_reports'][s]['macro_accuracy'] for s in classifiers['sta']['source_reports']]
    classifiers['same_input_paired_difference']={'mean_source_macro_accuracy_difference':float(np.mean(paired)),'source_95ci':interval(paired)}
    counts=Counter()
    for row in events:
        for key,value in row['scores']['source_paired_additions'].items():
            if isinstance(value,int):counts[key]+=value
    source_pairing=dict(counts)
    source_pairing.update(raw_h_add=counts['raw_additions']/counts['raw_pairable_opportunities'] if counts['raw_pairable_opportunities'] else None,
                         source_paired_h_add=counts['paired_additions']/counts['paired_opportunities'] if counts['paired_opportunities'] else None,
                         source_present_recall=counts['source_detected_present']/counts['present_opportunities'] if counts['present_opportunities'] else None,
                         scope='Repeated case opportunities, known source entities only; counts are not independent videos; this is a diagnostic using the same extractor')
    report={'status':'COMPLETED','completed_unix':time.time(),'protocol_sha256':sha256(root/'protocol.json'),
            'calibration_sha256':sha256(root/'calibration.json'),'event':results,'STA':classifiers,
            'STA_predictions':prediction_rows,'source_paired_additions':source_pairing,
            'case_counts':dict(Counter(r['corpus'] for r in merged)),'new_human_review':False,'novelty_demonstrated':False}
    save(root/'summary.json',report)
    print({k:v['adoption_gate'] for k,v in results.items()},'STA',classifiers['sta']['gate'],flush=True)


if __name__=='__main__':main()
