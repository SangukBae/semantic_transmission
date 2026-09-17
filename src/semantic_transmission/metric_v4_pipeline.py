"""Reproducible duration/delay and selective-STA evaluation; fresh scene families."""
import argparse
from collections import Counter, OrderedDict
import gzip
import json
from pathlib import Path
import pickle
import runpy
import shutil
import time

import numpy as np

from .artifacts import sha256
from .metric_v3_features import VisualObserver, PixelObserver, pixels
from .metric_v3_formal import array,read,save
from .metric_v3_cases import annotation_events, truth_event_change
from .event_duration_metric import PARAMETERS,evaluate_observations,sfr_observed
from .sta_video_validation import simulate,encode_video,score_video_attribution,LABEL
from .sta_selective import fit,calibrate as fit_rejection,predict,CLASSES
from .metric_v4_cases import FAMILIES,scene,draw,variants,truth_query
from .metric_v4_audit import audit_events,source_paired_additions

OLD=Path('outputs/ere_sta_formal_20260914_v1')
CODES=('event_duration_metric.py','sta_selective.py','metric_v4_cases.py','metric_v4_audit.py','metric_v4_pipeline.py',
       'event_metric.py','object_metric.py','motion_metric.py','metric_v3_features.py','sta_video_validation.py',
       'attribution_metric.py','automatic_metrics.py','temporal_baselines.py','tracking_baselines.py',
       'metric_v3_cases.py','metric_v2_cases.py')
METRICS=('ghost_max','delay_max','sfr_inferred','ere','ser','tlp_alex','mte_tail','idf1_mask_error','lpips_alex','tof_farneback')
STAGE_NAMES=tuple(s+'_'+m for s in ('source_tx','tx_rx','rx_reconstruction') for m in ('mse','mae'))


def stage_distances(a,tx,rx,b):
    out={}
    for stage,x,y in (('source_tx',a,tx),('tx_rx',tx,rx),('rx_reconstruction',rx,b)):
        delta=(x.astype(np.float32)-y.astype(np.float32))/255.
        out.update({stage+'_mse':float(np.square(delta).mean()),stage+'_mae':float(np.abs(delta).mean())})
    return out


def rows(path):return [json.loads(line) for line in path.read_text().splitlines()]

def verify(root,calibrated=False):
    p=read(root/'protocol.json')
    for name,digest in p['code_sha256'].items():
        if sha256(Path('src/semantic_transmission')/name)!=digest:raise ValueError('Frozen code changed: '+name)
    if sha256(OLD/'protocol.json')!=p['old_protocol_sha256']:raise ValueError('Old experiment changed')
    if calibrated:
        if sha256(root/'calibration.json')!=(root/'calibration.sha256').read_text().strip():raise ValueError('Calibration changed')
    return p


def enable_memoization():
    operation=Path('scripts/run_metric_v3_memoized.py')
    if sha256(operation)!=read(OLD/'memoization_probe.json')['script_sha256']:raise ValueError('Unverified optimization')
    runpy.run_path(str(operation),run_name='memoization_helpers')['enable']()


class CacheOnly(VisualObserver):
    def __init__(self):
        self.root=OLD/'visual_cache'/read(OLD/'protocol.json')['signature'];self.recent=OrderedDict()
    def observe(self,frames):
        if not (self.root/(pixels(frames)+'.pkl.gz')).exists():raise FileNotFoundError('Missing old RGB cache')
        return super().observe(frames)


def prepare(root):
    if root.exists():raise FileExistsError('Use a fresh output directory: '+str(root))
    old=read(OLD/'protocol.json');root.mkdir(parents=True)
    p={'schema':'duration_delay_sta_v1','created_unix':time.time(),'old_run':str(OLD),
       'old_protocol_sha256':sha256(OLD/'protocol.json'),'parameters':PARAMETERS,
       'code_sha256':{name:sha256(Path('src/semantic_transmission')/name) for name in CODES},
       'old_calibration_is_development_only':True,'new_heldout_families':FAMILIES,'heldout_sources_per_family':8,
       'metrics':METRICS,'primary_candidates':['ghost_max','delay_max'],
       'threshold':'old rendered normal observations 95th percentile higher; strictly greater',
       'primary_gate':{'auc':.9,'tpr':.8,'fpr':.1,'coverage':.95,'target_event_coverage':.8,
                       'ghost_auc_mean_absolute_error':.25,'delay_seconds_mean_absolute_error':.25,'deadline_miss_f1':.8},
       'STA_gate':{'macro_accuracy_abstain_as_wrong':.8,'normal_false_alarm':.1,'coverage':.95},
       'STA_classifier':'five standardized nearest source-balanced centroids; previous development16 train, previous heldout32 calibrate rejection',
       'STA_rejection':'per-true-class development-calibration distance 95th higher; correct margin 5th lower; nonfinite/unreadable -> abstain',
       'bootstrap':{'unit':'source','iterations':1000,'seed':20260914},
       'new_human_review':False,'memoization_script_sha256':sha256(Path('scripts/run_metric_v3_memoized.py')),
       'evaluation_scope':'controlled new procedural scene families; not natural-video validity or actual LGVSC causal validation',
       'future_heldout_scores_observed':0}
    save(root/'protocol.json',p)
    (root/'frozen_source').mkdir()
    for name in CODES:shutil.copy2(Path('src/semantic_transmission')/name,root/'frozen_source'/name)
    shutil.copy2('docs/EVENT_DURATION_STA_PROTOCOL.md',root/'PROTOCOL.frozen.md')
    new=[]; sources=[]; truth_checks=[]
    for fi,family in enumerate(FAMILIES):
        for k in range(8):
            seed=260914700+fi*100+k; sid=family+'/'+str(seed);stem=sid.replace('/','__')
            state=scene(seed,family);a,at=draw(state);folder=root/'sources'/stem;folder.mkdir(parents=True)
            np.savez_compressed(folder/'source.npz',frames=a)
            np.savez_compressed(folder/'truth.npz',**at)
            events=annotation_events(at['centers'],at['visible'])
            source={'source_id':sid,'source_stem':stem,'family':family,'source_pixel_sha256':pixels(a),'events':events}
            sources.append(source)
            def add(case_id,b,bt,metadata,tx=None,rx=None,status=None,packets=None):
                dest=root/'cases'/case_id;dest.mkdir(parents=True)
                np.savez_compressed(dest/'reconstruction.npz',frames=b)
                if bt is not None:np.savez_compressed(dest/'truth.npz',**bt)
                record={**source,'case_id':case_id,'split':'heldout','reconstruction_pixel_sha256':pixels(b),**metadata}
                if tx is not None:
                    for name,value in (('tx',tx),('rx',rx)):np.savez_compressed(dest/(name+'.npz'),frames=value);record[name+'_pixel_sha256']=pixels(value)
                    save(dest/'transport.json',status);record['transport_sha256']=sha256(dest/'transport.json')
                    with (dest/'packets.pkl').open('wb') as f:pickle.dump(packets,f,protocol=5)
                    record['packets_sha256']=sha256(dest/'packets.pkl')
                new.append(record)
            controls=[]
            for name,b,bt,meta in variants(state):
                cid=stem+'__'+name
                if meta['target']=='control':
                    assert np.array_equal(at['visible'],bt['visible']) and np.array_equal(at['centers'],bt['centers'])
                    oracle={};controls.append((name,b,bt))
                else:
                    oracle=truth_query(at,bt,meta['target_object'],meta['target_event'],meta['target_frame'])
                    assert any(e['object_id']==meta['target_object'] and e['type']==meta['target_event'] and e['time_s']==meta['target_frame']/8 for e in events)
                    if meta['kind']=='event_delay':assert oracle['delay_s']==meta['severity']
                    if meta['kind']=='never_update':assert oracle['delay_status']=='deadline_miss'
                    if meta['target']=='ghost':assert oracle['ghost_auc']>0
                add(cid,b,bt,{'corpus':'event',**meta,'oracle':oracle,'variant':name})
                truth_checks.append({'case_id':cid,'status':'PASSED','oracle':oracle})
            encoded=encode_video(a)
            for name,b,bt in controls:
                add(stem+'__sta_control_'+name,b,bt,{'corpus':'sta','kind':'control','label':'normal','severity':0.,'eligible':True,'variant':name},
                    a,a,{'tx_status':['ok']*len(a),'rx_status':['ok']*len(a)}, {'tx_packets':encoded,'rx_packets':encoded})
            for kind in ('tx_omission','packet_loss','bit_corruption','generation_freeze','unsupported_addition'):
                for severity in (.125,.25):
                    sim=simulate(a,kind,severity,encoded);indices=sim['reconstruction_indices']
                    bt={key:value[indices] for key,value in at.items()}
                    changed=kind=='unsupported_addition' or truth_event_change(events,annotation_events(bt['centers'],bt['visible']))
                    add(stem+f'__sta_{kind}_{severity}',sim['reconstruction'],bt,{'corpus':'sta','kind':kind,'label':LABEL[kind],
                        'severity':severity,'eligible':bool(changed),'variant':kind,'truth_event_changed':bool(changed)},
                        sim['tx_frames'],sim['rx_frames'],{k:sim[k] for k in ('tx_status','rx_status','tx_indices','rx_indices','reconstruction_indices')},
                        {k:sim[k] for k in ('tx_packets','rx_packets')})
            # No readable representation or missing source media must never be called normal.
            for missing in ('no_received_keyframes','missing_source_media'):
                new.append({**source,'case_id':stem+'__'+missing,'split':'heldout','corpus':'unobservable',
                            'kind':missing,'label':'unobservable','eligible':True})
            print('prepared',family,k+1,'/ 8',len(new),flush=True)
    old_sources={r['source_pixel_sha256'] for r in rows(OLD/'cases.jsonl')}
    assert not old_sources & {r['source_pixel_sha256'] for r in sources}
    (root/'cases.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in new))
    save(root/'sources.json',sources);save(root/'truth_audit.json',{'status':'PASSED','results':truth_checks})
    save(root/'preparation.json',{'sources':len(sources),'cases':dict(Counter(r['corpus'] for r in new)),
        'STA_eligibility':dict(Counter(str(r.get('eligible')) for r in new if r['corpus']=='sta')),
        'source_RGB_overlap_with_old':False,'manifest_sha256':sha256(root/'cases.jsonl'),
        'RGB_and_truth_file_sha256':{str(f.relative_to(root)):sha256(f) for f in sorted(root.glob('**/*.npz'))}})


def development(root):
    verify(root)
    if (root/'calibration.json').exists() or (root/'scores').exists():raise ValueError('Development is locked')
    enable_memoization();observer=CacheOnly();old_p=read(OLD/'protocol.json');base=Path(old_p['base_run'])
    samples=[];sta=[];stage=[]
    for old_split in ('development','heldout'):
        normal={r['case_id']:r for r in read(OLD/('sta_semantic_controls_'+old_split+'.json'))['rows']}
        for index,row in enumerate(normal.values()):
            output=root/'development_event'/(row['case_id']+'.json')
            a=array(base/'inputs'/(row['source_stem']+'.npz'),row['source_pixel_sha256'])
            b=array(base/'cases'/(row['case_id']+'.npz'),row['reconstruction_pixel_sha256'])
            if output.exists():value=read(output)
            else:
                ao,bo=observer.observe(a),observer.observe(b)
                result=evaluate_observations(ao,bo)
                result['sfr_inferred']=sfr_observed(ao,bo,result['entity_groups'])
                old_visual=read(OLD/'scores/visual/ere'/(row['case_id']+'.json'))['scores']
                old_pixel=read(OLD/'scores/pixel/ere'/(row['case_id']+'.json'))['scores']
                value={**row,'scores':{**old_visual,**old_pixel,**result}}
                save(output,value)
            samples.append(value)
            common={k:row[k] for k in ('case_id','source_id','split')};common['label']='normal'
            sta.append({**common,'scores':row['scores']})
            stage.append({**common,'scores':stage_distances(a,a,a,b)})
            if (index+1)%20==0:print('development',old_split,index+1,'/',len(normal),flush=True)
        for row in rows(OLD/'sta_cases.jsonl'):
            if row['split']!=old_split or row['label']=='normal' or not row['eligible']:continue
            value=read(OLD/'scores/visual/sta'/(row['case_id']+'.json'))
            sta.append({**row,'scores':{k:value['scores']['sta'][k] for k in CLASSES if k!='normal'}})
        stage.extend(r for r in read(OLD/('same_input_baseline_'+old_split+'.json'))['rows'] if r['label']!='normal')
    thresholds={}
    for name in METRICS:
        vals=[r['scores'].get(name) for r in samples]
        vals=[v for v in vals if v is not None and np.isfinite(v)]
        thresholds[name]=float(np.quantile(vals,.95,method='higher')) if vals else None
    models={}
    for key,data,names in (('sta',sta,('edr','clr','gfr','ghr')),('stage',stage,STAGE_NAMES)):
        valid=[r for r in data if all(r['scores'].get(n) is not None and np.isfinite(r['scores'][n]) for n in names)]
        training=[r for r in valid if r['split']=='development'];cal=[r for r in valid if r['split']=='heldout']
        fitted=fit(training,names);rejection=fit_rejection(cal,fitted)
        models[key]={'fit':fitted,'rejection':rejection,'training_cases':len(training),'calibration_cases':len(cal),
                     'unmeasurable_development_cases':len(data)-len(valid)}
        save(root/('development_'+key+'.json'),data)
    save(root/'calibration.json',{'created_unix':time.time(),'protocol_sha256':sha256(root/'protocol.json'),
        'thresholds':thresholds,'models':models,'normal_cases':len(samples),'new_heldout_scores_observed':0,
        'scope':'All former heldout data is development now; fit16/calibrate32 old sources, fresh scene families reserved'})
    (root/'calibration.sha256').write_text(sha256(root/'calibration.json')+'\n')
    print('FROZEN',thresholds,flush=True)


def run(root,part):
    p=verify(root,True);prep=read(root/'preparation.json')
    if sha256(root/'cases.jsonl')!=prep['manifest_sha256']:raise ValueError('Changed manifest')
    import torch,cv2
    torch.set_num_threads(4);torch.manual_seed(20260914);cv2.setNumThreads(1)
    enable_memoization();old_p=read(OLD/'protocol.json');base=read(Path(old_p['base_run'])/'protocol.json')
    observer=VisualObserver(base,root,old_p['signature']) if part=='visual' else PixelObserver()
    save(root/('runtime_'+part+'.json'),{'started_unix':time.time(),'torch':torch.__version__,'numpy':np.__version__,
         'GPU':torch.cuda.get_device_name(),'protocol_sha256':sha256(root/'protocol.json'),'part':part})
    cache={}; manifest=rows(root/'cases.jsonl')
    for index,row in enumerate(manifest):
        dest=root/'scores'/part/(row['case_id']+'.json')
        if dest.exists():
            prior=read(dest)
            if prior['calibration_sha256']!=sha256(root/'calibration.json'):raise ValueError('Resume calibration differs')
            continue
        started=time.time()
        if row['corpus']=='unobservable':
            result={'observable':False,'reason':row['kind'],'sta':{k:None for k in CLASSES if k!='normal'},'stage':{n:None for n in STAGE_NAMES}}
        else:
            folder=root/'cases'/row['case_id'];src=root/'sources'/row['source_stem']
            a=array(src/'source.npz',row['source_pixel_sha256']);b=array(folder/'reconstruction.npz',row['reconstruction_pixel_sha256'])
            key=pixels(a)+'_'+pixels(b)
            # Independent case variants can share observable RGB; scores reuse only RGB hashes.
            if key not in cache:
                pair=observer.compare(a,b)
                if part=='visual':
                    ao,bo=observer.observe(a),observer.observe(b)
                    metric=evaluate_observations(ao,bo)
                    pair.update(metric);pair['sfr_inferred']=sfr_observed(ao,bo,metric['entity_groups'])
                    with np.load(src/'truth.npz') as f:truth={k:f[k] for k in f.files}
                    pair['external_audit']=audit_events(ao,metric,truth)
                    pair['source_paired_additions']=source_paired_additions(ao,bo,metric,truth)
                cache[key]=pair
            result=dict(cache[key])
            if row['corpus']=='sta':
                tx=array(folder/'tx.npz',row['tx_pixel_sha256']);rx=array(folder/'rx.npz',row['rx_pixel_sha256'])
                status=read(folder/'transport.json')
                if sha256(folder/'transport.json')!=row['transport_sha256']:raise ValueError('Transport changed')
                if part=='visual':
                    result['sta']=score_video_attribution(observer,a,b,tx,rx,status['tx_status'],status['rx_status'])
                    result['observable']=bool(result['sta']['units_reference'])
                else:result['stage']=stage_distances(a,tx,rx,b)
            if len(cache)>64:cache.pop(next(iter(cache)))
        save(dest,{**row,'scores':result,'elapsed_s':time.time()-started,'protocol_sha256':sha256(root/'protocol.json'),
                   'calibration_sha256':sha256(root/'calibration.json')})
        print(part,index+1,'/',len(manifest),row['case_id'],round(time.time()-started,2),flush=True)
    save(root/('completed_'+part+'.json'),{'status':'COMPLETED','completed_unix':time.time(),'cases':len(manifest)})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('prepare','development','visual','pixel','verify'))
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.phase=='prepare':prepare(args.output)
    elif args.phase=='development':development(args.output)
    elif args.phase in ('visual','pixel'):run(args.output,args.phase)
    else:verify(args.output,(args.output/'calibration.json').exists());print('frozen inputs verified')


if __name__=='__main__':main()
