#!/usr/bin/env python3
"""Allocate source groups, package ETRI review materials, and audit readiness."""
import argparse
import collections
import concurrent.futures as cf
import csv
import datetime as dt
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

from etri_benchmark import DATA, PILOT, REPO, save, sha, run, probe

LEVELS=('low','medium','high')
SPLITS=('development','calibration','test')


def candidates():
    return sum([json.loads((DATA/f'metadata/{d}_candidates_v2.json').read_text())
                for d in ['tvsum','clipshots']],[])


def duplicate_screen():
    rows=candidates();flags=[];exact=[]
    hashes={r['curation_id']:json.loads((DATA/'reports/scans'/r['dataset']/(r['source_id']+'.json')).read_text())['phash_2sec'] for r in rows}
    for a,b in itertools.combinations(rows,2):
        if a['source_sha256']==b['source_sha256']:exact.append([a['curation_id'],b['curation_id']])
        aa=[int(h,16) for h in hashes[a['curation_id']]];bb=[int(h,16) for h in hashes[b['curation_id']]]
        offsets=collections.Counter()
        for i,x in enumerate(aa):
            for j,y in enumerate(bb):
                if (x^y).bit_count()<=6:offsets[j-i]+=1
        if offsets:
            offset,matches=offsets.most_common(1)[0]
            if matches>=8:
                flags.append({'a':a['curation_id'],'b':b['curation_id'],'matching_2s_samples':matches,
                    'offset_sec':offset*2,'status':'REQUIRES_REVIEW_OR_SHARED_SPLIT'})
    result={'scope':'All 60 selected full sources, not only chosen windows','exact_duplicates':exact,
            'perceptual_flags':flags,'method':'32x32 pHash every 2s; Hamming <=6, at least 8 aligned matches',
            'limitations':'Screening is not proof of absence of crops, mirrored copies, short overlap or common pretraining exposure.'}
    save(DATA/'reports/duplicate_screen.json',result);print(json.dumps(result,ensure_ascii=False))


def extension_events(row):
    start=row['window']['start_sec']
    if row['duration_sec']<start+120:return None
    pred=json.loads((DATA/'reports/transnet'/row['dataset']/(row['source_id']+'.json')).read_text())
    events=[e for e in pred['events'] if start<e['peak_sec']<start+120]
    rate=len(events)/2
    level='low' if rate<=1 else ('medium' if rate<=8 else 'high')
    return events if level==row['level'] else None


def allocate():
    from scipy.optimize import milp, Bounds, LinearConstraint
    rows=candidates()
    reviews={}
    for dataset in ['tvsum','clipshots']:
        reviews.update(json.loads((DATA/f'annotations/visual_review_{dataset}.json').read_text()))
    related=json.loads((DATA/'metadata/related_content_groups.json').read_text())
    group_for={member:g['id'] for g in related for member in g['members']}
    for r in rows:
        r.update(reviews[r['curation_id']]);r['content_group']=group_for.get(r['curation_id'],r['curation_id'])
    duplicate=json.loads((DATA/'reports/duplicate_screen.json').read_text())
    assert not duplicate['exact_duplicates'], 'Exact duplicate sources must be removed'
    # Near-duplicate flags are conservatively locked to the same partition.
    pairs=[(g['members'][0],m) for g in related for m in g['members'][1:]]
    pairs += [(f['a'],f['b']) for f in duplicate['perceptual_flags']]
    n=len(rows)*3;matrix=[];lower=[];upper=[]
    def add(indices,lo,hi):
        vector=np.zeros(n)
        for index,value in indices:vector[index]+=value
        matrix.append(vector);lower.append(lo);upper.append(hi)
    for i in range(len(rows)):add([(3*i+j,1) for j in range(3)],1,1)
    for dataset in ['TVSum','ClipShots']:
        for level in LEVELS:
            selected=[i for i,r in enumerate(rows) if r['dataset']==dataset and r['level']==level]
            for j,target in enumerate([3,2,5]):add([(3*i+j,1) for i in selected],target,target)
            for category in {rows[i]['category'] for i in selected}:
                add([(3*i+2,1) for i in selected if rows[i]['category']==category],0,2)
            if dataset=='ClipShots' and level=='high':
                add([(3*i+2,1) for i in selected if rows[i]['category'] in ['running','skateboarding']],1,np.inf)
            if dataset=='ClipShots' and level=='medium':
                add([(3*i+2,1) for i in selected if rows[i]['category'] in ['aquatic_animals','music']],1,np.inf)
            # A continuous 120s extension of the same stratum must exist in test.
            eligible=[i for i in selected if extension_events(rows[i]) is not None]
            add([(3*i+2,1) for i in eligible],1,np.inf)
    index={r['curation_id']:i for i,r in enumerate(rows)}
    for a,b in pairs:
        for j in range(3):add([(3*index[a]+j,1),(3*index[b]+j,-1)],0,0)
    pilot=json.loads((PILOT/'manifest.json').read_text())['videos']
    pilot_ids={r['source_video_id'] for r in pilot}
    for i,r in enumerate(rows):
        if r['source_id'] in pilot_ids:add([(3*i,1)],1,1)
    costs=[]
    for row in rows:
        for split in SPLITS:
            costs.append(int(hashlib.sha256(('20260924'+row['source_id']+split).encode()).hexdigest()[:8],16)/2**32)
    result=milp(np.array(costs),integrality=np.ones(n),bounds=Bounds(0,1),
        constraints=LinearConstraint(np.array(matrix),lower,upper),options={'time_limit':60})
    if not result.success:raise ValueError(result.message)
    selections=[]
    for i,row in enumerate(rows):
        r=dict(row,id=row['curation_id'],split=SPLITS[int(np.argmax(result.x[3*i:3*i+3]))],
               role='primary_60s',clip_start_sec=row['window']['start_sec'],clip_duration_sec=60,
               previously_piloted=row['source_id'] in pilot_ids,
               source_review_status='AI_OVERVIEW_ONLY_INDEPENDENT_REVIEW_PENDING')
        if r['dataset']=='TVSum':
            r.update(source_url='https://www.youtube.com/watch?v='+r['source_id'],
                dataset_url='https://github.com/yalesong/tvsum',
                download_url='https://people.csail.mit.edu/yalesong/tvsum/tvsum50_ver_1_1.tgz',
                license_status='Author README CC BY 3.0; legacy archive terms differ; unresolved applicability retained')
        else:
            r.update(source_url=None,dataset_url='https://github.com/Tangshitao/ClipShots',
                download_url='https://drive.google.com/file/d/1iESqNwFTG9f-HXH9ghiv3W-HCa279hfb/view',
                license_status='Public research distribution; raw-video license not explicitly specified; code MIT is not a raw-video license')
        selections.append(r)
    extensions=[]
    for dataset in ['TVSum','ClipShots']:
        for level in LEVELS:
            eligible=[r for r in selections if r['dataset']==dataset and r['level']==level and r['split']=='test' and extension_events(r) is not None]
            row=min(eligible,key=lambda r:hashlib.sha256(('extension20260924'+r['source_id']).encode()).hexdigest())
            events=extension_events(row)
            extensions.append(dict(row,id=row['id']+'_120s',role='extension_120s',parent_id=row['id'],
                clip_duration_sec=120,transitions=events,shot_count=len(events),
                extension_status='Same source and split as parent; not an additional independent sample'))
    save(DATA/'metadata/selections.json',selections+extensions)
    save(DATA/'reports/allocation.json',{'seed':20260924,'main_sources':len(selections),
        'groups':len({r['content_group'] for r in selections}),'extensions':len(extensions),
        'split_counts':dict(collections.Counter(r['split'] for r in selections)),
        'heldout_source_definition':'No reconstruction inspected for selection. Existing-source pretraining exposure is unknown.',
        'extension_ids':[r['id'] for r in extensions]})
    print(json.dumps({'primary':len(selections),'extensions':[r['id'] for r in extensions],
                     'splits':dict(collections.Counter(r['split'] for r in selections))}))


def validate_splits(rows):
    errors=[]
    for key in ['source_id','source_sha256','content_group']:
        buckets=collections.defaultdict(set)
        for r in rows:buckets[r[key]].add(r['split'])
        errors += [f'{key} crosses splits: {k}' for k,v in buckets.items() if len(v)>1]
    for r in rows:
        if r.get('previously_piloted') and r['split']!='development':errors.append('Pilot source outside development: '+r['id'])
    return errors


def csv_write(path,fields,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)


def make_tasks():
    rows=json.loads((DATA/'manifest.json').read_text());boundaries=[];jobs=[];review_tasks=[]
    for r in rows:
        start=r['clip_start_sec'];duration=r['clip_duration_sec']
        pred=json.loads((DATA/'reports/transnet'/r['dataset']/(r['source_id']+'.json')).read_text())
        proposals=[]
        for event in pred['low_confidence_events']:
            if start<event['peak_sec']<start+duration:
                proposals.append({'start_sec':max(0,event['start_sec']-start),'end_sec':min(duration,event['end_sec']-start),
                    'peak_sec':event['peak_sec']-start,'probability':event['probability'],
                    'origin':'TransNetV2 threshold 0.2 candidate; selected stratum uses 0.5',
                    'status':'UNREVIEWED','shot_or_semantic_scene':'UNRESOLVED'})
        official=[]
        if r['dataset']=='ClipShots':
            ann=json.loads((DATA/'metadata/clipshots_annotations_only_gradual.json').read_text()).get(r['source_id']+'.mp4')
            if ann:
                pts=np.load(DATA/'reports/transnet'/r['dataset']/(r['source_id']+'.npz'))['pts']
                for a,b in ann['transitions']:
                    if not 0<=a<=b<len(pts):continue
                    lo,hi=float(pts[a])-start,float(pts[b])-start
                    if hi>0 and lo<duration:official.append({'start_sec':max(0,lo),'end_sec':min(duration,hi),
                        'official_frames':[a,b],'frame_index_origin':'Zero-based assumption for review; convention requires independent check',
                        'official_frame_num':ann['frame_num'],'decoded_frames':len(pts),
                        'status':'PROVIDED_PARTIAL_SHOT_LABEL_NOT_SEMANTIC_SCENE_TRUTH'})
        boundaries.append({'id':r['id'],'duration_sec':duration,'candidates':proposals,
            'official_partial_labels':official,'coverage':'Full continuous review still required; candidates can miss cuts.',
            'status':'INDEPENDENT_REVIEW_PENDING'})
        for reviewer in ['reviewer_A','reviewer_B']:
            review_tasks.append({'id':r['id'],'reviewer_slot':reviewer,'reviewer_id':'','source_sha256':r['source_sha256'],
                                'coverage_start_sec':0,'coverage_end_sec':duration,'status':'NOT_STARTED',
                                'independent':False,'adjudicated':False})
        for seed in [42,43,44]:
            for method in ['baseline','mitigation']:
                jobs.append({'job_id':f"{r['id']}_awgn10_seed{seed}_{method}",'id':r['id'],'split':r['split'],
                    'role':r['role'],'source_group':r['content_group'],'input_path':r['processed_path'],
                    'input_sha256':r['processed_sha256'],'duration_sec':duration,'expected_frames':r['frames'],
                    'channel':'AWGN','snr_db':10,'channel_seed':seed,'decoder_seed':2025,'method':method,
                    'pair_id':f"{r['id']}_awgn10_seed{seed}",'execution_status':'NOT_RUN',
                    'launch_gate':'METHOD_CONFIG_AND_LONG_RUNNER_VALIDATION_PENDING' if r['split']=='development' else 'DEVELOPMENT_CALIBRATION_AND_METHOD_FREEZE_PENDING',
                    'method_config_sha256':None,'source_visibility_at_receiver':False})
    save(DATA/'annotations/source_boundaries.provisional.json',boundaries)
    save(DATA/'annotations/source_review_tasks.json',review_tasks)
    save(DATA/'plans/paired_jobs.json',jobs)
    csv_write(DATA/'plans/paired_jobs.csv',list(jobs[0]),jobs)
    csv_write(DATA/'annotations/reconstruction_review_tasks.csv',
        ['job_id','reviewer_slot','reviewer_id','status','reviewed_seconds','zero_error_confirmed','annotation_file'],
        [dict(job_id=j['job_id'],reviewer_slot=slot,reviewer_id='',status='NOT_STARTED',reviewed_seconds=0,zero_error_confirmed='',annotation_file='') for j in jobs for slot in ['reviewer_A','reviewer_B']])
    csv_write(DATA/'annotations/reconstruction_errors.template.csv',
        ['job_id','event_id','reviewer_id','start_sec','end_sec','type','entity','relation_or_action','severity',
         'bbox_x0','bbox_y0','bbox_x1','bbox_y1','temporal_tags','certainty','source_evidence','output_evidence','status'],[])
    csv_write(DATA/'annotations/boundary_refresh.template.csv',
        ['job_id','reference_scene_id','reference_time_sec','detected_time_sec','keyframe_source_frame',
         'packet_created_time_sec','packet_received_time_sec','packet_applied_time_sec','missed_detection',
         'fallback','residual_old_scene','status'],[])
    csv_write(DATA/'plans/result_receipts.template.csv',
        ['job_id','execution_status','output_path','output_sha256','output_frames','output_duration_sec',
         'input_sha256','method_config_sha256','received_packet_sha256','visual_channel_uses','metadata_channel_uses',
         'additional_channel_uses','wall_seconds','peak_gpu_memory_mb','retries','failure_reason'],[])
    pilot=json.loads((PILOT/'manifest.json').read_text())['videos']
    save(DATA/'metadata/diagnostics.json',{'scope':'TUM pilot inputs preserved as supplemental diagnostics, excluded from main balanced estimates.',
        'videos':[r for r in pilot if r.get('dataset','').lower().startswith('tum')]})
    print(json.dumps({'paired_jobs':len(jobs),'source_review_tasks':len(review_tasks)}))


def source_reviews_complete(tasks, expected):
    """An empty file, repeated reviewer or uncovered time is not ground truth."""
    if not expected:return False
    for video_id,duration in expected.items():
        approved=[r for r in tasks if r.get('id')==video_id and r.get('status')=='COMPLETE'
            and r.get('independent') is True and r.get('adjudicated') is True
            and r.get('coverage_start_sec')==0 and r.get('coverage_end_sec',-1)>=duration
            and str(r.get('reviewer_id','')).strip()]
        if len({r['reviewer_id'] for r in approved})<2:return False
    return True


def validate_job_inputs(rows,jobs):
    inputs={r['id']:r for r in rows};errors=[]
    for job in jobs:
        row=inputs.get(job['id'])
        if row is None:
            errors.append('Unknown job input: '+job['id']);continue
        if job['input_sha256']!=row['processed_sha256'] or job['expected_frames']!=row['frames']:
            errors.append('Job does not reference its frozen input: '+job['id'])
    return errors


def audit():
    rows=json.loads((DATA/'manifest.json').read_text())
    primary=[r for r in rows if r['role']=='primary_60s'];errors=validate_splits(rows)
    if len(primary)!=60 or len({r['source_id'] for r in primary})!=60:errors.append('Expected 60 distinct primary sources')
    for dataset in ['TVSum','ClipShots']:
        for level in LEVELS:
            for split,n in zip(SPLITS,[3,2,5]):
                if sum(r['dataset']==dataset and r['level']==level and r['split']==split for r in primary)!=n:
                    errors.append(f'Allocation mismatch: {dataset}/{level}/{split}')
    source_hashes={}
    for r in rows:
        path=Path(r['processed_path'])
        if not path.exists() or sha(path)!=r['processed_sha256']:errors.append('Processed hash mismatch: '+r['id'])
        source=Path(r['source_path'])
        if source not in source_hashes:source_hashes[source]=sha(source)
        if source_hashes[source]!=r['source_sha256']:errors.append('Source hash mismatch: '+r['id'])
        duration=r['clip_duration_sec'];expected=int(duration*24)
        if r['frames']!=expected or abs(r['output_duration_sec']-duration)>.001:errors.append('Duration mismatch: '+r['id'])
        if r['source_time_span_sec']<duration-.15:errors.append('Source span too short: '+r['id'])
        if r['full_output_decode']!='PASS':errors.append('Decode failed: '+r['id'])
        if r['source_max_frame_gap_sec']>=.125:errors.append('Source capture gap: '+r['id'])
        mapping=json.loads((DATA/'metadata/time_mapping'/(r['id']+'.json')).read_text())
        if len(mapping)!=expected or [x['output_frame'] for x in mapping]!=list(range(expected)):
            errors.append('Incomplete time mapping: '+r['id'])
        if r['role']=='extension_120s':
            parent=next((x for x in primary if x['id']==r['parent_id']),None)
            if not parent or parent['source_id']!=r['source_id'] or parent['split']!=r['split']:errors.append('Extension leakage: '+r['id'])
    selections=json.loads((DATA/'metadata/selections.json').read_text())
    if {r['id'] for r in rows}!={r['id'] for r in selections}:errors.append('Manifest/selection ID mismatch')
    for r in selections:
        actual=next((x for x in rows if x['id']==r['id']),{})
        digest=hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest()
        if actual.get('selection_sha256')!=digest:errors.append('Stale selection metadata: '+r['id'])
    pilot=json.loads((DATA/'metadata/pilot_freeze_before.json').read_text())
    pilot_bad=[r['path'] for r in pilot['files'] if not Path(r['path']).exists() or sha(Path(r['path']))!=r['sha256']]
    if pilot_bad:errors.append('Pilot frozen files changed')
    jobs=json.loads((DATA/'plans/paired_jobs.json').read_text())
    errors.extend(validate_job_inputs(rows,jobs))
    pairs=collections.defaultdict(list)
    for j in jobs:pairs[j['pair_id']].append(j)
    if len(jobs)!=len(rows)*6:errors.append('Incomplete planned jobs')
    for pair,items in pairs.items():
        if len(items)!=2 or {j['method'] for j in items}!={'baseline','mitigation'}:errors.append('Invalid pair: '+pair)
        for key in ['input_sha256','channel','snr_db','channel_seed','decoder_seed','expected_frames']:
            if len({j[key] for j in items})!=1:errors.append('Unmatched pair '+key+': '+pair)
    tasks=json.loads((DATA/'annotations/source_review_tasks.json').read_text())
    truth_ready=source_reviews_complete(tasks,{r['id']:r['clip_duration_sec'] for r in rows})
    stats={'status':'PASS_INPUT_PREPARATION' if not errors else 'FAIL','errors':errors,
        'primary_sources':len(primary),'primary_minutes':sum(r['output_duration_sec'] for r in primary)/60,
        'extension_videos':sum(r['role']=='extension_120s' for r in rows),
        'total_processed_videos':len(rows),'total_frames':sum(r['frames'] for r in rows),
        'split_counts':dict(collections.Counter(r['split'] for r in primary)),
        'content_groups':len({r['content_group'] for r in primary}),
        'test_content_groups':len({r['content_group'] for r in primary if r['split']=='test'}),
        'cfr_repeated_frames':sum(r['cfr_repeated_source_frames'] for r in rows),
        'max_sampling_error_sec':max(r['max_sampling_error_sec'] for r in rows),
        'max_source_frame_gap_sec':max(r['source_max_frame_gap_sec'] for r in rows),
        'paired_jobs_planned':len(jobs),'pilot_frozen_files_verified':len(pilot['files'])-len(pilot_bad),
        'independent_source_ground_truth_ready':truth_ready,'performance_evaluation_certified':False,
        'pending':['Independent full-duration source and group review','Long-video runner validation and method/configuration freeze',
                   'Actual paired AWGN reconstructions and transmission-cost receipts',
                   'Blinded independent Added/Missing/Distorted labels and result audit'],
        'classification_status':'Provisional model counts plus AI overview review; not independent ground truth'}
    save(DATA/'reports/readiness_audit.json',stats);print(json.dumps(stats,ensure_ascii=False,indent=2))
    if errors:raise SystemExit(1)


def freeze():
    result=json.loads((DATA/'reports/readiness_audit.json').read_text())
    if result['status']!='PASS_INPUT_PREPARATION':raise ValueError('Input audit must pass before freeze')
    selected=[]
    for pattern in ['manifest.*','index.html','metadata/*.json','metadata/*.md','metadata/*.tsv',
                    'metadata/transnetv2/*','metadata/time_mapping/*.json',
                    'annotations/*.json','annotations/*.csv','plans/*','reports/readiness_audit.json',
                    'reports/duplicate_screen.json','reports/allocation.json','processed/*.mp4','review/v2/*.jpg',
                    'review/extensions/*.jpg','review/previews/*.mp4','reports/preview_validation.json',
                    'reports/source_inventory.json','reports/archive_member_audit.json','reports/extraction.json',
                    'reports/acquisition_retry.json','reports/unit_tests.json','reports/package_validation.json',
                    'reports/scans/*/*.json','reports/transnet/*/*.json','reports/transnet/*/*.npz',
                    'reports/preparation/*','reports/source_validation/*/*.json']:
        selected.extend(DATA.glob(pattern))
    selected.extend(REPO/'scripts'/name for name in ['etri_benchmark.py','etri_shot_candidates.py','etri_benchmark_package.py','etri_benchmark_review.html'])
    selected.extend([REPO/'tests/test_etri_benchmark.py',REPO/'docs/ETRI_BENCHMARK_V1.md'])
    save(DATA/'freeze_manifest.json',{'frozen_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'scope':'Prepared inputs, source-group split, provisional annotation drafts and evaluation plans. No performance result claim.',
        'independent_labels_frozen':False,'model_configuration_frozen':False,
        'files':[{'path':str(p.resolve()),'sha256':sha(p)} for p in sorted(set(selected)) if p.is_file()]})


def review_page():
    rows=json.loads((DATA/'manifest.json').read_text())
    boundaries={r['id']:r for r in json.loads((DATA/'annotations/source_boundaries.provisional.json').read_text())}
    small=[]
    for r in rows:
        small.append({k:r[k] for k in ['id','dataset','source_id','level','split','role','category','content_group',
            'clip_start_sec','clip_duration_sec','shot_count','observations_ko','tags','processed_sha256']} |
            {'boundaries':boundaries[r['id']],'video':'processed/'+r['id']+'.mp4',
             'master':'processed/'+r['id']+'.mp4',
             'sheet':('review/extensions/' if r['role']=='extension_120s' else 'review/v2/')+r['id']+'.jpg'})
        small[-1]['video']='review/previews/'+r['id']+'.mp4'
        if r['role']=='extension_120s':
            notes=json.loads((DATA/'annotations/extension_visual_review.json').read_text())
            small[-1]['observations_ko']=notes[r['id']]['observations_ko']
    template=(REPO/'scripts/etri_benchmark_review.html').read_text()
    payload=json.dumps(small,ensure_ascii=False).replace('</','<\\/')
    (DATA/'index.html').write_text(template.replace('/*__DATA__*/',payload))


def previews():
    rows=json.loads((DATA/'manifest.json').read_text())
    root=DATA/'review/previews';root.mkdir(exist_ok=True)
    def encode(r):
        dest=root/(r['id']+'.mp4')
        run(['ffmpeg','-v','error','-nostdin','-y','-threads','2','-i',r['processed_path'],
             '-an','-c:v','libx264','-threads','2','-preset','fast','-crf','12','-profile:v','high',
             '-pix_fmt','yuv420p','-vsync','0','-movflags','+faststart',str(dest)])
        p=probe(dest,True)
        assert int(p['streams'][0]['nb_read_frames'])==r['frames']
        assert abs(float(p['format']['duration'])-r['clip_duration_sec'])<.001
        return {'id':r['id'],'preview_sha256':sha(dest),'master_sha256':r['processed_sha256'],
            'frames':r['frames'],'duration_sec':r['clip_duration_sec'],'encoding':'Browser preview only: H.264 High, CRF 12',
            'model_input':False,'probe':p}
    with cf.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(encode,rows))
    save(DATA/'reports/preview_validation.json',results)
    print(json.dumps({'previews':len(results),'timing_and_frame_counts':'PASS'}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['duplicates','allocate','tasks','audit','freeze','review-page','previews'])
    args=parser.parse_args()
    {'duplicates':duplicate_screen,'allocate':allocate,'tasks':make_tasks,'audit':audit,'freeze':freeze,'review-page':review_page,'previews':previews}[args.command]()


if __name__=='__main__':main()
