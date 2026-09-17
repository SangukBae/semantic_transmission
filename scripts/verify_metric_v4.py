#!/usr/bin/env python3
"""Final preservation, truth, chronology and byte replay audit for v4."""
import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import time

import numpy as np

from semantic_transmission.artifacts import sha256
from semantic_transmission.metric_v3_formal import read,save,array
from semantic_transmission.metric_v4_pipeline import verify,rows,OLD
from semantic_transmission.sta_video_validation import receive
from semantic_transmission.metric_v4_cases import truth_query


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();root=args.output
    p=verify(root,True);prep=read(root/'preparation.json');cal=read(root/'calibration.json')
    assert sha256(root/'cases.jsonl')==prep['manifest_sha256']
    for name,digest in prep['RGB_and_truth_file_sha256'].items():assert sha256(root/name)==digest,name
    manifest=rows(root/'cases.jsonl');counts=Counter();times=[];score_hashes={}
    for part in ('visual','pixel'):
        paths=list((root/'scores'/part).glob('*.json'))
        assert {f.stem for f in paths}=={r['case_id'] for r in manifest}
        assert read(root/('completed_'+part+'.json'))['status']=='COMPLETED'
        for row in manifest:
            path=root/'scores'/part/(row['case_id']+'.json');r=read(path)
            assert r['protocol_sha256']==sha256(root/'protocol.json')
            assert r['calibration_sha256']==sha256(root/'calibration.json')
            for field in ('case_id','source_id','corpus'):assert r[field]==row[field]
            times.append(path.stat().st_mtime);counts[part+'/'+row['corpus']]+=1
            score_hashes[str(path.relative_to(root))]=sha256(path)
    assert cal['created_unix']<min(times)
    assert max(f.stat().st_mtime for f in (root/'development_event').glob('*.json'))<cal['created_unix']
    assert p['future_heldout_scores_observed']==0 and cal['new_heldout_scores_observed']==0
    for filename,script in (('statistics_declaration.json','scripts/report_metric_v4.py'),('input_audit_declaration.json','scripts/audit_metric_v4_inputs.py')):
        d=read(root/filename);assert d['created_unix']<min(times) and d['heldout_scores_observed']==0
        assert d['script_sha256']==sha256(Path(script))
    operation=read(root/'report_serialization_operation.json')
    assert operation['wrapper_sha256']==sha256(Path('scripts/run_metric_v4_report.py'))
    assert operation['frozen_statistics_sha256']==sha256(Path('scripts/report_metric_v4.py'))
    assert not any(operation[key] for key in ('metric_changed','statistics_changed','threshold_changed'))
    training=set(cal['models']['sta']['fit']['training_sources']);development=set(cal['models']['sta']['rejection']['calibration_sources'])
    test={r['source_id'] for r in manifest}
    assert not training&development and not test&(training|development)
    old_pixels={r['source_pixel_sha256'] for r in rows(OLD/'cases.jsonl')}
    assert not old_pixels&{r['source_pixel_sha256'] for r in manifest}
    replay=0;packet_status=Counter();truth_cases=0
    for row in manifest:
        folder=root/'cases'/row['case_id']
        if row['corpus']=='sta':
            assert sha256(folder/'transport.json')==row['transport_sha256']
            assert sha256(folder/'packets.pkl')==row['packets_sha256']
            with (folder/'packets.pkl').open('rb') as f:packets=pickle.load(f)
            recorded=read(folder/'transport.json')
            for side in ('tx','rx'):
                decoded,statuses,_=receive(packets[side+'_packets'])
                expected=array(folder/(side+'.npz'),row[side+'_pixel_sha256'])
                assert np.array_equal(decoded,expected) and statuses==recorded[side+'_status']
                packet_status.update(side+'/'+s for s in statuses)
            replay+=1
        elif row['corpus']=='event' and row['target']!='control':
            with np.load(root/'sources'/row['source_stem']/'truth.npz') as f:a={k:f[k] for k in f.files}
            with np.load(folder/'truth.npz') as f:b={k:f[k] for k in f.files}
            assert truth_query(a,b,row['target_object'],row['target_event'],row['target_frame'])==row['oracle']
            truth_cases+=1
    prior=read(Path('docs/validation/2026-09-14-ere-sta-formal.json'))
    for name,digest in prior['artifact_sha256'].items():assert sha256(OLD/name)==digest,name
    for name,digest in read(OLD/'score_artifact_sha256.json')['sha256'].items():assert sha256(OLD/name)==digest,name
    model=read(Path(read(OLD/'protocol.json')['base_run'])/'protocol.json')['models'];model_root=Path(model['root'])
    for path,key in ((model_root/'sam2.1_hiera_tiny.pt','sam2_weights_sha256'),(model_root/'dinov2_vits14_pretrain.pth','dino_weights_sha256'),
                     (Path.home()/'.cache/torch/hub/checkpoints/raft_small_C_T_V2-01064c6d.pth','raft_weights_sha256'),
                     (Path.home()/'.cache/clip/ViT-B-32.pt','clip_weights_sha256')):assert sha256(path)==model[key]
    save(root/'score_artifact_sha256.json',{'recorded_unix':time.time(),'score_count':len(score_hashes),'sha256':score_hashes})
    result={'status':'PASSED','recorded_unix':time.time(),'score_counts':dict(counts),'score_records':sum(counts.values()),
            'RGB_truth_files_verified':len(prep['RGB_and_truth_file_sha256']),'STA_packet_roundtrips':replay,
            'packet_status_counts':dict(packet_status),'independent_target_truth_reproduced':truth_cases,
            'old_score_files_preserved':len(read(OLD/'score_artifact_sha256.json')['sha256']),
            'old_top_level_artifacts_preserved':len(prior['artifact_sha256']),'model_weight_hashes_verified':4,
            'source_ID_and_RGB_overlap':False,'calibration_created_unix':cal['created_unix'],'first_heldout_score_mtime':min(times),
            'last_heldout_score_mtime':max(times),'interpretation':'Integrity and operational checks, not metric adoption'}
    save(root/'final_integrity.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
