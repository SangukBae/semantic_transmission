#!/usr/bin/env python3
"""Audit matched SPARC adaptive/uniform/fixed outputs and score final MP4s."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.bsu_predict import inspect_frames
from semantic_transmission.official_quality import OfficialMetrics
from semantic_transmission.packets import accounting
from semantic_transmission.research_quality import read_video
from semantic_transmission.sparc import STAGES, COMPLETE
from semantic_transmission.webvid5 import fingerprint
from semantic_transmission.webvid_ablation import snapshot


def read(path):
    return json.loads(path.read_text())


def audit(root, mode):
    p=read(root/'protocol.json')
    if p['signature']!=fingerprint({k:v for k,v in p.items() if k!='signature'}):raise ValueError('protocol changed')
    deps={};checked=0
    for stage in STAGES:
        receipt=read(root/f'stages/{stage}.json')
        if (receipt['status']!='PASSED' or receipt['identity']!=p['signature'] or receipt['dependencies']!=deps
                or snapshot(root,receipt['required'])!=receipt['artifacts']):raise ValueError('invalid stage receipt: '+stage)
        checked+=len(receipt['artifacts']);deps[stage]=fingerprint(receipt)
    source=read(root/'source.json');result=read(root/'RESULT.json')
    if inspect_frames(Path(source['directory']),source['video']['fps'])!=source:raise ValueError('source changed')
    if (result['status']!=COMPLETE or result['anchor_policy']!=mode or
            result['protocol_sha256']!=sha256(root/'protocol.json') or
            result['movie_sha256']!=sha256(root/'reconstruction/reconstruction.mp4')):raise ValueError('invalid completion record')
    cost=accounting((root/'sender/packet.bin').stat().st_size)['complex_channel_uses']
    if result['channel_uses']!=cost or read(root/'received/channel.json')['complex_channel_uses']!=cost:
        raise ValueError('actual packet cost mismatch')
    return p,source,result,checked


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for mode in ('adaptive','uniform','fixed'):parser.add_argument('--'+mode,type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('comparison output exists; use a new directory')
    cases=[('adaptive',args.adaptive,'adaptive'),('uniform',args.uniform,'uniform-matched'),
           ('fixed',args.fixed,'fixed-stride2')]
    audited=[audit(root,mode) for _,root,mode in cases]
    first=audited[0];shared_profile={k:v for k,v in first[0]['profile'].items() if k!='anchor_policy'}
    for p,source,result,_ in audited[1:]:
        if (source['video']!=first[1]['video'] or source['hashes']!=first[1]['hashes'] or
                {k:v for k,v in p['profile'].items() if k!='anchor_policy'}!=shared_profile or
                result['semantic_keys']!=first[2]['semantic_keys']):raise ValueError('comparison conditions differ')
        if any(p[key]!=first[0][key] for key in ('code','vendor_code','binaries')):
            raise ValueError('comparison code or binaries differ')
        if any(first[0]['assets'].get(path)!=digest for path,digest in p['assets'].items()):
            raise ValueError('comparison codec or interpolator assets differ')
        if any(p['runtimes'][role]!=first[0]['runtimes'][role] for role in ('codec','receiver','channel')):
            raise ValueError('comparison inference runtime differs')
    semantics=[read(root/'semantic.json') for _,root,_ in cases]
    if any(any(s[key]!=semantics[0][key] for key in ('identity','keys','decisions')) for s in semantics[1:]):
        raise ValueError('comparison semantic decisions differ')
    if len(first[2]['indices'])!=len(audited[1][2]['indices']):raise ValueError('adaptive/uniform anchor counts differ')
    args.output.mkdir(parents=True)
    source=first[1]
    reference=np.stack([np.asarray(Image.open(Path(source['directory'])/f'{i}.png').convert('RGB'))
                        for i in range(source['video']['frames'])])
    reports={}
    for (label,root,_),(_,_,result,checked) in zip(cases,audited):
        frames=read_video(root/'reconstruction/reconstruction.mp4')
        if frames.shape!=reference.shape:raise ValueError('final MP4 frame alignment differs')
        quality,rows=OfficialMetrics().evaluate(reference,frames)
        reports[label]=dict(anchor_count=len(result['indices']),indices=result['indices'],
            channel_uses=result['channel_uses'],quality=quality,movie_sha256=result['movie_sha256'],
            protocol_sha256=result['protocol_sha256'],verified_artifact_hashes=checked)
        with (args.output/f'{label}_per_frame.csv').open('x') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        print(label,json.dumps(reports[label]),flush=True)
    write_json(args.output/'COMPARISON.json',dict(status='COMPLETE_SPARC_MATCHED_COMPARISON',
        video=source['video'],source_frame_hashes_identical=True,source_frame_hashes=source['hashes'],
        semantic_keys=first[2]['semantic_keys'],same_source_semantic_selection=True,
        same_anchor_count_adaptive_uniform=True,equal_actual_rate=False,
        source_reference='normalized sender RGB PNGs',evaluated_output='decoded final MP4',
        metric_profile='lgvsc_official_metrics_v1',results=reports,
        adaptive_rate_change_vs_fixed=reports['adaptive']['channel_uses']/reports['fixed']['channel_uses']-1,
        adaptive_rate_change_vs_uniform=reports['adaptive']['channel_uses']/reports['uniform']['channel_uses']-1,
        independent_action_or_semantic_evaluation=False,
        limits=['one video at one SNR and seed','fixed thresholds, no training or quality-based tuning',
                'uniform control shares detected cuts and anchor count but does not preserve semantic keys',
                'fixed control uses SPARC scene resets and metadata envelope, not a byte-identical legacy BSU packet'] ))


if __name__=='__main__':main()
