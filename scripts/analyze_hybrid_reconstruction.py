"""Read-only run audit and CPU evidence for the completed hybrid/AI-caption run.

Writes only an analysis subdirectory. Does not run a model or overwrite frozen
execution receipts, captions, keyframes, quality measurements or videos.
"""
import csv
import datetime
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from semantic_transmission.artifacts import sha256,write_json
from semantic_transmission.assisted_captions import validate_bundle
from semantic_transmission.etri_60s_check import pts_audit
from semantic_transmission.hybrid_selection import verify_selection
from semantic_transmission.video_io import probe
from semantic_transmission.webvid5 import fingerprint,read_json
from semantic_transmission.webvid_ablation import snapshot

SELECTION=REPO/'outputs/etri_hybrid_keys_tv_low_08_v1'
ROOT=SELECTION/'reconstruction_assistant_captions'
DEST=ROOT/'analysis'
BASE=REPO/'outputs/etri_60s_tv_low_08_42057b2ee8ed/baseline'
RUNS={'skem110_pllava':BASE,
      'ai85_pllava':REPO/'outputs/etri_visual_keys_20260929/assistant',
      'hybrid79_ai_captions':ROOT/'run'}
METRICS=('psnr_db','ssim','lpips_vgg','clip','dists')


def rows(run):
    with (run/'quality_delivered_mp4.csv').open() as stream:
        values=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(stream)]
    if [r['frame'] for r in values] != list(range(1440)):
        raise ValueError('quality CSV frame axis changed')
    return values


def mean(values,indices):
    return {k:float(np.mean([values[i][k] for i in indices])) for k in METRICS}


def audit():
    protocol,selection=verify_selection(SELECTION)
    bundle=validate_bundle(SELECTION/'assistant_captions/captions_bundle.json',SELECTION)
    execution=read_json(ROOT/'execution_protocol.json')
    if fingerprint({k:v for k,v in execution.items() if k!='signature'}) != execution['signature']:
        raise ValueError('execution identity changed')
    for name,digest in execution['code'].items():
        if sha256(REPO/name)!=digest: raise ValueError(f'execution code changed: {name}')
    for name,digest in execution['baseline_files'].items():
        if sha256(name)!=digest: raise ValueError(f'baseline file changed: {name}')
    if execution['caption_bundle_sha256']!=sha256(SELECTION/'assistant_captions/captions_bundle.json'):
        raise ValueError('caption bundle changed after run')
    result=read_json(ROOT/'RESULT.json')
    dependencies={}; artifacts={}
    for name in result['stage_seconds']:
        stage=read_json(ROOT/f'stages/{name}.json')
        if (stage['status']!='PASSED' or stage['identity']!=execution['signature']
                or stage['dependencies']!={k:fingerprint(v) for k,v in dependencies.items()}):
            raise ValueError(f'completed stage identity changed: {name}')
        if snapshot(ROOT,stage['required'])!=stage['artifacts']:
            raise ValueError(f'completed stage files changed: {name}')
        dependencies[name]=stage; artifacts.update(stage['artifacts'])
    cfg=read_json(ROOT/'run/run_config.json')
    if cfg != execution['config']: raise ValueError('run config changed')
    current=read_json(ROOT/'run/captions.json')
    if [r['text'] for r in current] != [r['text'] for r in bundle['records']]:
        raise ValueError('used captions differ from authored bundle')
    audit=read_json(ROOT/'run/output_audit.json')
    if audit['source_indices']!=list(range(1440)) or audit['cross_segment_transfers']!=77:
        raise ValueError('incomplete output timeline/reference state')
    details={}
    for name,run in RUNS.items():
        quality=read_json(run/'quality.json')
        video=run/'receiver/reconstruction/sample_0000.mp4'
        if quality['status']!='PASSED' or sha256(video)!=quality['video_sha256']:
            raise ValueError(f'quality/video mismatch: {name}')
        if quality['source_sha256']!=protocol['input_sha256']:
            raise ValueError(f'different source: {name}')
        info=probe(video)
        if any(info[k]!=protocol[k] for k in ('fps','frames','width','height')):
            raise ValueError('video shape changed')
        details[name]=dict(video_sha256=sha256(video),video=info,pts=pts_audit(video,1440,24))
    pts_audit(ROOT/'comparison.mp4',1440,24)
    return dict(status='VERIFIED',stages=len(dependencies),unique_stage_artifacts=len(artifacts),
        videos=details,caption_text_match=True,selection_keys=len(selection['indices']),
        cross_segment_transfers=77,reconstruction_sha256=details['hybrid79_ai_captions']['video_sha256'])


def main():
    if (DEST/'AI_REVIEW.json').exists():
        raise SystemExit('Completed AI review exists; refusing to overwrite reviewed evidence. Use audit() for a read-only recheck.')
    verified=audit()
    DEST.mkdir(exist_ok=True)
    values={name:rows(run) for name,run in RUNS.items()}
    keys={name:set(read_json(run/'keyframes.json')['indices']) for name,run in RUNS.items()}
    union=set().union(*keys.values()); interior=[i for i in range(1440) if i not in union]
    plan=read_json(REPO/'outputs/etri_visual_keys_20260929/quality_plan.json')
    conditions={}
    for name,run in RUNS.items():
        q=read_json(run/'quality.json'); channel=read_json(run/'channel_accounting.json')
        avg=mean(values[name],range(1440))
        if any(abs(avg[k]-q['delivered_mp4'][k])>1e-8 for k in METRICS):
            raise ValueError('CSV and aggregate metrics disagree')
        stage_root=run.parent
        elapsed={p.stem:read_json(p)['seconds'] for p in (stage_root/'stages').glob('*.json')}
        conditions[name]=dict(keyframes=len(keys[name]),quality=q['delivered_mp4'],lossless_quality=q['lossless_frames'],
            common_interior_count=len(interior),common_interior_quality=mean(values[name],interior),
            own_key_position_quality=mean(values[name],sorted(keys[name])),
            own_generated_position_quality=mean(values[name],[i for i in range(1440) if i not in keys[name]]),
            channel_uses={k:channel[k] for k in ('total_complex_channel_uses','visual_complex_channel_uses','digital_complex_channel_uses')},
            metadata_bytes=channel['packet_bytes'],
            caption_utf8_bytes=sum(len(r['text'].encode()) for r in read_json(run/'captions.json')),
            stage_seconds=elapsed,stage_minutes_total=sum(elapsed.values())/60,
            uniform_windows=[dict(frames=[a,b],quality=mean(values[name],range(a,b+1))) for a,b in plan['uniform_windows']],
            event_windows={k:dict(frames=[a,b],quality=mean(values[name],range(a,b+1))) for k,(a,b) in plan['source_event_windows'].items()})
    new=conditions['hybrid79_ai_captions']; deltas={}
    for name in ('skem110_pllava','ai85_pllava'):
        old=conditions[name]
        deltas[name]=dict(quality={k:new['quality'][k]-old['quality'][k] for k in METRICS},
            common_interior_quality={k:new['common_interior_quality'][k]-old['common_interior_quality'][k] for k in METRICS},
            channel_percent_change=100*(new['channel_uses']['total_complex_channel_uses']/old['channel_uses']['total_complex_channel_uses']-1))
    differences=[values['hybrid79_ai_captions'][i]['lpips_vgg']-values['skem110_pllava'][i]['lpips_vgg'] for i in range(1440)]
    extremes={}
    for label,reverse in [('metric_regressions',True),('metric_improvements',False)]:
        chosen=[]
        for i in sorted(range(1440),key=lambda i:differences[i],reverse=reverse):
            if all(abs(i-j)>=24 for j in chosen):chosen.append(i)
            if len(chosen)==6:break
        extremes[label]=chosen
    groups={'prior_source_plan':plan['ai_review_frames'],**extremes,
            'previous_failures':[0,18,136,1206]}
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
    sources=[BASE/'data/frames/sample',*[r/'receiver/reconstruction/sample_0000_frames' for r in RUNS.values()]]
    labels=['SOURCE','SKEM110 + PLLaVA','AI85 + PLLaVA','HYBRID79 + AI CAPTION']
    sheets=[]
    for group,indices in groups.items():
        for page,start in enumerate(range(0,len(indices),4)):
            these=indices[start:start+4]
            canvas=Image.new('RGB',(1536,len(these)*244),'#202020');draw=ImageDraw.Draw(canvas)
            for row,i in enumerate(these):
                for col,(label,folder) in enumerate(zip(labels,sources)):
                    path=folder/(f'{i}.png' if col==0 else f'{i:05d}.png')
                    with Image.open(path) as im:canvas.paste(im.convert('RGB').resize((384,213)),(col*384,row*244+30))
                    suffix=''
                    if col:
                        name=list(RUNS)[col-1];kind='KEY_POS' if i in keys[name] else 'GENERATED'
                        suffix=f" {kind} L={values[name][i]['lpips_vgg']:.3f}"
                    draw.text((col*384+3,row*244),label,font=font,fill='white')
                    draw.text((col*384+3,row*244+14),f'f{i} {i/24:.3f}s'+suffix,font=font,fill='white')
            path=DEST/f'{group}_{page:02d}.jpg';canvas.save(path,quality=95)
            sheets.append(dict(path=path.name,frames=these,sha256=sha256(path),viewed=False,
                scope='Lossless pre-MP4 PNG; labels display delivered-MP4 LPIPS. KEY_POS denotes output position, not exact received pixels.'))
    result=dict(status='NUMERIC_AUDIT_COMPLETE_VISUAL_REVIEW_PENDING',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source='tv_low_08',verification=verified,conditions=conditions,deltas=deltas,evidence=sheets,
        requested_review_frames=sorted({i for indices in groups.values() for i in indices}),
        review_sampling='Existing source-based frame plan reused, plus metric-selected extremes and previous failures. Current visual review is post hoc, AI-assisted and unblinded.',
        hybrid79_pllava_completed=(SELECTION/'reconstruction/RESULT.json').exists(),
        limitations=['One development source and seed; no independent semantic error ground truth.',
            'Both selection and captions changed; the missing hybrid79+PLLaVA run prevents caption-only attribution.',
            'Reported pipeline times exclude prior keyframe selection and offline AI-caption authoring.',
            'Lower channel use and completed reconstruction do not establish hallucination mitigation.'],
        independent_semantic_review='PENDING',hallucination_mitigation_verified=False,
        input_hashes={'run_result':sha256(ROOT/'RESULT.json'),'report_code':sha256(Path(__file__))})
    write_json(DEST/'analysis.json',result)
    print({'status':result['status'],'stages':verified['stages'],'artifacts':verified['unique_stage_artifacts'],
        'minutes':new['stage_minutes_total'],'deltas':deltas,'common_interior':len(interior),'sheets':len(sheets),
        'review_frames':len(result['requested_review_frames']),'extremes':extremes})


if __name__=='__main__':main()
