"""Matched-source evaluation of the sender pilot, outside receiver inference."""
import csv
import html
import json
from pathlib import Path
import sys
import time

from . import auto_extraction as ae
from .artifacts import sha256, write_json
from .sender_repair_experiment import load_plan, ARMS


def mean_rows(rows, indices, fields):
    import numpy as np
    chosen = [row for row in rows if row['frame'] in indices]
    if len(chosen) != len(indices): raise ValueError('quality frame coverage differs')
    return {k: float(np.mean([row[k] for row in chosen])) for k in fields}


def evaluate(root):
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw
    from .research_quality import read_video
    from .receiver_alignment_evaluation import comparison_video
    from .etri_60s_check import pts_audit
    from .validation_progress import emit
    cv2.setNumThreads(2)
    p=load_plan(root)
    selection=ae.read(root/'selection.json')
    parent=Path(p['parent'])
    runs={'previous':parent/'reconstruction/event_pairs/run', **{a:root/a/'run' for a in ARMS}}
    paths={a:r/'receiver/reconstruction/sample_0000.mp4' for a,r in runs.items()}
    paths={'source':Path(p['source']['normalized_video']),**paths}
    pixels={k:read_video(v) for k,v in paths.items()}
    count=p['source']['frames']
    if any(x.shape != (count,p['source']['height'],p['source']['width'],3) for x in pixels.values()):
        raise ValueError('source/output frame shape mismatch')
    all_indices=set(range(count))
    common_nonkey=all_indices-set(p['original_keys'])-set(selection['keys'])
    metrics=('psnr','ssim','lpips','clip','dists')
    quality={}; transmission={}
    for name,run in runs.items():
        q=ae.read(run/'quality.json')
        if q['video_sha256'] != sha256(paths[name]) or q['source_sha256'] != sha256(paths['source']):
            raise ValueError('quality receipt source/video mismatch')
        with (run/'quality_delivered_mp4.csv').open() as f:
            rows=[{k:int(v) if k=='frame' else float(v) for k,v in row.items()} for row in csv.DictReader(f)]
        rows=[dict(row,psnr=row['psnr_db'],lpips=row['lpips_vgg']) for row in rows]
        quality[name]=dict(all=mean_rows(rows,all_indices,metrics),nonkey=mean_rows(rows,common_nonkey,metrics))
        c=ae.read(run/'channel_accounting.json')
        if c['bit_errors'] or not c['metadata_exact_match']: raise ValueError('metadata transport failure')
        transmission[name]={k:c[k] for k in ('visual_complex_channel_uses','digital_complex_channel_uses','total_complex_channel_uses')}
    # Isolate the flow ablation: exact received keys, captions, generator settings,
    # shapes and actual random tensors. Flow metadata/T5 embeddings intentionally differ.
    left,right=(runs[a] for a in ARMS)
    key_hashes={a:{path.name:sha256(path) for path in sorted((run/'receiver/frames/sample/key_frames_received').glob('*.png'))}
                for a,run in runs.items()}
    paired=dict(received_key_pixels_equal=key_hashes['selector']==key_hashes['selector_flow'],
        captions_equal=ae.read(left/'captions.json')==ae.read(right/'captions.json'),
        random_draws_equal=ae.read(left/'receiver/generation_noise.json')['records']==ae.read(right/'receiver/generation_noise.json')['records'],
        generator_weights_and_settings_equal=True)
    ca,cb=(ae.read(root/a/'execution_protocol.json')['config'] for a in ARMS)
    ca.pop('flow_sampling_policy'); cb.pop('flow_sampling_policy')
    paired['generator_weights_and_settings_equal']=ca==cb
    if not all(paired.values()): raise ValueError(f'flow isolation failed: {paired}')
    shared_keys=set(p['original_keys']) & set(selection['keys'])
    parent_shared=all(key_hashes['previous'][f'{k}.png']==key_hashes['selector'][f'{k}.png'] for k in shared_keys)
    gray={k:[cv2.cvtColor(cv2.resize(frame,(288,160),interpolation=cv2.INTER_AREA),cv2.COLOR_RGB2GRAY) for frame in x]
          for k,x in pixels.items()}
    transitions={k:[] for k in pixels}
    started=time.monotonic()
    for i in range(1,count):
        flow={k:cv2.calcOpticalFlowFarneback(x[i-1],x[i],None,.5,3,15,3,5,1.2,0) for k,x in gray.items()}
        for k,x in gray.items():
            transitions[k].append(dict(frame=i,seconds=i/p['source']['fps'],
                mad=float(np.abs(x[i].astype(float)-x[i-1]).mean()),
                flow_error=float(np.linalg.norm(flow[k]-flow['source'],axis=2).mean()),
                flow_magnitude=float(np.linalg.norm(flow[k],axis=2).mean())))
        if i%48==0 or i==count-1:
            emit(i,count-1)
            print(f'MOTION {i}/{count-1} ({100*i/(count-1):.1f}%) elapsed={time.monotonic()-started:.0f}s',flush=True)
    temporal={}
    for k,rows in transitions.items():
        mad=np.array([r['mad'] for r in rows]); source_mad=np.array([r['mad'] for r in transitions['source']])
        temporal[k]=dict(mad_p95=float(np.percentile(mad,95)),mad_max=float(mad.max()),mad_mean=float(mad.mean()),
            mad_error=float(np.abs(mad-source_mad).mean()),flow_error=float(np.mean([r['flow_error'] for r in rows])),
            flow_magnitude=float(np.mean([r['flow_magnitude'] for r in rows])),
            top_transitions=sorted(rows,key=lambda r:r['mad'],reverse=True)[:5])
    write_json(root/'transitions.json',transitions)
    gate=p['pilot_gate']; decisions={}
    for arm in ARMS:
        prev,cur=quality['previous']['nonkey'],quality[arm]['nonkey']
        checks=dict(flow=temporal[arm]['flow_error'] <= temporal['previous']['flow_error']*gate['flow_error_ratio_max'],
            jump=temporal[arm]['mad_p95'] <= temporal['previous']['mad_p95']*gate['mad_p95_ratio_max'],
            psnr=cur['psnr'] >= prev['psnr']-gate['nonkey_psnr_drop_max_db'],
            lpips=cur['lpips'] <= prev['lpips']+gate['nonkey_lpips_increase_max'],
            transmission=transmission[arm]['total_complex_channel_uses'] <= transmission['previous']['total_complex_channel_uses']*gate['transmission_ratio_max'])
        decisions[arm]=dict(checks=checks,numerical_pilot_gate_passed=all(checks.values()),
            adopted_as_general_default=False)
    samples=root/'samples'; samples.mkdir(exist_ok=True)
    indices=sorted(set(range(0,count,48)) | {count-1} |
        {r['frame'] for arm in runs for r in temporal[arm]['top_transitions'][:3]} |
        {r['frame']-1 for arm in runs for r in temporal[arm]['top_transitions'][:3]})
    names=list(pixels); sheets=[]
    width,height=384,214
    for batch in range(0,len(indices),3):
        ids=indices[batch:batch+3]
        canvas=Image.new('RGB',(width*len(names),(height+24)*len(ids)),'white')
        draw=ImageDraw.Draw(canvas)
        for row,index in enumerate(ids):
            for col,name in enumerate(names):
                canvas.paste(Image.fromarray(pixels[name][index]).resize((width,height)),(col*width,row*(height+24)+24))
                draw.text((col*width+4,row*(height+24)+4),f'{name} f={index} t={index/24:.3f}s',fill='black')
        path=samples/f'{batch//3:02d}.jpg'; canvas.save(path,quality=94); sheets.append(path)
    comparison_video(root/'comparison.mp4',list(pixels.values()),names,p['source']['fps'])
    pts_audit(root/'comparison.mp4',count,p['source']['fps'])
    result=dict(status='COMPLETE_QUANTITATIVE_PILOT_VISUAL_REVIEW_PENDING',signature=p['signature'],
        source_frames=count,fps=p['source']['fps'],key_count=dict(previous=len(p['original_keys']),candidate=len(selection['keys'])),
        candidate_keys=selection['keys'],common_nonkey_frames=len(common_nonkey),quality=quality,temporal=temporal,
        transmission=transmission,flow_ablation_verified=paired,parent_shared_received_keys_equal=parent_shared,
        decisions=decisions,training_executed=False,default_policy_changed=False,
        independent_semantic_review='PENDING',hallucination_mitigation_verified=False,
        sampled_visual_indices=indices,video_sha256={k:sha256(v) for k,v in paths.items()},
        scope='One 16-second development video, one seed and SNR. No held-out generalization claim.',
        videos={k:str(v) for k,v in paths.items()},report=str(root/'report.html'),comparison=str(root/'comparison.mp4'))
    write_json(root/'results.json',result)
    table=''
    for name in runs:
        q=quality[name]['nonkey']; t=temporal[name]; c=transmission[name]
        table+=f'<tr><td>{name}</td><td>{q["psnr"]:.3f}</td><td>{q["lpips"]:.5f}</td><td>{t["flow_error"]:.6f}</td><td>{t["mad_p95"]:.3f}</td><td>{c["total_complex_channel_uses"]:,}</td></tr>'
    doc='''<!doctype html><meta charset="utf-8"><title>Sender repair pilot</title>
<style>body{font-family:sans-serif;max-width:1600px;margin:2em auto}td,th{padding:.6em;border:1px solid #ccc}table{border-collapse:collapse}video,img{width:100%}pre{white-space:pre-wrap}</style>
<h1>Source-reviewed capped keys + motion sampling pilot</h1>
<p>Left to right: source / previous Qwen / selector / selector + flow. Same 384-frame input, 24 FPS, AWGN 10 dB. New captions are required for changed intervals; selector comparison is an end-to-end sender comparison. Flow comparison preserves keys, captions and actual random tensors.</p>
<p>Experimental: one development video. Independent semantic review and held-out generalization remain pending. No default replacement.</p>
<video controls src="comparison.mp4"></video><table><tr><th>Arm</th><th>Common nonkey PSNR ↑</th><th>LPIPS ↓</th><th>Motion error ↓</th><th>MAD p95 ↓</th><th>Channel uses ↓</th></tr>'''+table+'</table>'
    doc+='<h2>Predeclared numerical pilot gates</h2><pre>'+html.escape(json.dumps(decisions,indent=2))+'</pre>'
    doc+='<h2>Source event review (same Qwen, not independent)</h2><pre>'+html.escape(json.dumps(selection,ensure_ascii=False,indent=2))+'</pre>'
    doc+='<h2>Regular samples and largest transitions; visual judgment pending</h2>'
    doc+=''.join(f'<img loading="lazy" src="samples/{path.name}">' for path in sheets)
    (root/'report.html').write_text(doc)
    write_json(root/'status.json',dict(status=result['status']))


if __name__=='__main__': evaluate(Path(sys.argv[1]).resolve())
