"""Offline counterfactual labels and source-excluded gate evaluation."""
import html
from pathlib import Path
import time

from .artifacts import sha256,write_json
from .webvid5 import read_json


def motion(source,pixels,start=1):
    import cv2
    import numpy as np
    def gray(frames):
        return [cv2.cvtColor(cv2.resize(f,(288,160),interpolation=cv2.INTER_AREA),cv2.COLOR_RGB2GRAY) for f in frames]
    a,b=gray(source),gray(pixels);rows=[]
    for i in range(1,len(a)):
        flows=[cv2.calcOpticalFlowFarneback(x[i-1],x[i],None,.5,3,15,3,5,1.2,0) for x in (a,b)]
        rows.append(dict(frame=i,flow_error=float(np.linalg.norm(flows[0]-flows[1],axis=2).mean()),
            source_mad=float(np.abs(a[i].astype(float)-a[i-1]).mean()),
            generated_mad=float(np.abs(b[i].astype(float)-b[i-1]).mean()),
            generated_motion=float(np.linalg.norm(flows[1],axis=2).mean())))
    part=[r for r in rows if r['frame']>=start]
    return dict(flow_error=float(np.mean([r['flow_error'] for r in part])),
        mad_mean=float(np.mean([r['generated_mad'] for r in part])),
        mad_p95=float(np.percentile([r['generated_mad'] for r in part],95)),
        mad_max=max(r['generated_mad'] for r in part),
        mad_error=float(np.mean([abs(r['generated_mad']-r['source_mad']) for r in part])),
        generated_motion=float(np.mean([r['generated_motion'] for r in part])),rows=rows)


def evaluate(root,split):
    import cv2
    import numpy as np
    import torch
    from PIL import Image,ImageDraw
    from .receiver_gate_experiment import load,arm_complete
    from .research_quality import read_video,read_frames
    from .official_quality import OfficialMetrics,summarize
    from .receiver_alignment_evaluation import comparison_video
    from .etri_60s_check import pts_audit
    from .validation_progress import emit
    p=load(root);signature=p['signature'];cv2.setNumThreads(2);torch.set_num_threads(2)
    cases=[c for c in p['cases'] if c['split']==split]
    arms=['baseline','constant_edit'] if split=='train' else p['arms']
    originals={};sources=[];outputs=[];groups=[];raw_prefix_checks={}
    for c in cases:
        if c['source_video'] not in originals:originals[c['source_video']]=read_video(c['source_video'])
        source=originals[c['source_video']][c['keys'][0]:c['keys'][-1]+1]
        for arm in arms:
            directory=root/split/c['name']/arm
            if not arm_complete(directory,signature):raise ValueError('incomplete generation')
            pixels=read_frames(directory/'frames')
            if source.shape!=pixels.shape:raise ValueError('source frame correspondence differs')
            prefix=c['keys'][1]-c['keys'][0]+1
            if arm=='baseline':baseline_pixels=pixels
            elif not np.array_equal(baseline_pixels[:prefix],pixels[:prefix]):raise ValueError('shared first segment pixels changed')
            noise=read_json(directory/'generation_noise.json')
            if arm!='baseline' and not noise['matched_reference']:raise ValueError('counterfactual noise not matched')
            sources.append(source);outputs.append(pixels);groups.append((c,arm,len(source)))
        raw_prefix_checks[c['name']]=True
    metric=OfficialMetrics();started=time.monotonic()
    def progress(n,total):
        emit(n,total)
        if n==1 or n%196==0 or n==total:
            print(f'QUALITY {split} {n}/{total} ({100*n/total:.1f}%) elapsed={time.monotonic()-started:.0f}s',flush=True)
    _,all_rows=metric.evaluate(np.concatenate(sources),np.concatenate(outputs),progress)
    result={};offset=0
    for j,((c,arm,n),source,pixels) in enumerate(zip(groups,sources,outputs)):
        rows=[dict(r,frame=i) for i,r in enumerate(all_rows[offset:offset+n])];offset+=n
        keys=[k-c['keys'][0] for k in c['keys']];start=keys[1]+1
        nonkey=[r for r in rows if r['frame'] not in keys]
        affected=[r for r in nonkey if r['frame']>=start]
        temporal=motion(source,pixels,start)
        value=dict(nonkey=summarize(nonkey,(len(nonkey),*source.shape[1:])),
            affected=dict(summarize(affected,(len(affected),*source.shape[1:])),**{k:v for k,v in temporal.items() if k!='rows'}),
            temporal_rows=temporal['rows'],decision=read_json(root/split/c['name']/arm/'gate_decision.json'),
            output_sha256=sha256(root/split/c['name']/arm/'reconstruction.mp4'))
        result.setdefault(c['name'],{})[arm]=value
        print(f'SCORE {j+1}/{len(groups)} ({100*(j+1)/len(groups):.1f}%) {c["name"]}/{arm}: '
            f'PSNR={value["affected"]["psnr_db"]:.3f} LPIPS={value["affected"]["lpips_vgg"]:.4f} flow={value["affected"]["flow_error"]:.5f}',flush=True)
    report=dict(status='EVALUATED',signature=signature,split=split,cases=result,
        evaluated_domain='lossless pre-MP4 PNG; delivered videos separately validated, not scored as PNG',
        affected_domain='second segment non-key frames for fidelity; seam and second-segment transitions for motion',
        shared_first_segment_pixels_equal=raw_prefix_checks,source_pixels_used_for_generation=False,
        source_pixels_used_for_offline_evaluation=True,independent_semantic_review='PENDING')
    write_json(root/f'{split}_metrics.json',report)
    if split=='train':return
    decisions={};gate=p['adoption_gate'];comparisons=root/'comparisons';comparisons.mkdir(exist_ok=True)
    body=['<!doctype html><meta charset="utf-8"><title>Learned receiver history gate</title>',
        '<style>body{font:15px system-ui;max-width:1600px;margin:2em auto}td,th{border:1px solid #aaa;padding:.5em}table{border-collapse:collapse}video,img{width:100%}</style>',
        '<h1>Learned receiver history gate: first pilot</h1><p>Source / baseline / constant edit / learned gate. '
        'Generator fixed. Training: 12 windows from 3 original videos. Evaluation: 4 other known development scenes. '
        'Same received keys/captions/random tensors. No new transmission or output pixel interpolation. '
        'Metrics below use lossless generation frames; source images are only used in offline evaluation and teacher labels.</p>']
    for c in cases:
        name=c['name'];before=result[name]['baseline']['affected'];candidate=result[name]['learned_gate']['affected']
        checks=dict(motion=candidate['flow_error']<=before['flow_error']*gate['flow_ratio_max'],
            psnr=candidate['psnr_db']>=before['psnr_db']-gate['psnr_drop_max'],
            lpips=candidate['lpips_vgg']<=before['lpips_vgg']+gate['lpips_increase_max'],
            worst_transition=candidate['mad_max']<=before['mad_max']*gate['mad_max_ratio_max'])
        decisions[name]=dict(checks=checks,passed=all(checks.values()),
            chosen_ratio=result[name]['learned_gate']['decision']['ratio'],predicted_utility=result[name]['learned_gate']['decision']['prediction'])
        source=originals[c['source_video']][c['keys'][0]:c['keys'][-1]+1]
        # Comparisons display delivered MP4s; quantitative labels remain explicitly PNG.
        arrays=[source]+[read_video(root/split/name/a/'reconstruction.mp4') for a in arms]
        out=comparisons/name;out.mkdir(exist_ok=True)
        comparison_video(out/'comparison.mp4',arrays,['Source','Baseline','Constant edit .35','Learned gate'],p['generator']['fps'])
        pts_audit(out/'comparison.mp4',len(source),p['generator']['fps'])
        seam=c['keys'][1]-c['keys'][0]+1
        largest=max(result[name]['learned_gate']['temporal_rows'],key=lambda r:r['generated_mad'])['frame']
        indices=sorted({0,seam-1,seam,min(seam+10,len(source)-1),len(source)-1,max(0,largest-1),largest})
        canvas=Image.new('RGB',(384*4,238*len(indices)),'white');draw=ImageDraw.Draw(canvas)
        for row,i in enumerate(indices):
            for col,(array,label) in enumerate(zip(arrays,['Source',*arms])):
                canvas.paste(Image.fromarray(array[i]).resize((384,214)),(col*384,row*238+24))
                draw.text((col*384+4,row*238+4),f'{label} frame {c["keys"][0]+i}',fill='black')
        canvas.save(out/'samples.jpg',quality=94)
        write_json(out/'sample_indices.json',dict(local=indices,source=[c['keys'][0]+i for i in indices],selection='seam, regular and maximum generated MAD'))
        body.append(f'<h2>{html.escape(name)}</h2><video controls src="comparisons/{name}/comparison.mp4"></video>'
            '<table><tr><th>Arm</th><th>Affected nonkey PSNR ↑</th><th>LPIPS ↓</th><th>Motion error ↓</th><th>Max MAD ↓</th></tr>')
        for arm in arms:
            m=result[name][arm]['affected']
            body.append(f'<tr><td>{arm}</td><td>{m["psnr_db"]:.3f}</td><td>{m["lpips_vgg"]:.5f}</td><td>{m["flow_error"]:.6f}</td><td>{m["mad_max"]:.3f}</td></tr>')
        body.append(f'</table><p>Gate decision: {html.escape(str(decisions[name]))}</p><img src="comparisons/{name}/samples.jpg">')
    final=dict(status='TRAINED_AND_EVALUATED_VISUAL_REVIEW_PENDING',signature=signature,training_executed=True,
        training=read_json(root/'training.json'),decisions=decisions,adopted=False,default_receiver_changed=False,
        source_pixels_used_for_generation=False,output_pixel_interpolation=False,additional_transmitted_information=False,
        all_counterfactual_noise_matched=True,shared_first_segment_pixels_equal=raw_prefix_checks,
        independent_semantic_review='PENDING',hallucination_mitigation_verified=False,
        scope=p['scope'],metric_domain=report['evaluated_domain'],report=str(root/'report.html'),
        limitations=['Small counterfactual utility MLP; not end-to-end temporal adapter training.',
            'One seed, AWGN 10 dB, two segments restarted at first key; no long-range or arbitrary-SNR claim.',
            'Evaluation originals excluded from training, but these are previously examined development scenes, not an untouched final benchmark.',
            'No independent Added/Missing/Distorted ground truth.'])
    write_json(root/'results.json',final)
    body.append('<p>Not adopted as default. Visual sample review and independent semantic review are separate.</p>')
    (root/'report.html').write_text('\n'.join(body))
