"""Offline evaluation for the shared-grid sequential/joint receiver pilot."""
import gc
import html
from pathlib import Path
import subprocess
import time

from .artifacts import sha256, write_json
from .webvid5 import read_json
from .research_quality import read_video, read_frames
from .official_quality import OfficialMetrics, summarize
from .receiver_alignment_evaluation import temporal_scores, comparison_video

LABELS = ['Source (evaluation only)', 'Legacy receiver (frozen)', 'Shared grid / sequential', 'Shared grid / joint']


def evaluate(output, protocol, signature):
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    cv2.setNumThreads(2);torch.set_num_threads(2)
    gc.collect();torch.cuda.empty_cache()
    start = time.monotonic()
    sources, generated, groups, input_hashes = [], [], [], {}
    source_by_case = {}
    for case in protocol['cases']:
        source_path = Path(case['run'])/'data/normalized.mp4'
        source = read_video(source_path)[case['keys'][0]:case['keys'][-1]+1]
        input_hashes[str(source_path)] = sha256(source_path)
        source_by_case[case['name']] = source
        for arm in protocol['arms']:
            pixels = read_frames(output/case['name']/arm/'frames')
            if pixels.shape != source.shape or len(pixels) != 49:
                raise ValueError('evaluation window mismatch')
            sources.append(source);generated.append(pixels);groups.append((case,arm))
    def progress(done,total):
        if done == 1 or done%49 == 0 or done == total:
            print(f'QUALITY {done}/{total} ({100*done/total:.1f}%) elapsed={time.monotonic()-start:.0f}s',flush=True)
    _, scores = OfficialMetrics().evaluate(np.concatenate(sources),np.concatenate(generated),progress)
    rows = []
    for i, ((case,arm),source,pixels) in enumerate(zip(groups,sources,generated)):
        per_frame = [dict(r,frame=j) for j,r in enumerate(scores[i*49:(i+1)*49])]
        nonkeys = [r for r in per_frame if r['frame'] not in {0,24,48}]
        temporal = temporal_scores(source,pixels,25)
        source_mad = np.asarray(temporal['per_transition']['source_mad'])
        out_mad = np.asarray(temporal['per_transition']['generated_mad'])
        temporal['mad_error_mean'] = float(np.abs(source_mad-out_mad).mean())
        temporal['boundary_transitions'] = {str(k):dict(source_mad=float(source_mad[k-1]),
            generated_mad=float(out_mad[k-1]),absolute_error=float(abs(out_mad[k-1]-source_mad[k-1]))) for k in [17,25,34]}
        row = dict(case=case['name'],arm=arm,all_frames=summarize(per_frame,source.shape),
            nonkey_frames=summarize(nonkeys,(46,*source.shape[1:])),temporal=temporal)
        rows.append(row)
        write_json(output/case['name']/arm/'metrics.json',dict(row,per_frame=per_frame))
        print(f'EVALUATE {i+1}/12 ({100*(i+1)/12:.1f}%) {case["name"]}/{arm}: '
              f'nonkey PSNR={row["nonkey_frames"]["psnr_db"]:.3f} LPIPS={row["nonkey_frames"]["lpips_vgg"]:.4f} '
              f'flow error={temporal["flow_error_mean"]:.4f} MAD error={temporal["mad_error_mean"]:.4f}',flush=True)
    del sources,generated
    comparisons,probes,pair_audits = {},{},{}
    body = ["<!doctype html><html lang='en'><meta charset='utf-8'><title>Joint receiver windows</title>",
        "<style>body{font:15px system-ui;max-width:1400px;margin:24px auto}video{width:100%}td,th{border:1px solid #ccc;padding:6px}table{border-collapse:collapse}</style>",
        "<h1>Receiver overlapping-window pilot</h1><p>Experimental, not adopted. No training or output pixel interpolation.</p>",
        "<p>Shared sequential vs shared joint isolates co-denoising on identical grids, received key conditions, noise, prompts, schedule and 60 model calls. "
        "Legacy is a frozen descriptive reference: it has a different grid and noise tensor shape.</p>",
        "<p>Four restarted 49-frame windows, one seed. Independent semantic review pending. Lower frame changes can reflect freezing.</p>"]
    for case in protocol['cases']:
        name=case['name'];directory=output/name
        shared=directory/'paired_inputs.json';digest=sha256(shared)
        traces=[read_json(directory/a/'sampling_trace.json') for a in ['shared_sequential','shared_joint']]
        for trace in traces:
            if trace['paired_input_sha256'] != digest or not trace['rng_unchanged'] or len(trace['calls']) != 60 or not trace['received_anchor_preserved']:
                raise ValueError('paired input/noise/compute verification failed')
        pair_audits[name]=dict(shared_inputs_sha256=digest,same_noise_and_keys=True,same_model_calls=60,
                               rng_unchanged=True,received_anchor_preserved=True)
        arrays=[source_by_case[name]]+[read_frames(directory/a/'frames') for a in protocol['arms']]
        comparison_video(directory/'comparison.mp4',arrays,LABELS,24)
        comparisons[name]=dict(path=str(directory/'comparison.mp4'),sha256=sha256(directory/'comparison.mp4'))
        write_json(directory/'panel_sources.json',dict(order=LABELS,source=case['run']+'/data/normalized.mp4',
            source_frame_range=[case['keys'][0],case['keys'][-1]],
            generated_frames=[str(directory/a/'frames') for a in protocol['arms']],
            panel_pixels='direct generated PNGs before independent H264 comparison encoding'))
        for frame in [12,16,17,24,25,33,34,36,48]:
            panels=[]
            for array,label in zip(arrays,LABELS):
                panel=np.zeros((344,576,3),dtype=np.uint8);panel[24:]=array[frame]
                cv2.putText(panel,f'{label} / {frame}',(8,17),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
                panels.append(panel)
            Image.fromarray(np.hstack(panels)).save(directory/f'comparison_frame_{frame:02d}.png')
        for video in [directory/a/'reconstruction.mp4' for a in protocol['arms']]+[directory/'comparison.mp4']:
            probe=read_probe(video)
            probes[str(video.relative_to(output))]=probe
        body.append(f'<h2>{html.escape(name)}: {html.escape(case["scene"])}</h2><video controls loop src="{name}/comparison.mp4"></video>'
                    '<p>Source / legacy receiver / shared sequential / shared joint</p>'
                    '<table><tr><th>Arm</th><th>Nonkey PSNR ↑</th><th>Nonkey LPIPS ↓</th><th>Flow error ↓</th><th>MAD error ↓</th></tr>')
        for row in rows:
            if row['case']==name:
                body.append(f'<tr><td>{row["arm"]}</td><td>{row["nonkey_frames"]["psnr_db"]:.3f}</td>'
                    f'<td>{row["nonkey_frames"]["lpips_vgg"]:.4f}</td><td>{row["temporal"]["flow_error_mean"]:.4f}</td>'
                    f'<td>{row["temporal"]["mad_error_mean"]:.4f}</td></tr>')
        body.append('</table>')
    body.append('</html>');(output/'report.html').write_text('\n'.join(body))
    result=dict(status='PILOT_COMPLETE_INDEPENDENT_REVIEW_PENDING',signature=signature,adopted=False,
        training_executed=False,source_pixels_used_for_generation=False,output_pixel_interpolation=False,
        paired_arms=['shared_sequential','shared_joint'],legacy_is_noise_matched=False,rows=rows,
        pair_audits=pair_audits,probes=probes,comparisons=comparisons,evaluation_source_hashes=input_hashes,
        evaluation_seconds=time.monotonic()-start,evaluation_code_sha256=sha256(Path(__file__)))
    write_json(output/'results.json',result)
    write_json(output/'status.json',{k:result[k] for k in ['status','signature','adopted']})
    print(f'DONE {result["status"]}: {output/"report.html"}',flush=True)


def read_probe(path):
    import json
    result=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_frames',
        '-show_entries','stream=width,height,r_frame_rate,nb_frames,duration:frame=best_effort_timestamp_time',
        '-of','json',str(path)]))
    stream=result['streams'][0];pts=[float(f['best_effort_timestamp_time']) for f in result['frames']]
    if len(pts)!=49 or stream['r_frame_rate']!='24/1' or any(abs(t-i/24)>1e-6 for i,t in enumerate(pts)):
        raise ValueError('invalid MP4 frame count or timestamps')
    return dict(stream,sha256=sha256(path),pts_24fps_passed=True)
