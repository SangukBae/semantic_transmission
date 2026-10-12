"""Offline measurements and review artifacts for the learned-motion pilot."""
import html
from pathlib import Path
import time

from .artifacts import sha256,write_json
from .webvid5 import read_json


def evaluate(root,p):
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from .bsu_motion_experiment import source_window,BASE,OLD,ARMS,progress
    from .research_quality import read_frames
    from .official_quality import OfficialMetrics,summarize
    from .receiver_alignment_experiment import complete_arm
    from .receiver_alignment_evaluation import temporal_scores,comparison_video
    torch.set_num_threads(2);cv2.setNumThreads(2)
    start=time.monotonic();results=[];cases=[c for c in p['cases'] if c['split']=='test']
    total=sum(len(ARMS)+(c['name']=='zea') for c in cases);done=0
    for c in cases:
        source=source_window(c['source_video'],*c['keys'])
        arrays=[source];labels=['Source']
        arms=ARMS+(['pllava_caption'] if c['name']=='zea' else [])
        for arm in arms:
            directory=root/'generation'/c['name']/arm
            if not complete_arm(directory,p['signature']):raise ValueError('generation incomplete or changed')
            pixels=read_frames(directory/'frames')
            receipt=directory/'evaluation.json'
            if receipt.exists():
                row=read_json(receipt)
                if row['signature']!=p['signature']:raise ValueError('stale evaluation')
            else:
                def tick(count,n):
                    if count==1 or count%384==0 or count==n:
                        print(f'QUALITY {c["name"]}/{arm} {count}/{n} ({100*count/n:.1f}%)',flush=True)
                summary,per_frame=OfficialMetrics().evaluate(source,pixels,progress=tick)
                temporal=temporal_scores(source,pixels,1)
                mad=temporal['per_transition']['generated_mad']
                temporal.update(mad_p95=float(np.percentile(mad,95)),mad_max=float(np.max(mad)),
                    interpretation='no internal key boundary; seam fields unused',resize='288x160 bilinear')
                row=dict(case=c['name'],arm=arm,signature=p['signature'],frames=len(source),all_frames=summary,
                    nonkeys=summarize(per_frame[1:-1],source[1:-1].shape),temporal=temporal,
                    generated_sha256=sha256(directory/'reconstruction.mp4'))
                write_json(directory/'per_frame.json',per_frame);write_json(receipt,row)
            results.append(row);arrays.append(pixels);labels.append(arm);done+=1
            progress('EVALUATE completed arms',done,total,start)
        directory=root/'review'/c['name'];directory.mkdir(parents=True,exist_ok=True)
        comparison_video(directory/'comparison.mp4',arrays,labels,p['generator']['fps'])
        selected=np.linspace(0,len(source)-1,9).round().astype(int)
        for i in selected:
            panels=[]
            for pixels,label in zip(arrays,labels):
                panel=np.zeros((344,576,3),dtype=np.uint8);panel[24:]=pixels[i]
                cv2.putText(panel,f'{label} frame={i}',(8,17),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
                panels.append(panel)
            Image.fromarray(np.concatenate(panels,axis=1)).save(directory/f'frame_{i:04d}.png')
    rows={(r['case'],r['arm']):r for r in results};gates=[]
    gate=p['adoption_gate']
    for c in cases:
        base=rows[c['name'],'baseline'];new=rows[c['name'],'motion_learned']
        checks=dict(psnr=new['all_frames']['psnr_db']-base['all_frames']['psnr_db']>=gate['psnr_gain_min'],
            lpips=new['all_frames']['lpips_vgg']-base['all_frames']['lpips_vgg']<=gate['lpips_increase_max'],
            flow=new['temporal']['flow_error_mean']<=base['temporal']['flow_error_mean']*gate['flow_ratio_max'],
            motion_spikes=new['temporal']['mad_p95']<=base['temporal']['mad_p95']*gate['mad_p95_ratio_max'])
        gates.append(dict(case=c['name'],checks=checks,passed=all(checks.values())))
    # Decoder parity is checked independently of the paired caption/motion results.
    replay=read_frames(root/'generation/zea/baseline/frames')
    historic=read_frames(BASE/'receiver/reconstruction/sample_0000_frames')
    parity=dict(exact_pixels=bool(np.array_equal(replay,historic)),max_absolute_error=int(np.abs(replay.astype(int)-historic.astype(int)).max()),
        mean_absolute_error=float(np.abs(replay.astype(float)-historic).mean()),actual_noise_matched=True)
    channel=read_json(root/'channel.json');full=next(r for r in channel['reports'] if r['filename']=='full_motion.bin')
    channel_summary=dict(p['channel'],motion_digital_channel_uses=full['complex_channel_uses'],
        motion_total_channel_uses=p['channel']['visual_complex_channel_uses']+full['complex_channel_uses'],
        ldpc_blocks=full['blocks'],bit_errors=full['bit_errors'],remaining_padding_bytes=full['padding_bits']//8,
        physical_link_headers_included=False,visual_stream_reused_exactly=True,
        short_diagnostics='motion-only transport tested; not full end-to-end rate benchmarks')
    # Reevaluate fixed22 and LGVSC on the same first381 PNG boundary for context.
    contextual=[]
    for name,path in [('historical_lgvsc',OLD/'lgvsc/run/receiver/reconstruction/sample_0000_frames'),
                      ('historical_fixed22',OLD/'qwen/reconstruction/event_pairs/run/receiver/reconstruction/sample_0000_frames')]:
        original=read_frames(path)
        contextual.append(dict(name=name,frames=len(original),note='historical different keys and noise; descriptive comparator only'))
    zea_common=[]
    for arm in ARMS+['pllava_caption']:
        per=read_json(root/'generation/zea'/arm/'per_frame.json')
        zea_common.append(dict(arm=arm,common381=summarize(per[:381],(381,320,576,3))))
    result=dict(status='COMPLETE_PILOT_INDEPENDENT_SEMANTIC_REVIEW_PENDING',signature=p['signature'],adopted=False,
        quantitative_gate_passed=all(g['passed'] for g in gates),adoption_gates=gates,baseline_replay=parity,
        channel=channel_summary,training=read_json(root/'training.json'),cases=results,zea_common381=zea_common,
        historical_comparator_scope=contextual,
        limitations=['two training originals and one validation original; limited training diversity',
            'one full harbor development video and three short diagnostic clips; one seed/SNR',
            'latent-prior supervision, not end-to-end STDiT adapter training',
            'coarse 2-D camera/regional motion cannot represent unseen surfaces or object identities',
            'same information budget does not prove same reconstruction quality',
            'independent semantic event/error annotation remains pending'])
    write_json(root/'RESULT.json',result)
    body=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>BSU motion pilot</title>',
        '<style>body{font:15px system-ui;max-width:1400px;margin:30px auto}video{width:100%}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc}</style>',
        '<h1>BSU: 동일 난수 캡션 비교 · 압축 움직임 어댑터</h1><p>실험 후보. 기본 모델 미적용. 독립 의미 검수 대기.</p>',
        f'<p>전체 항구 영상 전송량: {channel_summary["motion_total_channel_uses"]:,} complex channel uses; '
        f'packet {channel_summary["motion_packet_bytes"]} bytes / {channel_summary["capacity_bytes"]} byte capacity.</p>']
    for c in cases:
        body.append(f'<h2>{html.escape(c["name"])}</h2><video controls loop src="review/{c["name"]}/comparison.mp4"></video>'
            '<p>Source / baseline / motion_zero / motion_learned / (Zea only) pllava_caption</p>'
            '<table><tr><th>Arm</th><th>PSNR ↑</th><th>LPIPS ↓</th><th>Flow error ↓</th><th>MAD p95</th></tr>')
        for r in results:
            if r['case']==c['name']:
                body.append(f'<tr><td>{r["arm"]}</td><td>{r["all_frames"]["psnr_db"]:.3f}</td>'
                    f'<td>{r["all_frames"]["lpips_vgg"]:.4f}</td><td>{r["temporal"]["flow_error_mean"]:.4f}</td><td>{r["temporal"]["mad_p95"]:.3f}</td></tr>')
        body.append('</table>')
    body.append('</html>');(root/'report.html').write_text('\n'.join(body))
    print('EVALUATION complete; inspect RESULT.json and report.html before adoption',flush=True)
