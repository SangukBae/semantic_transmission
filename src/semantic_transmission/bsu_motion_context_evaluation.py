"""Compare historical and new reconstructions at the same pre-MP4 boundary."""
import html
from pathlib import Path

from .artifacts import sha256,write_json
from .webvid5 import read_json
from .bsu_motion_experiment import DEFAULT,BASE,OLD,load,source_window


def main():
    import numpy as np
    import torch
    from .official_quality import OfficialMetrics,summarize
    from .research_quality import read_frames
    from .receiver_alignment_evaluation import temporal_scores
    root=DEFAULT;p=load(root);torch.set_num_threads(2)
    source=source_window(BASE/'data/normalized.mp4',0,380)
    rows=[]
    names=['baseline','pllava_caption','motion_zero','motion_geometry','motion_learned']
    for name in names:
        directory=root/'generation/zea'/name
        if name=='motion_geometry':
            summary=read_json(directory/'evaluation.json')['common381']
        else:
            frames=read_json(directory/'per_frame.json')[:381]
            summary=summarize(frames,source.shape)
        pixels=read_frames(directory/'frames')[:381]
        temporal=temporal_scores(source,pixels,1)
        temporal['mad_p95']=float(np.percentile(temporal['per_transition']['generated_mad'],95))
        channel=read_json(BASE/'channel_accounting.json')['total_complex_channel_uses']
        if name=='pllava_caption':
            digital=next(r for r in read_json(root/'channel.json')['reports'] if r['filename']=='caption_only.bin')
            channel=p['channel']['visual_complex_channel_uses']+digital['complex_channel_uses']
        rows.append(dict(name=name,keys=2,quality=summary,temporal=temporal,channel_uses=channel,
            paired_actual_noise=True,generation_receipt_sha256=sha256(directory/'result.json')))
    for name,run,keys in [('historical_lgvsc',OLD/'lgvsc/run',2),
                          ('historical_fixed22',OLD/'qwen/reconstruction/event_pairs/run',22)]:
        path=run/'receiver/reconstruction/sample_0000_frames'
        pixels=read_frames(path)[:381]
        cache=root/f'{name}_png_metrics.json'
        frame_hashes={str(f):sha256(f) for f in sorted(path.glob('*.png'))[:381]}
        if cache.exists():
            q=read_json(cache)
            if q['frame_hashes']!=frame_hashes:raise ValueError('historical frames changed')
            summary=q['quality']
        else:
            summary,per_frame=OfficialMetrics().evaluate(source,pixels)
            write_json(cache,dict(quality=summary,frame_hashes=frame_hashes,per_frame=per_frame))
        temporal=temporal_scores(source,pixels,1)
        temporal['mad_p95']=float(np.percentile(temporal['per_transition']['generated_mad'],95))
        rows.append(dict(name=name,keys=keys,quality=summary,temporal=temporal,
            channel_uses=read_json(run/'channel_accounting.json')['total_complex_channel_uses'],
            paired_actual_noise=False,scope='descriptive context; original key positions, intervals and captions retained'))
        print(f'CONTEXT measured {name} on common 381 PNG frames',flush=True)
    lookup={r['name']:r for r in rows};learned=lookup['motion_learned'];fixed=lookup['historical_fixed22']
    target=dict(channel_at_or_below_lgvsc=learned['channel_uses']<=lookup['historical_lgvsc']['channel_uses'],
        psnr_at_or_above_fixed22=learned['quality']['psnr_db']>=fixed['quality']['psnr_db'],
        lpips_at_or_below_fixed22=learned['quality']['lpips_vgg']<=fixed['quality']['lpips_vgg'])
    result=dict(status='COMPLETE_MATCHED_PNG_CONTEXT',frames=list(range(381)),
        source_sha256=sha256(BASE/'data/normalized.mp4'),boundary='lossless pre-MP4 PNG; previous published table used delivered MP4',
        rows=rows,target=target,code_sha256=sha256(Path(__file__)),
        paired_caption_scope='baseline vs pllava_caption, same received [0,383] and actual recorded noise',
        paired_motion_scope='same received keys/caption/noise; extra motion bytes remain within the same three LDPC blocks',
        independent_semantic_review='PENDING',adopted=False)
    write_json(root/'comparison.json',result)
    body=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>BSU motion analysis</title>',
        '<style>body{font:15px system-ui;max-width:1400px;margin:30px auto}video{width:100%}td,th{padding:8px;border:1px solid #ccc}table{border-collapse:collapse}</style>',
        '<h1>BSU 캡션 · 움직임 · 학습 기여도 비교</h1>',
        '<p>같은 원본 0~380번, MP4 저장 전 PNG. 기존 문서의 MP4 수치와 직접 혼합하지 않는다. 역사적 LGVSC·22키는 키 위치·실제 난수가 달라 참고 비교다.</p>',
        '<table><tr><th>조건</th><th>키</th><th>전송량</th><th>PSNR ↑</th><th>LPIPS ↓</th><th>광류 오차 ↓</th><th>MAD p95</th></tr>']
    for r in rows:
        body.append(f'<tr><td>{html.escape(r["name"])}</td><td>{r["keys"]}</td><td>{r["channel_uses"]:,}</td>'
            f'<td>{r["quality"]["psnr_db"]:.3f}</td><td>{r["quality"]["lpips_vgg"]:.4f}</td>'
            f'<td>{r["temporal"]["flow_error_mean"]:.4f}</td><td>{r["temporal"]["mad_p95"]:.3f}</td></tr>')
    body.extend(['</table>','<p><a href="report.html">전체 구간·다른 장면의 대조 결과</a> · <a href="comparison.json">공통 조건 수치</a> · <a href="geometry_ablation_result.json">CNN 기여도</a></p>',
        '<h2>원본 / 기본 BSU / 움직임 0 / 움직임+학습 / PLLaVA 캡션</h2><video controls loop src="review/zea/comparison.mp4"></video>',
        '<h2>원본 / 움직임 정렬만 / 움직임 정렬+CNN 보정</h2><video controls loop src="review/zea/geometry_vs_learned.mp4"></video>',
        '<p>독립 의미 검수 대기. 항구·짧은 3개 개발 장면, 단일 seed/SNR 결과다. 모델 기본 설정에 채택하지 않았다.</p></html>'])
    (root/'analysis.html').write_text('\n'.join(body))
    print('CONTEXT all comparisons complete',flush=True)


if __name__=='__main__':main()
