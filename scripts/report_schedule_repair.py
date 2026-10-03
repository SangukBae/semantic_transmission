"""Render audited metrics and explicitly recorded visual review; no inference."""
import argparse
import csv
import html
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.webvid5 import read_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    root=args.output.resolve()/'analysis'
    a=read_json(root/'analysis.json')
    review=read_json(root/'AI_REVIEW.json')
    assert review['status']=='AI_EXPLORATORY_FRAME_REVIEW_COMPLETE'
    assert set(a['review_frames']) <= set(review['review_frames'])
    for entry in a['evidence']:
        assert sha256(root/entry['path'])==entry['sha256']
        entry['viewed']=True
    a.update(status='ANALYSIS_COMPLETE_AI_REVIEW_ONLY',visual_review=str(root/'AI_REVIEW.json'),verdict=review['verdict'])
    write_json(root/'analysis.json',a)
    roots={'original':Path(a['original']),'before':Path(a['previous']),'after':Path(a['current'])}
    tables={}
    for key,path in roots.items():
        with (path/'run/quality_delivered_mp4.csv').open() as f:
            tables[key]=[{k:float(v) for k,v in row.items()} for row in csv.DictReader(f)]
    fig,axes=plt.subplots(2,2,figsize=(12,6),layout='constrained')
    for row,(metric,label) in enumerate((('psnr_db','PSNR (dB), higher is better'),('lpips_vgg','LPIPS, lower is better'))):
        for col,stop in enumerate((33,240)):
            ax=axes[row,col]
            for key,name,color in (('before','Collision fix','#aa3377'),('after','+ Short schedule fix','#0077bb')):
                ax.plot(np.arange(stop)/24,[v[metric] for v in tables[key][:stop]],label=name,c=color,lw=1.4)
            ax.axvspan(0,8/24,color='gray',alpha=.15)
            if col:ax.axvline(5,color='gray',ls=':',lw=1)
            ax.set(title='Beginning' if col==0 else 'Full 10 s',xlabel='Time (s)',ylabel=label)
            ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(root/'quality_timeline.svg')
    fig.savefig(root/'quality_timeline.png',dpi=150)
    plt.close(fig)
    labels={'psnr_db':'PSNR ↑','ssim':'SSIM ↑','lpips_vgg':'LPIPS ↓','clip':'CLIP ↑','dists':'DISTS ↓'}
    rows=''.join(f"<tr><td>{label}</td><td>{a['quality']['original'][k]:.4f}</td><td>{a['quality']['before'][k]:.4f}</td><td>{a['quality']['after'][k]:.4f}</td></tr>" for k,label in labels.items())
    regions=''
    for k,name in (('first_0_8','처음 0~0.333초'),('following_9_32','바로 다음 0.375~1.333초'),
                   ('rest_33_239','이후 1.375~9.958초'),('second_half_120_239','후반 5~9.958초')):
        r=a['regions'][k]
        regions+=f"<tr><td>{name}</td><td>{r['before']['psnr_db']:.3f} → {r['after']['psnr_db']:.3f}</td><td>{r['before']['lpips_vgg']:.4f} → {r['after']['lpips_vgg']:.4f}</td></tr>"
    notes=''.join(f"<tr><td>{', '.join(f'{i/24:.3f}' for i in r['frames'])}초</td><td>{html.escape(r['note'])}</td></tr>" for r in review['observations'])
    evidence=''.join(f"<details><summary>{e['path']} · 프레임 {e['frames']}</summary><img loading='lazy' src='{e['path']}' alt='원본 시간표 수정 전후 비교'></details>" for e in a['evidence'])
    mp4=''.join(f"<details><summary>최종 MP4 검토 · 프레임 {e['frames']}</summary><img loading='lazy' src='{e['path']}' alt='최종 MP4 대조'></details>" for e in review['mp4_evidence'])
    def duration(seconds):return f'{int(seconds)//60}분 {int(round(seconds))%60:02d}초'
    before,after=a['timing']['before'],a['timing']['after']
    page=f"""<!doctype html><html lang='ko'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>보행 전체 생성 시간표 수정 분석</title><style>body{{max-width:1500px;margin:24px auto;padding:0 18px;font:17px/1.65 system-ui,sans-serif;color:#e8edf5;background:#151c27}}a{{color:#91cbff}}video,img{{width:100%}}table{{width:100%;border-collapse:collapse;margin:18px 0}}td,th{{border:1px solid #526079;padding:10px;text-align:left}}th,.note{{background:#23324a}}.note{{padding:15px}}button{{padding:8px 12px;margin:5px;cursor:pointer}}details{{padding:10px;border:1px solid #526079;margin:8px 0}}</style>
<h1>보행 10초: 짧은 구간 생성 시간표 수정</h1><p class='note'>{html.escape(review['conclusion'])}</p>
<p>240프레임·15구간. 첫 구간 시간표 1곳 수정, 나머지 14구간의 시간표·조건 위치 유지. 원본 / 조건 충돌 수정 결과 / 시간표 추가 수정 순서입니다.</p>
<video id='v' controls preload='metadata' src='../comparison.mp4'></video>
<p><button data-time='0.125'>초반</button><button data-time='3.375'>3.375초</button><button data-time='5.333333'>5.33초</button><button data-time='5.541667'>5.54초</button><button data-time='7.333333'>7.33초</button><button data-time='8.666667'>8.67초</button></p>
<h2>전체 MP4 평균</h2><table><tr><th>지표</th><th>두 수정 전 FP32</th><th>조건 충돌 수정</th><th>+ 시간표 수정</th></tr>{rows}</table>
<h2>후반 영향</h2><table><tr><th>구간</th><th>PSNR ↑, 시간표 수정 전 → 후</th><th>LPIPS ↓, 시간표 수정 전 → 후</th></tr>{regions}</table>
<p>구간별 LPIPS 평균: {a['segments_lpips_improved']}개 개선, {a['segments_lpips_regressed']}개 악화. 화질 점수 변화이며 의미 오류 발생률은 아닙니다.</p>
<img src='quality_timeline.svg' alt='시간별 PSNR LPIPS'><p><a href='quality_timeline.svg'>SVG</a> · <a href='quality_timeline.png'>PNG</a></p>
<h2>장면 관찰</h2><table><tr><th>시점</th><th>관찰</th></tr>{notes}</table>
<h2>시간·비교 조건</h2><ul><li>영상 생성: {duration(before['generation_seconds'])} → {duration(after['generation_seconds'])}. 한 번씩의 관측이며 속도 향상 검증은 아닙니다.</li>
<li>이번 완료 단계 합계: {duration(after['stage_seconds_sum'])}. 양쪽 모두 기존 T5 저장값 재사용.</li>
<li>수신 자료·캡션·조건 위치 동일. VAE·초기·반복 생성 잡음 {a['noise_scopes']}개 범위, {a['noise_draws']}개 텐서 해시 일치.</li>
<li>첫 9프레임은 이전 짧은 시간표 수정 시험과 픽셀 일치. 15구간 모두 유효한 생성 시각·조건 보존 검사 통과.</li>
<li>뒤 구간은 달라진 이전 출력을 참조하므로 영상이 달라질 수 있습니다. 추가 전송량은 0입니다.</li></ul>
<h2>판정</h2><p>{html.escape(review['next'])}</p><p>PNG {len(review['review_frames'])}시점, 최종 MP4 {len(review['mp4_frames'])}시점 AI 검토. 한 영상·한 시드의 탐색 검사이며 독립 의미 오류 정답·할루시네이션 완화 검증은 미완료입니다.</p>
<h2>표본 이미지</h2>{evidence}{mp4}
<p><a href='analysis.json'>수치·감사</a> · <a href='AI_REVIEW.json'>AI 관찰</a> · <a href='../RESULT.json'>완료 결과</a></p>
<script>document.querySelectorAll('button[data-time]').forEach(b=>b.onclick=()=>{{let v=document.getElementById('v');v.pause();v.currentTime=Number(b.dataset.time);}});</script></html>"""
    (root/'review.html').write_text(page,encoding='utf-8')
    write_json(root/'manifest.json',{str(p.relative_to(root)):sha256(p) for p in root.rglob('*') if p.is_file() and p.name!='manifest.json'})
    print(root/'review.html')


if __name__=='__main__':
    main()
