"""Compare a frozen assistant selection and its actual full-video reconstruction."""
import csv
import html
import json
from pathlib import Path
import sys
import subprocess

import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from semantic_transmission.artifacts import sha256,write_json
from semantic_transmission.webvid5 import read_json
from semantic_transmission.etri_60s_check import pts_audit

ROOT=REPO/'outputs/etri_visual_keys_20260929'
BASE=REPO/'outputs/etri_60s_tv_low_08_42057b2ee8ed/baseline'


def metric_rows(run):
    with (run/'quality_delivered_mp4.csv').open() as f:
        rows=list(csv.DictReader(f))
    assert [int(r['frame']) for r in rows]==list(range(1440))
    return [{k:float(v) for k,v in r.items()} for r in rows]


def mean_rows(rows,indices):
    return {k:float(np.mean([rows[i][k] for i in indices])) for k in ('psnr_db','ssim','lpips_vgg','clip','dists')}


def evidence(root,result):
    from PIL import Image,ImageDraw,ImageFont
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    plan=read_json(root/'quality_plan.json')
    assert sha256(root/'quality_plan.json')==read_json(root/'evaluation_freeze.json')['quality_plan_sha256']
    metrics=[metric_rows(BASE),metric_rows(root/'assistant')]
    difference=[metrics[1][i]['lpips_vgg']-metrics[0][i]['lpips_vgg'] for i in range(1440)]
    extremes=[]
    for reverse in (True,False):
        chosen=[]
        for i in sorted(range(1440),key=lambda i:difference[i],reverse=reverse):
            if all(abs(i-j)>=24 for j in chosen): chosen.append(i)
            if len(chosen)==6: break
        extremes.append(chosen)
    groups={'predeclared':plan['ai_review_frames'],'posthoc_regressions':extremes[0],'posthoc_improvements':extremes[1]}
    source=BASE/'data/frames/sample'
    dirs=[source,BASE/'receiver/reconstruction/sample_0000_frames',root/'assistant/receiver/reconstruction/sample_0000_frames']
    keysets=[set(read_json(BASE/'keyframes.json')['indices']),set(read_json(root/'assistant_selection.json')['indices'])]
    prior_path=root/'review/evidence.json'
    prior={r['path']:r for r in read_json(prior_path)} if prior_path.exists() else {}
    sheets=[]
    for group,indices in groups.items():
        for page,start in enumerate(range(0,len(indices),6)):
            these=indices[start:start+6]
            canvas=Image.new('RGB',(1728,len(these)*350),'#202020'); draw=ImageDraw.Draw(canvas)
            for row,i in enumerate(these):
                for col,(name,directory) in enumerate(zip(('source','SKEM','assistant'),dirs)):
                    p=directory/(f'{i}.png' if col==0 else f'{i:05d}.png')
                    with Image.open(p) as im: canvas.paste(im.convert('RGB'),(col*576,row*350+25))
                    label=f'{name} | f{i} {i/24:.3f}s'
                    if col:
                        kind='KEY' if i in keysets[col-1] else 'GENERATED'
                        label+=f" | {kind} | LPIPS {metrics[col-1][i]['lpips_vgg']:.3f}"
                    draw.text((col*576+4,row*350+3),label,font=font,fill='white')
            path=root/f'review/{group}_{page:02d}.jpg'; path.parent.mkdir(exist_ok=True)
            canvas.save(path,quality=94)
            relative=str(path.relative_to(root)); digest=sha256(path)
            before=prior.get(relative,{})
            sheets.append({'path':relative,'frames':these,'sha256':digest,
                           'scope':'Lossless pre-MP4 frames; labels show delivered-MP4 LPIPS',
                           'viewed':before.get('sha256')==digest and before.get('viewed',False)})
    write_json(root/'review/evidence.json',sheets)


def main():
    root=ROOT
    if not (root/'pipeline_complete.json').exists(): raise ValueError('Reconstruction pipeline is incomplete')
    for p,digest in read_json(root/'baseline_evidence_hashes.json').items(): assert sha256(p)==digest
    selected=read_json(root/'assistant_selection.json')
    assert sha256(root/'assistant_selection.json')==read_json(root/'selection_freeze.json')['selection_sha256']
    plan=read_json(root/'quality_plan.json')
    assert sha256(root/'quality_plan.json')==read_json(root/'evaluation_freeze.json')['quality_plan_sha256']
    keys={'skem':read_json(BASE/'keyframes.json')['indices'],'assistant':selected['indices']}
    union=set(keys['skem'])|set(keys['assistant']); inside=[i for i in range(1440) if i not in union]
    result={'status':'FULL_60S_COMPARISON_COMPLETE','source_id':'tv_low_08','source_split':'development',
        'independent_semantic_review':'PENDING','hallucination_mitigation_verified':False,
        'adopted_as_default':False,'selection':read_json(root/'selection_comparison.json'),
        'common_interior_frames':inside,'conditions':{},
        'limitations':['One known development video, one channel/generation seed; no independent semantic ground truth.',
           'Assistant uses 6fps visual overview, dense focus and 1s anchor rule; effects are not attributable to the model alone.',
           'Historical SKEM timing and interactive assistant wall time are different measurement scopes.',
           'Changed keyframes change captions, flow, segment layout and diffusion noise; exact received symbols are preserved only for common keys.']}
    for name,run in [('skem',BASE),('assistant',root/'assistant')]:
        q=read_json(run/'quality.json'); rows=metric_rows(run)
        assert q['status']=='PASSED' and q['video']['frames']==1440
        video=run/'receiver/reconstruction/sample_0000.mp4'; assert sha256(video)==q['video_sha256']
        pts_audit(video,1440,24)
        accounting=read_json(run/'channel_accounting.json')
        result['conditions'][name]={'run':str(run),'quality':q['delivered_mp4'],
            'lossless_quality':q['lossless_frames'],'common_interior_quality':mean_rows(rows,inside),
            'channel_uses':accounting['total_complex_channel_uses'],
            'visual_channel_uses':accounting['visual_complex_channel_uses'],
            'digital_channel_uses':accounting['digital_complex_channel_uses'],
            'uniform_windows':[dict(frames=[a,b],quality=mean_rows(rows,range(a,b+1))) for a,b in plan['uniform_windows']],
            'source_event_windows':{k:dict(frames=[a,b],quality=mean_rows(rows,range(a,b+1))) for k,(a,b) in plan['source_event_windows'].items()}}
    a,b=result['conditions']['assistant'],result['conditions']['skem']
    result['delta']={'quality':{k:a['quality'][k]-b['quality'][k] for k in ('psnr_db','ssim','lpips_vgg','clip','dists')},
        'channel_use_ratio':a['channel_uses']/b['channel_uses'],
        'common_interior_quality':{k:a['common_interior_quality'][k]-b['common_interior_quality'][k] for k in ('psnr_db','ssim','lpips_vgg','clip','dists')}}
    result['channel_pairing']=read_json(root/'assistant/channel_accounting.json')['common_frames_exact']
    result['pipeline']=read_json(root/'pipeline_complete.json')
    result['selection']['quality_status']='FULL_RECONSTRUCTION_COMPLETE_INDEPENDENT_SEMANTIC_REVIEW_PENDING'
    result['report_code_sha256']=sha256(Path(__file__))
    observations=read_json(root/'AI_REVIEW.json') if (root/'AI_REVIEW.json').exists() else None
    result['ai_sample_review']={'status':'COMPLETE' if observations else 'PENDING',
        'independent':False,'path':'AI_REVIEW.json' if observations else None}
    if observations:
        result['ai_sample_review']['sha256']=sha256(root/'AI_REVIEW.json')
        result['decision']=observations['decision']
    write_json(root/'RESULT.json',result)
    write_json(root/'selection_comparison.json',result['selection'])
    evidence(root,result)
    comparison=root/'comparison.mp4'
    if not comparison.exists():
        subprocess.run(['ffmpeg','-v','error','-nostdin','-i',str(BASE/'data/normalized.mp4'),
            '-i',str(BASE/'receiver/reconstruction/sample_0000.mp4'),
            '-i',str(root/'assistant/receiver/reconstruction/sample_0000.mp4'),
            '-filter_complex','[0:v][1:v][2:v]hstack=inputs=3[v]','-map','[v]','-an',
            '-c:v','libx264','-crf','18','-preset','fast','-pix_fmt','yuv420p',str(comparison)],check=True)
    pts_audit(comparison,1440,24)
    rows=''
    for name,label in [('skem','기존 SKEM'),('assistant','AI 직접 선택 + 최대 1초')]:
        d=result['conditions'][name]; s=result['selection'][name]
        cells=[label,s['keyframes'],s['max_gap_seconds'],f"{d['quality']['psnr_db']:.3f}",
               f"{d['quality']['ssim']:.4f}",f"{d['quality']['lpips_vgg']:.4f}",
               f"{d['quality']['clip']:.4f}",f"{d['quality']['dists']:.4f}",f"{d['channel_uses']:,}"]
        rows+='<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in cells)+'</tr>'
    gallery=''.join(f'<h3>{html.escape(p.stem)}</h3><img loading="lazy" src="sheets/{p.name}">' for p in sorted((root/'sheets').glob('selected*.jpg')))
    review=('<h2>AI 표본 검토</h2><ul>'+''.join(f'<li>{html.escape(t)}</li>' for t in observations['summary_ko'])+'</ul>'
            if observations else '<p>비교 프레임 준비 완료. AI 시각 검토·독립 오류 검수는 아직 완료되지 않았습니다.</p>')
    comparison_gallery=''
    for group,label in [('predeclared','사전에 정한 검토 프레임'),('posthoc_regressions','LPIPS 악화가 큰 프레임 — 결과 확인 후 선정'),('posthoc_improvements','LPIPS 개선이 큰 프레임 — 결과 확인 후 선정')]:
        comparison_gallery+='<details><summary>'+label+'</summary>'+''.join(
            f'<img loading="lazy" src="review/{p.name}">' for p in sorted((root/'review').glob(f'{group}_*.jpg')))+'</details>'
    runtime=sum(result['pipeline']['stages_seconds'].values())/60
    boundary_gallery=''
    if observations:
        boundary_gallery='<details><summary>새 오류의 수신·생성·MP4 확인</summary>'+''.join(
            f'<img loading="lazy" src="review/{name}.jpg">' for name in ('f12_boundaries','delivered_error_checks'))+'</details>'
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>tv_low_08 키프레임 비교</title>
<style>body{font:16px system-ui;max-width:1740px;margin:24px auto;padding:16px}video,img{max-width:100%}td,th{border:1px solid #ccc;padding:9px}table{border-collapse:collapse}</style>
<h1>tv_low_08: 기존 SKEM과 AI 직접 선택</h1><p>동일한 연속 60초·24fps·AWGN 10dB·생성 30단계. 한 개발 영상의 비교이며 독립 의미 오류 검수는 미완료입니다.</p>
<p>왼쪽 원본 / 가운데 기존 SKEM / 오른쪽 AI 직접 선택. 아래 수치는 전달 MP4 전체 프레임 평균입니다.</p>
<video controls preload="metadata" src="comparison.mp4"></video>
<table><tr><th>방식</th><th>키프레임</th><th>최대 간격(초)</th><th>PSNR ↑</th><th>SSIM ↑</th><th>LPIPS ↓</th><th>CLIP ↑</th><th>DISTS ↓</th><th>전송량(복소 심볼)</th></tr>'''+rows+'</table>'+review+f'<p>새 선택 이후 전체 처리: {runtime:.2f}분. 키프레임·설명·움직임·부호화 비용을 포함한 전송량이며 물리 링크 부가 비용은 제외합니다.</p>'+'''
<p>선택 시간: 기존 SKEM의 과거 실행 약 23시간 19분; AI 선택은 준비·시각 검토 포함 약 2분 38초. 동일 환경의 API 속도 비교가 아닙니다. 새 복원 시간은 별도입니다.</p>
<p>공통 키프레임 7개의 수신 자료는 바이트 단위로 재사용합니다. 새 프레임에는 같은 AWGN 분포를 적용합니다. 선택이 달라져 캡션·구간·생성 잡음까지 동일하지는 않습니다.</p>
<p><a href="RESULT.json">수치·범위</a> · <a href="AI_REVIEW.json">AI 관찰·판정</a> · <a href="assistant_selection.json">선택 이유</a> · <a href="quality_plan.json">사전 평가 계획</a></p>
<h2>같은 시각의 원본·복원 비교</h2><p>프레임은 MP4 인코딩 전 PNG, LPIPS는 전달 MP4 기준. KEY는 키프레임 위치, GENERATED는 중간 생성 위치입니다.</p>'''+comparison_gallery+boundary_gallery+'<img src="selection_timeline.svg"><details><summary>선택한 키프레임 전체</summary>'+gallery+'</details></html>'
    (root/'review.html').write_text(page)
    selection_page=root/'selection.html'
    selection_page.write_text(selection_page.read_text().replace(
        '키프레임 선택 완료 — 실제 복원 비교는 진행 중',
        '키프레임 선택·60초 복원 비교 완료</h1><p><a href="review.html">실제 복원 비교·화질 결과 보기</a></p><h1>선택 목록'))
    print(json.dumps({'status':result['status'],'delta':result['delta']},ensure_ascii=False))


if __name__=='__main__': main()
