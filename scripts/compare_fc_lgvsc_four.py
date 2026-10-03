"""Read-only audits and CPU frame comparisons of LGVSC versus FC-LGVSC."""
import argparse
import csv
import datetime as dt
import html
import os
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from semantic_transmission import fc_lgvsc as fc, fc_lgvsc_batch as batch
from semantic_transmission.short_video_batch import baseline_alignment, channel_source
from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.webvid5 import read_json, fingerprint
from semantic_transmission.webvid_ablation import snapshot
from semantic_transmission.video_io import probe

REPO = fc.REPO
KEYS = ('psnr_db', 'ssim', 'lpips_vgg', 'clip', 'dists')
LABELS = ('PSNR ↑', 'SSIM ↑', 'LPIPS ↓', 'CLIP ↑', 'DISTS ↓')
SHORT = REPO / 'outputs/etri_latest_short3_20261001'
OLD_TV = REPO / 'outputs/etri_60s_tv_low_08_42057b2ee8ed'
POINTS = dict(tv_low_08=[3,8,12,235,600,698,806,960,1036,1140,1308,1421],
              single_subject=[52,67,81,269], candle_flowers=[36,99,108,129],
              person_walk=[3,8,12,104,133,171,176,191,208])


def rows(path):
    with path.open() as f:
        return [{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]


def means(values, ids=None):
    selected = values if ids is None else [values[i] for i in ids]
    return {k:float(np.mean([r[k] for r in selected])) for k in KEYS}


def receipts(root, signature):
    records = {}
    for path in (root/'stages').glob('*.json'):
        r = read_json(path)
        assert r['status']=='PASSED' and r['identity']==signature, path
        assert snapshot(root,r['required'])==r['artifacts'], path
        records[path.stem] = r
    for r in records.values():
        assert all(fingerprint(records[k])==value for k,value in r['dependencies'].items())
    return records


def frames(path, wanted):
    wanted = set(wanted)
    cap = cv2.VideoCapture(str(path))
    result = {}
    try:
        for index in range(max(wanted)+1):
            ok, frame = cap.read()
            assert ok, (path,index)
            if index in wanted:
                result[index] = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    finally:
        cap.release()
    return result


def duration(value):
    s=round(value)
    return f'{s//60}분 {s%60:02d}초'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=REPO/'outputs/fc_lgvsc_four_comparison_20261001')
    args=parser.parse_args()
    dest=args.output.resolve()
    dest.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(1)
    old_summary=read_json(SHORT/'analysis_lgvsc/COMPARISON.json')
    old_short={r['video']:r for r in old_summary['videos']}
    starts,ends,results=[],[],[]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    for job in batch.preflight():
        name=job['video']; current=Path(job['output']); run=current/'run'
        protocol=read_json(current/'execution_protocol.json')
        assert fingerprint({k:v for k,v in protocol.items() if k!='signature'})==protocol['signature']==job['signature']
        stages=receipts(current,protocol['signature'])
        assert set(stages)=={'initialize','reconstruct','audit','evaluate','comparison'}
        assert snapshot(run,fc.INPUTS)==protocol['inputs']
        cfg=read_json(run/'run_config.json'); keys=read_json(run/'keyframes.json')['indices']
        prior=Path(protocol['source']); result=read_json(current/'RESULT.json')
        assert result['status']=='PASS_FC_LGVSC_RECONSTRUCTION_REVIEW_PENDING'
        policy=read_json(run/'receiver_policy.json')
        fc.base.text.validate_usage(run)
        fc.base.tail.validate_trace(read_json(run/'receiver/tail_reference_trace.json'),len(keys)-2)
        fc.base.fix.validate_trace(read_json(run/fc.base.fix.TRACE),protocol['plan'])
        fc.paired.fix.validate_trace(read_json(run/fc.paired.fix.TRACE),keys,30)
        noise=read_json(run/fc.base.noise.TRACE); previous_noise=read_json(prior/'run'/fc.base.noise.TRACE)
        fc.base.noise.validate_trace(noise,policy['noise_contract'],len(keys)-1,30)
        assert noise['matched_reference'] and all(noise[k]==previous_noise[k] for k in ('records','runtime','contract'))
        source=run/'data/normalized.mp4'; video=run/'receiver/reconstruction/sample_0000.mp4'
        quality=read_json(run/'quality.json'); newrows=rows(run/'quality_delivered_mp4.csv')
        source_hash=sha256(source); new_hash=sha256(video)
        assert quality['status']=='PASSED' and quality['video_sha256']==new_hash==result['video_sha256']
        assert quality['source_sha256']==source_hash
        assert [r['frame'] for r in newrows]==list(range(cfg['frames']))
        assert all(abs(means(newrows)[k]-quality['delivered_mp4'][k])<1e-8 for k in KEYS)
        info=probe(video)
        assert (info['frames'],info['fps'],info['width'],info['height'])==(cfg['frames'],24,576,320)
        if name=='tv_low_08':
            baseline=OLD_TV/'baseline'; oldvideo=baseline/'receiver/reconstruction/sample_0000.mp4'
            oldq=read_json(baseline/'quality.json'); oldrows=rows(baseline/'quality_delivered_mp4.csv')
            oldprotocol=read_json(OLD_TV/'protocol.json')
            oldstages=receipts(OLD_TV,oldprotocol['signature'])
            assert oldq['video_sha256']==sha256(oldvideo) and oldq['source_sha256']==source_hash
            oldseconds=oldstages['reconstruct']['seconds']; oldmetric=oldq['delivered_mp4']
            profile=oldq['evaluation_profile']; kept=list(range(cfg['frames']))
            olddisplay=oldvideo; timing_evidence=snapshot(OLD_TV,['stages/reconstruct.json'])
        else:
            baseline=Path(read_json(SHORT/name/'protocol.json')['baseline'])
            oldvideo=baseline/'receiver/reconstruction/sample_0000.mp4'
            cache=read_json(SHORT/'analysis_lgvsc'/name/'metrics.json')
            assert cache['status']=='PASSED' and cache['source_sha256']==source_hash
            assert cache['old_video_sha256']==sha256(oldvideo)
            mapping,kept=baseline_alignment(baseline)
            assert kept==cache['kept_baseline_positions'] and probe(oldvideo)['frames']==len(mapping)
            oldmetric,oldrows,profile=cache['baseline'],cache['baseline_rows'],cache['profile']
            row=old_short[name]; timing_evidence=row['timing_evidence']
            assert all(sha256(Path(p))==h for p,h in timing_evidence.items())
            oldseconds=row['baseline_receiver_seconds']
            olddisplay=prior/'baseline_aligned.mp4'  # Viewing copy only; metrics use the original MP4.
        assert sha256(baseline/'data/normalized.mp4')==source_hash
        assert len(oldrows)==cfg['frames'] and quality['evaluation_profile']==profile=='lgvsc_official_metrics_v1'
        assert all(abs(means(oldrows)[k]-oldmetric[k])<1e-8 for k in KEYS)
        oldkeys=read_json(baseline/'receiver/decoder_inputs.json')['indices']
        nonkeys=[i for i in range(cfg['frames']) if i not in set(oldkeys)|set(keys)]
        oldchannel=read_json(channel_source(baseline)/'channel_accounting.json')
        newchannel=read_json(run/'channel_accounting.json')
        seconds={k:r['seconds'] for k,r in stages.items()}
        historical_t5=read_json(prior/'stages/prepare-text.json')['seconds']
        start=min(dt.datetime.fromisoformat(r['started']) for r in stages.values())
        end=max(dt.datetime.fromisoformat(r['started'])+dt.timedelta(seconds=r['seconds']) for r in stages.values())
        starts.append(start);ends.append(end)
        folder=dest/name;folder.mkdir(exist_ok=True)
        selected=POINTS[name]
        images=[frames(source,selected),frames(oldvideo,[kept[i] for i in selected]),frames(video,selected)]
        evidence=[]
        for page,offset in enumerate(range(0,len(selected),3)):
            ids=selected[offset:offset+3]; sheet=Image.new('RGB',(1536,320*len(ids)),(20,24,30));draw=ImageDraw.Draw(sheet)
            for row,i in enumerate(ids):
                for col,label in enumerate(('SOURCE','LGVSC BASELINE','FC-LGVSC')):
                    image=images[col][kept[i] if col==1 else i]
                    sheet.paste(Image.fromarray(image).resize((512,284)),(col*512,row*320+36))
                    draw.text((col*512+5,row*320+2),f'{label} | f{i} | {i/24:.3f}s',font=font,fill='white')
                    if col:
                        metric=[oldrows,newrows][col-1][i]
                        draw.text((col*512+5,row*320+18),f"PSNR {metric['psnr_db']:.2f} LPIPS {metric['lpips_vgg']:.3f}",font=font,fill='white')
            path=folder/f'samples_{page:02d}.jpg';sheet.save(path,quality=95)
            evidence.append(dict(path=str(path.relative_to(dest)),frames=ids,sha256=sha256(path)))
        prior_hash=sha256(prior/'run/receiver/reconstruction/sample_0000.mp4')
        payload=dict(video=name,frames=cfg['frames'],fps=24,baseline=str(baseline),current=str(current),
            quality_before=oldmetric,quality_after=quality['delivered_mp4'],
            metric_deltas={k:quality['delivered_mp4'][k]-oldmetric[k] for k in KEYS},
            common_generated_frames=len(nonkeys),common_generated_before=means(oldrows,nonkeys),common_generated_after=means(newrows,nonkeys),
            baseline_keyframes=len(oldkeys),current_keyframes=len(keys),
            baseline_channel_uses=oldchannel['total_complex_channel_uses'],current_channel_uses=newchannel['total_complex_channel_uses'],
            baseline_receiver_including_t5_seconds=oldseconds,current_cached_generation_seconds=seconds['reconstruct'],
            current_observed_t5_preparation_seconds=0,historical_t5_preparation_seconds=historical_t5,
            generation_plus_historical_t5_reference_seconds=seconds['reconstruct']+historical_t5,
            receiver_observed_change_percent=100*(seconds['reconstruct']/oldseconds-1),stage_seconds=seconds,
            current_stage_sum_seconds=sum(seconds.values()),current_stage_window_seconds=(end-start).total_seconds(),
            timing_evidence=timing_evidence,source_sha256=source_hash,baseline_video_sha256=sha256(oldvideo),video_sha256=new_hash,
            previous_fp32_video_sha256=prior_hash,identical_to_previous_fp32=new_hash==prior_hash,
            condition_repaired_segments=result['condition_repaired_segments'],schedule_repaired_segments=result['schedule_repaired_segments'],
            checked_stages=len(stages),noise_matches_previous_fp32=True,legacy_generation_noise_matched=False,
            reviewed_frame_candidates=selected,evidence=evidence,
            viewing_videos=[str(source),str(olddisplay),str(video)],
            selected_frame_metrics={str(i):dict(before=oldrows[i],after=newrows[i]) for i in selected})
        results.append(payload)
        print(name,dict(before=means(oldrows),after=means(newrows),baseline_seconds=oldseconds,current_seconds=seconds['reconstruct']),flush=True)
    summary=dict(status='NUMERIC_AUDIT_COMPLETE_VISUAL_REVIEW_PENDING',videos=results,
        current_generation_total_seconds=sum(r['current_cached_generation_seconds'] for r in results),
        baseline_receiver_total_seconds=sum(r['baseline_receiver_including_t5_seconds'] for r in results),
        current_successful_stage_window_seconds=(max(ends)-min(starts)).total_seconds(),
        current_successful_stage_sum_seconds=sum(r['current_stage_sum_seconds'] for r in results),
        independent_semantic_review='PENDING',hallucination_mitigation_verified=False,
        quality_scope='Same source frame timeline. Original decoded MP4 metrics; repeated legacy boundary frames removed for the three short videos. Cached baseline metrics verified against video/source hashes; current stage artifacts and mean CSV values verified.',
        time_scope='Baseline reconstruct includes inline T5. Current generation reuses FP32 tensors prepared earlier. The column adding historical T5 is a reference sum, not a newly measured cold run. Selection, caption authoring and sender costs excluded from receiver times. Stage window includes audits, evaluation, comparison and hashing between stages; launch/preflight/terminal exit overhead excluded.',
        limitations=['Multiple simultaneous method changes and unmatched legacy generation noise.',
            'Different segment counts and historical hardware load; not an isolated GPU speed comparison.',
            'Local original-decoder baselines include execution support changes; not full paper reproduction.',
            'Post hoc AI sample review is not independent semantic-error ground truth.'],
        analyzer_sha256=sha256(Path(__file__)))
    write_json(dest/'COMPARISON.json',summary)
    def link(path):return html.escape(quote(os.path.relpath(path,dest)))
    page=['<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>LGVSC / FC-LGVSC 4편 비교</title><style>body{font:16px/1.6 system-ui;max-width:1550px;margin:24px auto;padding:0 16px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #bbb;padding:8px}img{width:100%}.videos{display:flex;gap:6px}.videos video{width:33.1%}button{margin:8px;padding:8px}details{margin:12px 0}</style>'
        '<h1>기존 LGVSC / 현재 FC-LGVSC</h1><p>같은 원본·시점의 최종 MP4 평균. 독립 할루시네이션 검수는 미완료입니다.</p>',
        '<table><tr><th>영상</th>'+''.join(f'<th>{label}</th>' for label in LABELS)+'</tr>']
    for r in results:
        page.append('<tr><td>'+r['video']+'</td>'+''.join(f"<td>{r['quality_before'][k]:.4f} → {r['quality_after'][k]:.4f}</td>" for k in KEYS)+'</tr>')
    page+=['</table><h2>복원 시간</h2><p>기존은 T5 포함, 이번은 저장값 재사용. 마지막 열은 이전에 측정한 T5 최초 계산 비용을 더한 참고값이며 이번 실측이 아닙니다.</p><table><tr><th>영상</th><th>기존 LGVSC</th><th>이번 FC-LGVSC</th><th>T5 최초 비용 합산 참고</th></tr>']
    for r in results:
        page.append('<tr><td>'+r['video']+'</td>'+''.join(f'<td>{duration(r[k])}</td>' for k in ('baseline_receiver_including_t5_seconds','current_cached_generation_seconds','generation_plus_historical_t5_reference_seconds'))+'</tr>')
    page += ['</table>',f"<p>이번 생성 합계 {duration(summary['current_generation_total_seconds'])}. 첫 단계 시작부터 마지막 비교 완료까지 {duration(summary['current_successful_stage_window_seconds'])}.</p>"]
    for r in results:
        page+=[f"<section><h2>{r['video']}</h2><p>왼쪽부터 원본 / LGVSC / FC-LGVSC. 짧은 영상의 LGVSC는 중복 경계를 제거한 표시용 사본이며 지표는 원본 MP4 기준입니다.</p><div class='videos'>",
               *[f'<video controls preload="metadata" src="{link(p)}"></video>' for p in r['viewing_videos']],
               '</div><button class="play">함께 재생</button><button class="pause">함께 정지</button>',
               *[f'<button data-time="{i/24}">{i/24:.2f}초</button>' for i in r['reviewed_frame_candidates']]]
        for e in r['evidence']:
            page.append(f"<details><summary>표본 프레임 {e['frames']}</summary><img loading='lazy' src='{e['path']}'></details>")
        page.append('</section>')
    page+=['<p><a href="COMPARISON.json">수치·시간·검증 근거</a> · <a href="AI_REVIEW.json">AI 표본 검토 기록</a></p>'
        '<script>document.querySelectorAll("section").forEach(s=>{const v=[...s.querySelectorAll("video")];s.querySelector(".play").onclick=()=>{v.forEach(x=>{x.currentTime=v[0].currentTime;x.play()})};s.querySelector(".pause").onclick=()=>v.forEach(x=>x.pause());s.querySelectorAll("[data-time]").forEach(b=>b.onclick=()=>v.forEach(x=>{x.pause();x.currentTime=Number(b.dataset.time)}));});</script></html>']
    (dest/'review.html').write_text(''.join(page),encoding='utf-8')
    review_path=dest/'AI_REVIEW.json'
    if review_path.exists():
        review=read_json(review_path)
        assert review['status']=='AI_EXPLORATORY_FRAME_REVIEW_COMPLETE'
        for r in results:
            observed=review['videos'][r['video']]
            assert observed['video_sha256']==r['video_sha256']
            assert observed['frames']==r['reviewed_frame_candidates']
            assert observed['evidence']=={e['path']:e['sha256'] for e in r['evidence']}
            r['visual_observations']=observed['observations']
        summary.update(status='COMPARISON_COMPLETE_AI_SAMPLE_REVIEW',reviewed_timestamps=sum(len(r['reviewed_frame_candidates']) for r in results),
                       conclusion=review['conclusion'])
        write_json(dest/'COMPARISON.json',summary)
        page=(dest/'review.html').read_text()
        notes='<h2>AI 표본 검토: 29시점</h2><p>'+html.escape(review['conclusion'])+'</p>'
        notes+='<p>추가·누락·왜곡의 발생률을 측정한 독립 검수가 아닙니다.</p><ul>'
        for name,item in review['videos'].items():
            for note in item['observations']:
                notes+=f"<li><b>{name}</b> — {html.escape(note['note'])}</li>"
        notes+='</ul>'
        (dest/'review.html').write_text(page.replace('<h2>복원 시간</h2>',notes+'<h2>복원 시간</h2>'),encoding='utf-8')
    write_json(dest/'manifest.json',{str(p.relative_to(dest)):sha256(p) for p in dest.rglob('*') if p.is_file() and p.name!='manifest.json'})
    print(dest/'review.html',flush=True)


if __name__=='__main__':
    main()
