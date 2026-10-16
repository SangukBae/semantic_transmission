"""Audit and summarize the completed four-stage local BSU investigation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import html
import json
import os
from pathlib import Path
import statistics as st
import time
from . import bsu_motion_validation as common
from .artifacts import sha256,write_json


def check_hash(item):
    path,expected=item
    if sha256(path)!=expected:raise ValueError(f'artifact changed: {path}')
    return 1


def audit(root):
    start=time.monotonic();manifest=common.read(common.IMPORTED/'transfer_manifest.json')
    expected={str(common.IMPORTED/'inputs'/p):r['sha256'] for p,r in manifest['files'].items()}
    with ThreadPoolExecutor(max_workers=4) as pool:count=sum(pool.map(check_hash,expected.items()))
    kin=common.read(common.REPO/'outputs/bsu_kinetics_context/benchmark/protocol.json')
    kinetic_files={}
    for v in kin['videos'][:4]:
        prep=common.read(v['prepared']);kinetic_files[v['prepared']]=v['prepared_sha256']
        kinetic_files[prep['source_path']]=v['source_sha256'];kinetic_files[prep['normalized_video']]=prep['normalized_sha256']
        kinetic_files.update({str(Path(prep['source_frames'])/f'{k}.png'):h for k,h in prep['source_frame_hashes'].items()})
    with ThreadPoolExecutor(max_workers=4) as pool:kin_count=sum(pool.map(check_hash,kinetic_files.items()))
    for stage in ('flow','flow_reconstruction','budget_prepare','adapter_native'):
        if not common.verified(root,stage):raise ValueError(f'incomplete stage: {stage}')
    receipts=0
    for p in root.rglob('generation_complete.json'):
        for item in common.read(p)['outputs'].items():check_hash(item)
        receipts+=1
    for stage in ('flow_reconstruction','adapter','adapter_native'):
        for p,h in common.read(root/stage/'plan.json')['code'].items():check_hash((common.REPO/p,h))
    from .packets import accounting
    budgets=[]
    for j in common.read(root/'budget/plan.json')['jobs']:
        folder=Path(j['folder']);tx=folder/'trajectory_tx.bin';rx=folder/'trajectory_rx.bin'
        if tx.read_bytes()!=rx.read_bytes():raise ValueError('budget packet/channel mismatch')
        cost=accounting(tx.stat().st_size,bits_per_symbol=j['old_sender']['bits_per_symbol'])['complex_channel_uses']
        if cost!=j['channel_uses'] or cost>j['old_sender']['maximum_extra_channel_uses']:raise ValueError('allocation budget mismatch')
        budgets.append(dict(id=j['case']['id'],snr=j['case']['snr'],extra_symbols=cost,
            relative_to_base=cost/j['old_sender']['base_channel_uses'],windows=j['info']['sent_windows']))
    tracker=[]
    for j in common.read(root/'tracking/plan.json')['jobs']:
        folder=Path(j['folder']);tx=folder/'trajectory_tx.bin';rx=folder/'trajectory_rx.bin'
        if tx.read_bytes()!=rx.read_bytes():raise ValueError('tracker packet/channel mismatch')
        cost=accounting(tx.stat().st_size,bits_per_symbol=4)['complex_channel_uses']
        if cost!=j['maximum_channel_uses'] or cost!=j['actual_channel_uses']:raise ValueError('tracker actual-rate mismatch')
        tracker.append(dict(id=j['case']['id'],symbols=cost))
    plan=common.read(root/'adapter/plan.json');train={e['source_sha256'] for e in plan['examples'] if e['role']=='train'}
    test={e['source_sha256'] for e in plan['examples'] if e['role']!='train'}
    if train&test:raise ValueError('source split overlaps')
    for r in common.read(root/'adapter/data.json')['receipts']:
        check_hash((r['path'],r['sha256']));check_hash((r['received_path'],r['received_sha256']))
    training=common.read(root/'adapter/training.json');check_hash((root/'adapter/adapter.pt',training['checkpoint_sha256']))
    native=common.read(root/'adapter_native/plan.json');examples={e['id']:e for e in plan['examples'] if e['role']!='train'}
    from .generation_noise import validate_trace
    for c in native['cases']:
        pair=[];e=examples[c['id']]
        for arm in ('baseline','adapter'):
            run=root/'adapter_native'/c['id']/arm/'run'
            for p,src in [('received/metadata.bin',e['packet']),('received/trajectory.bin',e['trajectory'])]:
                check_hash((run/p,sha256(src)))
            original=Path(e['packet']).parent.parent
            for k in c['keys']:check_hash((run/f'receiver/frames/sample/key_frames_received/{k}.png',sha256(original/f'receiver/frames/sample/key_frames_received/{k}.png')))
            trace=common.read(run/'receiver/generation_noise.json');policy=common.read(run/'receiver_policy.json')
            validate_trace(trace,policy['noise_contract'],len(c['keys'])-1,30);pair.append(trace['records'])
        if pair[0]!=pair[1]:raise ValueError('native paired noise changed')
    # Snapshot generated evidence and executable code for checked reuse. This is
    # a post-run seal; it does not replace the prior input/plan receipts above.
    files=[p for stage in ('flow','flow_reconstruction','budget_prepare','budget','tracking','adapter','adapter_native')
        for p in (root/stage).rglob('*') if p.is_file() and p.suffix in ('.json','.png','.mp4','.bin','.pt','.npz')]
    code=[common.REPO/'src/semantic_transmission'/n for n in ('motion_sampling.py','trajectory_allocation.py','learned_trajectory.py',
        'trajectory_adapter.py','trajectory_adapter_runtime.py','bsu_motion_validation.py','bsu_budget_validation.py',
        'bsu_tracking_validation.py','bsu_adapter_validation.py','bsu_adapter_native.py')]
    with ThreadPoolExecutor(max_workers=4) as pool:hashes=dict(zip(map(str,files+code),pool.map(sha256,files+code)))
    write_json(root/'artifact_manifest.json',dict(status='POST_RUN_SEALED',files=hashes))
    result=dict(status='PASSED',imported_webvid_files_rehashed=count,kinetics_files_rehashed=kin_count,
        generation_receipts_verified=receipts,sealed_generated_files=len(files),train_sources=len(train),heldout_sources=len(test),
        source_disjoint=True,native_noise_identical=True,native_packets_and_keys_identical=True,
        budget_actual_costs=budgets,tracker_same_actual_costs=tracker,seconds=time.monotonic()-start,
        artifact_manifest_sha256=sha256(root/'artifact_manifest.json'))
    write_json(root/'AUDIT.json',result);print(json.dumps(result),flush=True)


def verify(root):
    seal=common.read(root/'artifact_manifest.json');check_hash((root/'artifact_manifest.json',common.read(root/'AUDIT.json')['artifact_manifest_sha256']))
    with ThreadPoolExecutor(max_workers=4) as pool:n=sum(pool.map(check_hash,seal['files'].items()))
    print(f'VERIFIED {n} sealed artifacts and code files',flush=True)


def contact_sheet(source,folders,dest,frames,title):
    import numpy as np
    from PIL import Image,ImageDraw
    times=np.linspace(0,frames-1,6).round().astype(int)[1:-1].tolist()
    canvas=Image.new('RGB',(4*288,3*184+38),'white');draw=ImageDraw.Draw(canvas);draw.text((8,8),title,fill='black')
    for row,(label,folder,padded) in enumerate([('source',source,False),*folders]):
        draw.text((4,38+row*184),label,fill='black')
        for col,t in enumerate(times):
            p=folder/(f'{t:05d}.png' if padded else f'{t}.png')
            with Image.open(p) as image:canvas.paste(image.convert('RGB').resize((288,160)),(288*col,60+row*184))
            draw.text((288*col+230,40+row*184),f'f={t}',fill='black')
    dest.parent.mkdir(exist_ok=True);canvas.save(dest)


def summarize(root):
    stages={k:common.read(root/k/'RESULT.json') for k in ('flow_reconstruction','budget','tracking','adapter','adapter_native')}
    def means_with_temporal(d,arm):
        m=dict(d['means'][arm]);rows=[r for r in d['rows'] if r['arm']==arm]
        for k in ('cadence_error_mean','false_freeze_rate'):m[k]=st.mean(r['temporal'][k] for r in rows)
        m['boundary_error']=st.mean(st.mean(b['maximum_change_error'] for b in r['boundaries']) for r in rows)
        return m
    criteria=common.read(root/'flow_reconstruction/plan.json')['acceptance']
    def comparison(a,b):
        changes={k:dict(before=a[k],after=b[k],delta=b[k]-a[k],relative_percent=(b[k]/a[k]-1)*100 if a[k] else None) for k in a}
        guards=dict(psnr=b['psnr_db']>=a['psnr_db']-criteria['max_psnr_loss_db'])
        for metric,name in [('lpips_vgg','lpips'),('flow_error_mean','flow'),('cadence_error_mean','cadence'),('boundary_error','boundary')]:
            if metric in a:guards[name]=b[metric]<=a[metric]*(1+criteria[f'max_{name}_relative_increase'])
        gain=b['psnr_db']>=a['psnr_db']+criteria['min_psnr_gain_db'] or b['lpips_vgg']<=a['lpips_vgg']*(1-criteria['min_lpips_relative_gain']) or b['flow_error_mean']<=a['flow_error_mean']*(1-criteria['min_flow_relative_gain'])
        return dict(changes=changes,guards=guards,gain=gain,aggregate_guard_pass=all(guards.values()) and gain)
    summary={}
    for stage,arms in [('flow_reconstruction',('baseline','distributed_flow')),('budget',('baseline','allocated')),('tracking',('lk','tapnextpp'))]:
        d=stages[stage];summary[stage]=comparison(*[means_with_temporal(d,a) for a in arms])
    for stage in ('adapter','adapter_native'):
        summary[stage]={role:{kind:comparison(v['baseline'],v['adapter']) for kind,v in kinds.items()} for role,kinds in stages[stage]['means'].items()}
    gallery=[]
    native=common.read(root/'adapter_native/plan.json')
    for c in native['cases']:
        dest=root/'review'/f'native_{c["id"]}.png';base=root/'adapter_native'/c['id']
        contact_sheet(Path(c['source_directory']),[(a,base/a/'refined/frames',True) for a in ('baseline','adapter')],dest,c['frames'],f'Native combined_context / {c["id"]}')
        gallery.append(dict(stage='adapter_native',id=c['id'],sheet=str(dest.relative_to(root)),videos=[str((base/a/'refined/reconstruction.mp4').relative_to(root)) for a in ('baseline','adapter')],
            source_video=os.path.relpath(Path(c['source_directory']).parent/'normalized.mp4',root)))
    for j in common.read(root/'budget/plan.json')['jobs']:
        if j['case']['snr']!=6:continue
        c=j['case'];base=Path(j['folder']);dest=root/'review'/f'budget_{c["id"]}.png'
        contact_sheet(common.source_directory(c),[(a,base/a/'frames',True) for a in ('baseline','allocated')],dest,c['frames'],f'Budget / {c["id"]} / 6dB')
        gallery.append(dict(stage='budget',id=c['id'],sheet=str(dest.relative_to(root)),videos=[str((base/a/'reconstruction.mp4').relative_to(root)) for a in ('baseline','allocated')],source_video=os.path.relpath(c['source_movie'],root)))
    for stage,arms in [('flow_reconstruction',('baseline','distributed_flow')),('tracking',('lk','tapnextpp'))]:
        for c in common.pilot_cases():
            base=root/stage/c['id'];dest=root/'review'/f'{stage}_{c["id"]}.png'
            folders=[base/a/'refined/frames' if stage=='flow_reconstruction' else base/a/'frames' for a in arms]
            contact_sheet(common.source_directory(c),[(a,f,True) for a,f in zip(arms,folders)],dest,c['frames'],f'{stage} / {c["id"]}')
            gallery.append(dict(stage=stage,id=c['id'],sheet=str(dest.relative_to(root)),videos=[str((f.parent/'reconstruction.mp4').relative_to(root)) for f in folders],source_video=os.path.relpath(c['source_movie'],root)))
    result=dict(status='COMPLETE_FOUR_STAGE_LOCAL_VALIDATION',summary=summary,
        conclusions=dict(flow='sampling defects corrected; reconstruction guards failed',budget='positive three-source six-condition fallback pilot; consumes additional actual rate under the old 5 percent cap',
            tracking='better synthetic tracking and LPIPS; motion guards failed at identical actual rate',
            adapter='33-frame pilot gains did not transfer to native full-video execution; do not promote'),
        production_default_changed=False,combined_all_four_improvement_claim=False,
        hallucination_mitigation_verified=False,semantic_review='contact sheets and videos are diagnostic, not independent blinded Added/Missing/Distorted ground truth',
        experiment_sizes=dict(flow_audit_sources=36,flow_audit_segments=199,flow_generation_pairs=5,budget_sources=3,budget_conditions=6,
            tracker_pairs=5,adapter_training_sources=19,adapter_training_intervals=30,adapter_training_steps=600,
            adapter_pilot_eval_sources=8,adapter_native_eval_sources=8,adapter_native_frames_per_arm=2253),
        audit=common.read(root/'AUDIT.json'),gallery=gallery,limitations=[
            'Single generation seed; small purposive cohorts, no population-level significance claim.',
            'Flow and tracking use five previously diagnosed development sources.',
            'Budget fallback improves formerly unsent conditions; actual channel uses increase 3.44-4.96 percent of base.',
            'TAPNext++ is compared with LK, not every candidate tracker from the literature.',
            'The adapter is a small new prototype, not a reproduction of Tora or other published full models.',
            'Native evaluation reuses the same eight heldout pilot sources; no checkpoint or strength tuning.',
            'Motion metrics use Farneback flow and are diagnostic proxies, not action-semantic ground truth.',
            'Budget/tracking/flow comparisons use matched MP4 encoding; adapter comparisons use lossless PNG arrays. Do not compare absolute scores between stages.'])
    testxml=root/'tests.xml'
    if testxml.exists():
        import xml.etree.ElementTree as ET
        suites=ET.parse(testxml).getroot().findall('testsuite');result['tests']={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    if (root/'visual_review.json').exists():result['visual_review']=common.read(root/'visual_review.json')
    write_json(root/'RESULT.json',result)
    write_json(common.REPO/'docs/validation/2026-10-15-bsu-motion-validation.json',result)
    lines=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>BSU 4단계 로컬 검증</title>',
        '<style>body{font:16px system-ui;max-width:1240px;margin:32px auto;padding:0 20px;line-height:1.6}table{border-collapse:collapse;width:100%}td,th{border:1px solid #bbb;padding:10px}img{max-width:100%}video{width:32%}code{background:#eee;padding:4px}</style>',
        '<h1>BSU 4단계 로컬 검증</h1><p>예산 배분의 제한적인 이득을 확인했습니다. 동작량 변경·추적기 교체·생성 어댑터는 전체 품질 기준을 통과하지 못했습니다. 기본 실행 모델은 변경하지 않았습니다.</p>',
        '<p>모든 비교는 combined_context를 사용했습니다. 생성 비교는 수신 키프레임·조건·실제 잡음 텐서를 일치시켰습니다. 예산 배분은 같은 상한 안에서 실제 전송량이 늘어납니다. 각 단계는 독립적인 비교이며 네 방법을 모두 합친 모델의 성능을 주장하지 않습니다.</p>',
        '<table><tr><th>단계</th><th>PSNR 변화</th><th>LPIPS 변화</th><th>흐름 오차 변화</th><th>판정</th></tr>']
    named=[('동작량',summary['flow_reconstruction']),('예산 배분',summary['budget']),('TAPNext++',summary['tracking']),
        ('어댑터 전체 WebVid',summary['adapter_native']['heldout_webvid']['combined_context']),('어댑터 전체 Kinetics',summary['adapter_native']['heldout_kinetics']['combined_context'])]
    for name,s in named:
        d=s['changes'];lines.append(f'<tr><td>{name}</td><td>{d["psnr_db"]["delta"]:+.3f} dB</td><td>{d["lpips_vgg"]["relative_percent"]:+.2f}%</td><td>{d["flow_error_mean"]["relative_percent"]:+.2f}%</td><td>{"평균 기준 통과 / 제한적 후보" if s["aggregate_guard_pass"] else "채택 보류"}</td></tr>')
    lines+=['</table><p>LPIPS·흐름 오차는 작을수록 좋습니다. 프레임별 품질 평균과 영상별 평균은 역할이 다릅니다. 여기서는 영상·조건별 값을 동일 가중치로 평균했습니다.</p>',
        '<h2>범위와 해석</h2><p>어댑터는 19개 WebVid 영상의 30개 구간으로 600회 학습했습니다. 33프레임·288×160 평가의 이득이 576×320·24fps·전체 구간 연결에서는 사라졌습니다. 학습/배포 해상도·시간 밀도·조건 위치 차이가 원인 후보지만 개별 기여도는 분리 검증하지 않았습니다.</p>',
        '<p>정지율이 낮아져도 원본과 다른 움직임이 늘면 개선이 아닙니다. 추적 정확도가 좋아도 현재 보정기의 합성 결과가 좋아진다고 볼 수 없습니다. 내용·동작 환각 감소는 독립 의미 검수가 없어 확인되지 않았습니다.</p>',
        '<p><a href="RESULT.json">전체 수치와 한계</a> · <a href="AUDIT.json">입력·잡음·전송량 검증</a> · <a href="artifact_manifest.json">결과물 SHA256 목록</a></p>',
        '<h2>비교 영상</h2><p>각 행: 원본 / 기준 / 후보. 재생 버튼은 세 영상을 같은 시각에서 재생합니다. 정지 화면은 동일한 네 시점의 표본입니다.</p>']
    if 'visual_review' in result:
        lines.append('<p>표본 시각 검수: '+html.escape(result['visual_review']['summary_ko'])+'</p>')
    for i,g in enumerate(gallery):
        lines.append(f'<h3>{html.escape(g["stage"]+" / "+g["id"])}</h3><img loading="lazy" src="{g["sheet"]}">')
        lines.append(f'<div id="pair{i}">'+''.join(f'<video controls preload="metadata" src="{html.escape(v)}"></video>' for v in [g['source_video'],*g['videos']])+f'</div><button onclick="play({i})">같이 재생</button> <button onclick="pause({i})">모두 정지</button>')
    lines.append('<script>function play(i){let v=[...document.querySelectorAll(`#pair${i} video`)];let t=v[0].currentTime;v.forEach(x=>{x.currentTime=t;x.play()})}function pause(i){document.querySelectorAll(`#pair${i} video`).forEach(x=>x.pause())}</script></html>')
    (root/'report.html').write_text('\n'.join(lines));print(root/'report.html')


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('audit','verify','summarize'));p.add_argument('--root',type=Path,default=common.ROOT)
    a=p.parse_args();globals()[a.stage](a.root)


if __name__=='__main__':main()
