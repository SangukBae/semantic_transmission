"""Source-reference diagnosis and anonymous evaluation, separate from extraction."""
from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path
import shutil

from . import auto_extraction as ae,qwen35_event_pairs as ep,qwen35_state_delta_resume as recovery
from . import qwen35_feedback_eval as ev
from .qwen35_state_delta_eval import freeze_write,role_costs,audit_syntax_recovery
from .artifacts import sha256,write_json

CODES=dict(pairs='M36',overlap_pairs='M84')
LABELS=dict(baseline='이전 79키 개별 점수 선택',pairs='사건 전후 쌍 보존',overlap_pairs='겹친 관찰 + 사건 쌍 보존')


def diagnose(root):
    """Reference-only feasibility analysis; never called by the extractor."""
    plan=ep.check_plan(root);parent=Path(plan['parent_root'])
    protocol=ae.read(parent/'evaluation/protocol.json');reference=ae.read(protocol['reference_path'])
    if sha256(protocol['reference_path'])!=protocol['reference_sha256']:raise ValueError('diagnostic reference changed')
    observations=ae.unseal(parent/'ledger.json')['observations'];old=ae.unseal(parent/'selection.json')
    seen=sorted({i for row in observations for i in row['indices']});pairs=ep.source_pairs(observations)
    source=dict(original_observed_frame_count=len(seen),
        events_whose_before_and_after_were_among_observed_source_images=ev.event_coverage(seen,reference),
        original_model_pairs=len(pairs),complete_selected_model_pairs=sum(p['before'] in old['keys'] and p['after'] in old['keys'] for p in pairs),
        one_side_selected_model_pairs=sum((p['before'] in old['keys'])!=(p['after'] in old['keys']) for p in pairs),
        note='Observed images do not prove correct recognition. Model proposals are not reference events.')
    if not (root/'source_pair_diagnostic.json').exists():write_json(root/'source_pair_diagnostic.json',source)
    if not (root/'diagnosis.json').exists():
        write_json(root/'diagnosis.json',dict(status='MECHANICAL_SOURCE_AND_SELECTION_DIAGNOSIS',
            reference_sha256=protocol['reference_sha256'],source=source,
            missed_events=[e for e in ev.event_coverage(old['keys'],reference)['events'] if not e['covered']],
            semantic_cause_review='Additional native-frame review required; availability alone is not recognition'))
    path=root/'reference_feasibility.json'
    if not path.exists():
        import scipy,numpy as np
        from scipy.optimize import milp,Bounds,LinearConstraint
        from scipy.sparse import lil_matrix
        n,gap=plan['frames'],plan['max_gap']
        intervals=[(i,i+gap-1) for i in range(1,n-gap+1)] + [(0,0),(n-1,n-1)]
        intervals.extend(tuple(e[key]) for e in reference['events'] for key in ['before_range','after_range'])
        matrix=lil_matrix((len(intervals),n))
        for j,(a,b) in enumerate(intervals):matrix[j,a:b+1]=1
        solution=milp(np.ones(n),integrality=np.ones(n),bounds=Bounds(0,1),
            constraints=LinearConstraint(matrix.tocsc(),np.ones(len(intervals)),np.full(len(intervals),np.inf)),
            options=dict(time_limit=30,mip_rel_gap=0))
        if not solution.success:raise RuntimeError('reference-only minimum not proven')
        keys=np.flatnonzero(solution.x>.5).tolist()
        write_json(path,dict(status='OPTIMAL_REFERENCE_ONLY_DIAGNOSTIC',oracle=True,extractor_input=False,
            reference_sha256=protocol['reference_sha256'],scipy_version=scipy.__version__,minimum_keys=len(keys),
            lower_bound=solution.mip_dual_bound,mip_gap=solution.mip_gap,diagnostic_keys=keys,
            event_coverage=ev.event_coverage(keys,reference),max_gap=gap,
            warning='Uses reference intervals; never provided to extraction. This is not a model result.'))
    return dict(source=source,reference_only_feasibility=ae.read(path))


def prepare(root):
    plan=ep.check_plan(root);parent=Path(plan['parent_root']);dest=root/'evaluation'
    oldprotocol=ae.read(parent/'evaluation/protocol.json');reference=ae.read(oldprotocol['reference_path'])
    if sha256(oldprotocol['reference_path'])!=oldprotocol['reference_sha256']:raise ValueError('reference changed')
    protocol=dict(oldprotocol,plan_signature=plan['signature'],
        caption_judgment='Anonymous sequential 10-second method batches, not simultaneous randomized judging. Source must be checked.',
        test_scope='Development repair on the same previously analyzed video; not held-out evidence.',
        causal_caveat='Caption prompt, sampling and original observation notes remain fixed. Selection changes source intervals. Overlap arm adds 29 observations only to selection.',
        reference_in_extractor=False,judged_output_tuning_forbidden=True)
    freeze_write(dest/'protocol.json',protocol)
    freeze_write(dest/'private_method_map.json',dict(M71='baseline',**{v:k for k,v in CODES.items()}))
    freeze_write(dest/'agent_provenance.json',dict(evaluator_model='gpt-6-astra',
        agents=['/root/astra_reference','/root/astra_judge_late'],invocation='existing user-requested Astra reference reviewers',
        historical_provenance_path=str(parent/'evaluation/agent_provenance.json'),
        historical_provenance_sha256=sha256(parent/'evaluation/agent_provenance.json'),reference_sha256=protocol['reference_sha256']))
    # Preserve historical packets/judgments verbatim, not a new judgment pass.
    for p in (parent/'evaluation/packets').glob('M71_*.json'):freeze_write(dest/'packets'/p.name,ae.read(p))
    for p in (parent/'evaluation/judgments').glob('M71_*.json'):freeze_write(dest/'judgments'/p.name,ae.read(p))
    for arm,code in CODES.items():
        captions=(ae.unseal(root/f'{arm}.json')['captions'] if (root/f'{arm}.json').exists() else
                  [ae.read(p) for p in sorted((root/'partial'/arm).glob('*.json'))])
        if any(row['segment']!=i for i,row in enumerate(captions)):
            raise ValueError('noncontiguous caption extraction')
        for lo in range(0,60,10):
            if not captions or captions[-1]['end_exclusive']<min((lo+10)*24,1439):continue
            bins=[]
            for ref in reference['bins'][lo:lo+10]:
                caps=[dict(caption_id=f'{code}_{r["segment"]:03d}',start=r['start'],end_exclusive=r['end_exclusive'],caption=r['caption'])
                      for r in captions if r['start']<ref['end_exclusive'] and ref['start']<r['end_exclusive']]
                bins.append(dict(reference=ref,candidates=[dict(method=code,captions=caps)]))
            packet=dict(evaluator_model='gpt-6-astra',reference_sha256=protocol['reference_sha256'],
                source_frames=plan['source_frames'],fps=24,frames=1440,bins=bins,instructions=protocol)
            freeze_write(dest/'packets'/f'{code}_{lo:02d}_{lo+9:02d}.json',packet)
    files=[root/'plan.json',dest/'protocol.json',dest/'agent_provenance.json',Path(protocol['reference_path']),
           *sorted((dest/'packets').glob('*.json'))]
    mp=dest/'input_manifest.json';manifest=ae.read(mp) if mp.exists() else {}
    for p in files:
        if str(p) in manifest and manifest[str(p)]!=sha256(p):raise ValueError('evaluation input changed')
        manifest[str(p)]=sha256(p)
    write_json(mp,manifest)
    truth={};judgments={}
    for p in (dest/'packets').glob('*.json'):
        for row in ae.read(p)['bins']:
            for candidate in row['candidates']:truth[(row['reference']['id'],candidate['method'])]=(candidate,p)
    for p in (dest/'judgments').glob('M71_*.json'):
        for row in ae.read(p)['judgments']:judgments[row['bin_id']]=(row,p)
    reused=[];pending={}
    for code in CODES.values():
        for i in range(60):
            if (i,code) not in truth:continue
            old,_=truth[(i,'M71')];new,packet=truth[(i,code)]
            if ev.candidate_identity(old)!=ev.candidate_identity(new):
                pending.setdefault(packet.name,[]).append(i);continue
            row,p=judgments[i];mapping={a['caption_id']:b['caption_id'] for a,b in zip(old['captions'],new['captions'])}
            reused.append(dict(row,method=code,unsupported_claims=[dict(e,caption_id=mapping[e['caption_id']]) for e in row['unsupported_claims']],
                reuse_of=dict(path=str(p),sha256=sha256(p),method='M71',bin_id=i,caption_id_mapping=mapping),
                evaluation_mode='exact caption and time identity; historical actual Astra labels reused'))
    if reused:write_json(dest/'judgments/exact_input_reuse.json',dict(evaluator_model='gpt-6-astra',
        reference_sha256=protocol['reference_sha256'],packet_sha256s=[sha256(p) for p in sorted((dest/'packets').glob('*.json'))],judgments=reused))
    write_json(dest/'pending.json',dict(changed_bins_by_packet=pending,exact_reused_pairs=[(x['method'],x['bin_id']) for x in reused]))
    # Key coverage is computed only here, after each selection has been frozen.
    metrics={}
    for arm in ep.ARMS:
        p=root/f'{arm}_selection.json'
        if p.exists():metrics[arm]=ev.event_coverage(ae.unseal(p)['keys'],reference)
    write_json(dest/'keyframe_metrics.json',metrics)
    return dict(ready_new_packets=len(pending),pending=pending,exact_reused=len(reused),keyframes=metrics)


def aggregate(root):
    recovery.install_normalizer();integrity=ep.verify(root);prepare(root)
    diagnose(root)
    dest=root/'evaluation';plan=ep.check_plan(root);parent=Path(plan['parent_root'])
    rows=ev.validate_judgments(root)
    if len(rows)!=180:raise ValueError('three methods times 60 bins required')
    protocol=ae.read(dest/'protocol.json');reference=ae.read(protocol['reference_path'])
    algorithm_audit=audit_selection(root,plan)
    bundles=dict(baseline=ae.unseal(parent/'budget_state.json'),**{a:ae.unseal(root/f'{a}.json') for a in ep.ARMS})
    metrics={}
    for arm,code in dict(baseline='M71',**CODES).items():
        bundle=bundles[arm];selected=[r for r in rows if r['method']==code]
        metrics[arm]=dict(keyframes=len(bundle['keys']),captions=len(bundle['captions']),
            error_bins=[r['bin_id'] for r in selected if r['unsupported_claims']],
            recalled_facts=sum(len(r['covered_fact_ids']) for r in selected),reference_facts=123,
            event_coverage=ev.event_coverage(bundle['keys'],reference),costs_by_role=role_costs(bundle),
            generation_seconds=sum(x['generation_seconds'] for x in role_costs(bundle).values()))
        from . import qwen35_state_delta as sd
        observations=ae.unseal(parent/'ledger.json')['observations']
        notes=[sd.local_notes(observations,c['source_indices']) for c in bundle['captions']]
        metrics[arm]['caption_words']=sum(len(c['caption'].split()) for c in bundle['captions'])
        metrics[arm]['empty_source_note_intervals']=sum(not n['facts'] and not n['events'] for n in notes)
    comparisons={}
    for arm in ep.ARMS:
        base=metrics['baseline'];new=metrics[arm]
        comparisons[arm]=dict(resolved_error_bins=sorted(set(base['error_bins'])-set(new['error_bins'])),
            new_error_bins=sorted(set(new['error_bins'])-set(base['error_bins'])),
            recalled_fact_delta=new['recalled_facts']-base['recalled_facts'],
            recovered_events=[b['event_id'] for a,b in zip(base['event_coverage']['events'],new['event_coverage']['events']) if b['covered'] and not a['covered']],
            lost_events=[b['event_id'] for a,b in zip(base['event_coverage']['events'],new['event_coverage']['events']) if a['covered'] and not b['covered']])
    syntax=dict(current=audit_syntax_recovery(root),reused_parent=audit_syntax_recovery(parent))
    result=dict(status='COMPLETE_EVENT_PAIR_DEVELOPMENT_EVALUATION',video_id=plan['video_id'],
        signature=plan['signature'],reference_sha256=protocol['reference_sha256'],evaluator_model='gpt-6-astra',
        metrics=metrics,comparisons=comparisons,extraction=integrity,syntax_recovery=syntax,
        algorithm_audit=algorithm_audit,
        reused_judgments=ae.read(dest/'pending.json')['exact_reused_pairs'],
        historical_result_path=str(parent/'evaluation/RESULT.json'),historical_result_sha256=sha256(parent/'evaluation/RESULT.json'),
        diagnosis_path=str(root/'diagnosis.json'),diagnosis_sha256=sha256(root/'diagnosis.json'),
        feasibility_path=str(root/'reference_feasibility.json'),feasibility_sha256=sha256(root/'reference_feasibility.json'),
        source_pair_diagnostic=ae.read(root/'source_pair_diagnostic.json'),
        judge_consistency=ae.read(dest/'judge_consistency.json') if (dest/'judge_consistency.json').exists() else None,
        limitations=['Known single development video with AI source reference','Frozen reference informed error diagnosis; no reference coordinates were extractor inputs',
                    'Sequential anonymous judging, not simultaneous randomization','Caption intervals and sampled frames change with selection',
                    'No human, unseen-video, receiver or actual transmission validation','Successful-generation sums include required cached calls, exclude loading/failures/I/O'],
        human_ground_truth=False,unseen_video_validation=False,reconstruction_evaluated=False,training=False)
    records=[ae.unseal(p) for p in (root/'calls').glob('*/*.json')];failures=[ae.unseal(p) for p in (root/'failed_attempts').glob('*/*.json')]
    result['actual_new_inference']=dict(successful_calls=len(records),successful_generation_seconds=sum(r['seconds'] for r in records),
        failed_calls=len(failures),failed_generation_seconds=sum(r['seconds'] for r in failures),execution=ae.read(root/'execution.json'))
    monitor=root/'device_monitor.jsonl'
    if monitor.exists():
        samples=[json.loads(line) for line in monitor.read_text().splitlines() if line.strip()]
        result['sampled_device_usage']=dict(samples=len(samples),peak_used_mib=max(r['used_mib'] for r in samples),
            min_free_mib=min(r['free_mib'] for r in samples),scope='2-second samples, device-wide including other processes, begins after initial loading')
    write_json(dest/'RESULT.json',result)
    for arm in ep.ARMS:
        bundle=bundles[arm];out=root/'conditions'/arm;images=out/'selected_frames';images.mkdir(parents=True,exist_ok=True)
        for k in bundle['keys']:
            source=Path(plan['source_frames'])/f'{k}.png';target=images/f'{k:05d}.png'
            if not target.exists():shutil.copyfile(source,target)
            if sha256(target)!=plan['source_frame_hashes'][str(k)]:raise ValueError('exported frame changed')
        write_json(out/'keyframes.json',dict(indices=bundle['keys'],fps=24))
        write_json(out/'captions.json',dict(model_id=plan['model_id'],quantization=plan['quantization'],captions=bundle['captions']))
        write_json(out/'provenance.json',dict(signature=plan['signature'],source_sha256=plan['normalized_sha256'],
            extraction_sha256=sha256(root/f'{arm}.json'),receiver_import_compatible=False,reconstruction_executed=False))
    report(root,result,rows,reference,bundles,plan)
    return result


def audit_selection(root,plan):
    """Replay the frozen optimizer and validate the overlap observation requests."""
    from . import qwen35_state_delta as sd
    parent=Path(plan['parent_root']);original=ae.unseal(parent/'ledger.json')['observations']
    old=ae.unseal(parent/'selection.json');results={}
    extra=ae.unseal(root/'overlap_ledger.json')
    records=[ae.unseal(c['path']) for c in extra['calls']]
    for obs in extra['observations']:
        ids=obs['indices'];prompt=sd.explicit_source_examples(plan['prompts']['observe'],ids)
        acceptable=[prompt,sd.explicit_source_examples(prompt,ids)]
        matching=[r for r in records if r['identity']['ids']==ids and r['result']==obs['result']
                  and any(r['identity']['prompt'].startswith(p+'\nVALID SOURCE') for p in acceptable)]
        if len(matching)!=1:raise ValueError('overlap observation differs from actual source/prompt')
    for arm in ep.ARMS:
        selection=ae.unseal(root/f'{arm}_selection.json')
        observations=original+(extra['observations'] if arm=='overlap_pairs' else [])
        pairs=ep.source_pairs(observations)
        keys,stats=ep.paired_keys(plan['frames'],plan['key_budget'],pairs,old['keys'],old['details']['pixel_novelty'],plan['max_gap'])
        if keys!=selection['keys'] or stats['weighted_pair_coverage']!=selection['solver']['weighted_pair_coverage']:
            raise ValueError('frozen pair solver replay mismatch')
        results[arm]=dict(status='PASS_EXACT_OPTIMIZER_REPLAY',weighted_pair_coverage=stats['weighted_pair_coverage'],
            complete_pairs=len(stats['covered_pairs']),total_pairs=len(pairs),retained_prior_keys=stats['retained_prior_keys'])
    return dict(status='PASS_SOURCE_PAIRS_OVERLAP_PROMPTS_AND_OPTIMAL_SELECTION',arms=results,
                original_observations=len(original),new_fixed_overlap_windows=len(extra['observations']),
                caption_notes_identical_policy=True,reference_input=False)


def report(root,result,rows,reference,bundles,plan):
    esc=html.escape
    body=['<!doctype html><meta charset="utf-8"><title>사건 쌍 선택 검증</title>',
          '<style>body{font-family:system-ui;max-width:1400px;margin:30px auto;color:#172335}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #bbc}section{padding:20px;background:#f5f7fa;margin:20px 0}img{width:31%;margin:4px}.error{color:#a11}</style>',
          '<h1>79개 키프레임에서 사건 전후 보존 실험</h1>',
          '<p>Qwen3.5-9B NF4 하나. 원본 60초·동결 Astra 기준. 기존 캡션 규칙과 관찰 메모를 고정하고 선택만 변경. 미사용 영상·사람 정답·복원 검증 아님.</p>',
          '<p>사건 좌표를 아는 진단 계산에서는 64키로 18사건과 최대 1초 간격을 충족할 수 있었다. 이는 정답을 쓰는 가능성 진단이며 추출 결과가 아니다. 추출기는 이 좌표를 읽지 않는다.</p>',
          '<table><tr><th>조건</th><th>키</th><th>사건/18</th><th>오류/60</th><th>사실/123</th><th>성공 생성 분</th></tr>']
    for arm,m in result['metrics'].items():
        body.append(f'<tr><td>{esc(LABELS[arm])}</td><td>{m["keyframes"]}</td><td>{m["event_coverage"]["covered"]}</td><td>{len(m["error_bins"])}</td><td>{m["recalled_facts"]}</td><td>{m["generation_seconds"]/60:.1f}</td></tr>')
    body.append('</table><p>시간은 조건별로 필요한 성공 호출의 생성 합계이며 재사용 호출을 포함한다. 로딩·실패·입출력을 제외하며 실시간 처리나 중단 없는 전체 시간 측정이 아니다.</p>')
    source=result['source_pair_diagnostic']
    body.append('<h2>원인과 수정 결과</h2><ul>')
    body.append('<li>'+esc(f"기존 관찰 {source['original_observed_frame_count']}프레임에는 기준 18개 사건의 전후 장면이 모두 있었다. 관측 해상도 자체보다 사건 인식·시점 기록·선택 과정에 누락이 있다.")+'</li>')
    body.append('<li>'+esc(f"Qwen이 제안한 중복 합친 {source['original_model_pairs']}쌍 중 기존 선택은 {source['complete_selected_model_pairs']}쌍만 양쪽을 남기고, {source['one_side_selected_model_pairs']}쌍은 한쪽만 남겼다. 이 모델 제안 쌍 수는 실제 정답 18사건과 다르다.")+'</li>')
    body.append('<li>원본 992프레임에는 개의 일부가 이미 보이지만 Qwen은 이를 등장 전으로 기록했다. 사건 쌍을 보존해도 부정확한 시점 자체는 고쳐지지 않는다.</li>')
    body.append('<li>1344프레임은 기존 관찰 창의 끝이자 다음 창의 시작이었다. 겹친 창에서는 1340→1344의 개 등장이 새로 기록됐다. 그러나 추가 후보가 유한한 예산에서 경쟁해 다른 사건 포착도 잃었다.</li>')
    for arm,comparison in result['comparisons'].items():
        body.append('<li>'+esc(LABELS[arm]+': 되찾은 사건 '+', '.join(comparison['recovered_events'])+'; 새로 놓친 사건 '+', '.join(comparison['lost_events'])+f"; 사실 재현 변화 {comparison['recalled_fact_delta']:+d}.")+'</li>')
    body.append('</ul><p>선택기가 잘 보존한 키프레임을 캡션이 정확히 설명하는 것은 별도 문제다. 예를 들어 55초에서는 기준 캡션의 개가 새 캡션에서 어둡고 흐린 물체로 바뀌었다. 오류 감소와 정보 보존을 함께 확인해야 한다.</p>')
    grid=root/'diagnostic_source_grid.jpg'
    if grid.exists():
        data=base64.b64encode(grid.read_bytes()).decode()
        body.append('<img style="width:100%" alt="원본 사건 전후 진단 프레임" src="data:image/jpeg;base64,'+data+'">')
    bykey={(r['bin_id'],r['method']):r for r in rows}
    for ref in reference['bins']:
        body.append(f'<section id="bin-{ref["id"]}"><h2>{ref["id"]}–{ref["id"]+1}초</h2>')
        for i in [ref['start'],(ref['start']+ref['end_exclusive']-1)//2,ref['end_exclusive']-1]:
            data=base64.b64encode((Path(plan['source_frames'])/f'{i}.png').read_bytes()).decode()
            body.append(f'<img title="원본 {i}" src="data:image/png;base64,{data}">')
        body.append('<p>기준 사실: '+esc(json.dumps(ref['facts'],ensure_ascii=False))+'</p>')
        for arm,code in dict(baseline='M71',**CODES).items():
            judge=bykey[(ref['id'],code)];body.append('<h3>'+esc(LABELS[arm])+'</h3>')
            for cap in bundles[arm]['captions']:
                if cap['start']<ref['end_exclusive'] and ref['start']<cap['end_exclusive']:
                    body.append(f'<p>{cap["start"]/24:.3f}–{cap["end_exclusive"]/24:.3f}s: '+esc(cap['caption'])+'</p>')
            body.append('<p>재현: '+esc(', '.join(judge['covered_fact_ids']))+'</p>')
            for error in judge['unsupported_claims']:body.append('<p class="error">'+esc(error['quote']+' — '+error['reason'])+'</p>')
            if judge['uncertain_claims']:body.append('<p>불확실: '+esc(json.dumps(judge['uncertain_claims'],ensure_ascii=False))+'</p>')
        body.append('</section>')
    path=root/'evaluation/report.html';path.write_text('\n'.join(body),encoding='utf-8')
    write_json(root/'evaluation/report_manifest.json',dict(report_sha256=sha256(path),result_sha256=sha256(root/'evaluation/RESULT.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','diagnose','report']);p.add_argument('--root',type=Path,default=ep.ROOT)
    a=p.parse_args();print(json.dumps(dict(prepare=prepare,diagnose=diagnose,report=aggregate)[a.action](a.root.resolve()),ensure_ascii=False))
