"""Cross-annotator source reference, blinded captions, and factorial reporting."""
import argparse
from collections import Counter
from html import escape
import hashlib
import random
from pathlib import Path

from semantic_transmission.qwen35_astra_transfer import RUN,ARMS,verify_teachers,validate_annotation
from semantic_transmission.qwen35_training_data import read_json,write_json,sha256
from semantic_transmission.qwen35_oracle_diagnostic import witnessed
from evaluate_qwen35_oracle_diagnostic import validate_review,validate_mapping


def digest(text):return hashlib.sha256(text.encode()).hexdigest()


def validate_source_review(packet_obj, judged):
    scores=validate_review(packet_obj,judged)
    for row in packet_obj['records']:
        allowed={f['index'] for f in row['source_frames']}
        for error in scores[row['label']]['errors']:
            if any(type(i) is not int or i not in allowed for i in error['source_indices']):
                raise ValueError('Error cites an uninspected source frame')
    return scores


def extension_gates(totals):
    gates={}
    for arm in ['teacher_r8','teacher_r32']:
        control=arm.replace('teacher','control');checks={}
        for split in ['validation','test']:
            a=totals[split][arm]
            for other in ['base',control]:
                b=totals[split][other]
                checks[f'{split}_vs_{other}']=a['fact_hits']>=b['fact_hits'] and a['action_hits']>=b['action_hits'] and a['error_windows']<=b['error_windows'] and a['generation_failures']<=b['generation_failures']
            checks[f'{split}_strict_gain']=any(a[k]>totals[split]['base'][k] for k in ['fact_hits','action_hits']) or a['error_windows']<totals[split]['base']['error_windows']
        gates[arm]=dict(passed=all(checks.values()),checks=checks)
    return gates


def freeze_reference(run):
    verify_teachers(run)
    files={};reference=[]
    for source_author,split,reference_author in [('A','validation','B'),('B','test','A')]:
        packet=read_json(run/'teacher'/f'packet8_{source_author}.json')
        path=run/'teacher'/f'reference8_{source_author}.json';labels=read_json(path)
        if labels['packet_sha256']!=sha256(run/'teacher'/f'packet8_{source_author}.json'):
            raise ValueError('Reference packet changed')
        if labels['annotator_model']!='gpt-6-astra' or not labels['source_only'] or labels['author']!=reference_author:
            raise ValueError('Reference not authored by the opposite annotator')
        by_id={r['id']:r for r in labels['records']}
        if len(by_id)!=len(labels['records']) or set(by_id)!={r['id'] for r in packet['records']}:
            raise ValueError('Incomplete independent reference')
        if not {r['sheet'] for r in packet['records']}<=set(labels['viewed_sheet_paths']):
            raise ValueError('Missing source inspection')
        for row in packet['records']:
            record=validate_annotation(by_id[row['id']],row)
            for fact in record['facts']:
                if type(fact.get('observable_in_four')) is not bool:
                    raise ValueError('Explicit four-frame observability required')
                if fact['observable_in_four'] and not witnessed(fact['evidence_groups'],[0,2,5,7]):
                    raise ValueError('Four-frame fact lacks four-frame evidence')
            reference.append(dict(record,split=split,source_author=reference_author,
                four_frame_fact_ids=[f['id'] for f in record['facts'] if f['observable_in_four']]))
        files[str(path.relative_to(run))]=sha256(path)
    old=run/'reference_freeze.json'
    obj=dict(files=files,teacher_freeze_sha256=sha256(run/'teacher_freeze.json'),
        policy='Source-only eight-frame reference by opposite four-frame teacher author, before candidate access. Explicit four-frame observability; same model family, not human truth.')
    payload={'records':reference}
    if (run/'reference.json').exists() and read_json(run/'reference.json')!=payload:
        raise ValueError('Reference content changed')
    write_json(run/'reference.json',payload)
    obj['reference_sha256']=sha256(run/'reference.json')
    # A second invocation must reproduce the exact frozen object.
    if old.exists():
        previous=read_json(old)
        if previous!=obj:raise ValueError('Frozen reference differs')
    else:write_json(old,obj)
    print('REFERENCE_FROZEN',len(reference),sum(len(r['four_frame_fact_ids']) for r in reference),'4-frame facts')


def verify_generation(run):
    verify_teachers(run)
    rf=read_json(run/'reference_freeze.json')
    if sha256(run/'reference.json')!=rf['reference_sha256']:raise ValueError('Changed reference')
    for p,h in rf['files'].items():
        if sha256(run/p)!=h:raise ValueError('Reference author file changed')
    ev=run/'evaluation';gf=read_json(ev/'freeze.json')
    if sha256(ev/'protocol.json')!=gf['protocol_sha256']:raise ValueError('Generation protocol changed')
    for p,h in gf['files'].items():
        if sha256(ev/p)!=h:raise ValueError('Generation row changed')
    for ck in read_json(ev/'protocol.json')['checkpoints'].values():
        path=Path(ck['path'])
        if sha256(path/'complete.json')!=ck['sha256']:raise ValueError('Changed checkpoint')
        for p,h in read_json(path/'complete.json')['files'].items():
            if sha256(path/p)!=h:raise ValueError('Checkpoint tensor changed')
    return {r['id']:r for r in read_json(run/'reference.json')['records']}


def packet(run):
    refs=verify_generation(run);ev=run/'evaluation';rng=random.Random(20261005)
    authors={};sheets={}
    for author in ['A','B']:
        for r in read_json(run/'teacher'/f'labels4_{author}.json')['records']:authors[r['id']]=r
        for r in read_json(run/'teacher'/f'packet8_{author}.json')['records']:sheets[r['id']]=r
    mapping={};generated={};packets={'A':[],'B':[]}
    for row_id,ref in refs.items():
        row=read_json(ev/'records'/f'{row_id}.json');cells=dict(row['cells'])
        cells['astra_4frames']=dict(caption=authors[row_id]['caption'],keys=[0,2,5,7],status='COMPLETE',seconds=None,generated_tokens=None)
        order=list(cells);rng.shuffle(order);dedup={}
        for cell in order:
            c=cells[cell];generated[(row_id,cell)]=c
            key=(tuple(c['keys']),c['caption'])
            if key not in dedup:
                label=f'c{len(mapping):03d}';dedup[key]=label
                facts=ref['facts'] # Review all facts; score observable subset separately.
                packets['B' if ref['split']=='validation' else 'A'].append(dict(label=label,row_id=row_id,
                    split=ref['split'],caption=c['caption'],caption_sha256=digest(c['caption']),keys=c['keys'],
                    facts=facts,observable_four_fact_ids=ref['four_frame_fact_ids'],source_sheet=sheets[row_id]['sheet'],
                    source_frames=sheets[row_id]['frames'],generation_status=c['status']))
                mapping[label]=[]
            mapping[dedup[key]].append(dict(row_id=row_id,cell=cell))
    for author,records in packets.items():
        obj=dict(author=author,records=records,generation_freeze_sha256=sha256(ev/'freeze.json'),
            reference_freeze_sha256=sha256(run/'reference_freeze.json'),
            scope='Model identities hidden; author did not create the opposite annotator four-frame captions. Same-family AI review, not independent human truth.')
        path=ev/f'packet_{author}.json'
        if path.exists() and read_json(path)!=obj:raise ValueError('Blinded packet changed')
        write_json(path,obj)
    write_json(ev/'mapping_private.json',mapping)
    write_json(ev/'all_packet.json',dict(records=packets['A']+packets['B']))
    print('PACKETS',{k:len(v) for k,v in packets.items()})


def score(run):
    refs=verify_generation(run);ev=run/'evaluation';pack=read_json(ev/'all_packet.json');reviews=[]
    for author in ['A','B']:
        part=read_json(ev/f'packet_{author}.json');judged=read_json(ev/f'review_{author}.json')
        if judged['packet_sha256']!=sha256(ev/f'packet_{author}.json') or judged['annotator_model']!='gpt-6-astra' or judged['author']!=author:raise ValueError('Judge identity mismatch')
        validate_source_review(part,judged);reviews+=judged['records']
    scores=validate_source_review(pack,dict(records=reviews));mapping=read_json(ev/'mapping_private.json')
    teacher={r['id']:r for a in ['A','B'] for r in read_json(run/'teacher'/f'labels4_{a}.json')['records']}
    generated={}
    for row_id in refs:
        cells=read_json(ev/'records'/f'{row_id}.json')['cells']
        for name,c in cells.items():generated[(row_id,name)]=c
        generated[(row_id,'astra_4frames')]=dict(caption=teacher[row_id]['caption'],keys=[0,2,5,7],status='COMPLETE',seconds=None,generated_tokens=None)
    validate_mapping(pack,mapping,generated)
    linked={(entry['row_id'],entry['cell']):scores[label] for label,entries in mapping.items() for entry in entries}
    names=['base',*ARMS,'astra_4frames','base_8frames'];totals={s:{n:Counter() for n in names} for s in ['validation','test']};details=[]
    for row_id,ref in refs.items():
        cells={};eligible=set(ref['four_frame_fact_ids']);events={f['id'] for f in ref['facts'] if f['kind']!='state'}
        for name in names:
            out=generated[(row_id,name)];review=linked[(row_id,name)];success=out['status']=='COMPLETE';t=totals[ref['split']][name]
            hit={f for f,ok in review['fact_hits'].items() if ok} if success else set()
            t.update(windows=1,observable_facts=len(eligible),fact_hits=len(hit&eligible),
                observable_actions=len(eligible&events),action_hits=len(hit&eligible&events),
                all_eight_frame_facts=len(ref['facts']),all_fact_hits=len(hit),
                error_windows=int(bool(review['errors'])),error_claims=len(review['errors']),
                generation_failures=int(not success),caption_words=len(out['caption'].split()),
                generation_seconds=out.get('seconds') or 0)
            cells[name]=dict(output=out,review=review)
        details.append(dict(id=row_id,split=ref['split'],facts=ref['facts'],four_frame_fact_ids=list(eligible),cells=cells))
    gates=extension_gates(totals)
    result=dict(status='COMPLETE_CROSS_ANNOTATOR_ASTRA_PILOT',splits=totals,gates=gates,rows=details,
        training={a:read_json(run/a/'RESULT.json') for a in ARMS},
        temporal_extension_gate_passed=any(x['passed'] for x in gates.values()),default_pipeline_changed=False,
        source_input_budget='Primary facts are independently marked observable in four exact images; eight-frame full-reference recall is secondary.',
        limitations=['One seed, eight new purposively chosen sources; not broad statistical or human-ground-truth validation.',
            'Two preconfigured Astra agents are distinct author/judge contexts but the same model family and shared filesystem.',
            'All four arms use the same76 training inputs and2epochs, but supervision text/token counts differ as intended.',
            'Fixed uniform evaluation keys test caption generation, not learned selection or transmission/reconstruction.'])
    write_json(run/'RESULT.json',result)
    render(run,result)
    print({s:{a:dict(t) for a,t in arms.items()} for s,arms in totals.items()});print('GATES',gates)


def render(run,result):
    parts=['<!doctype html><meta charset="utf-8"><title>Astra teacher transfer pilot</title>',
        '<style>body{font:16px system-ui;max-width:1500px;margin:30px auto;padding:16px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #bbb;padding:8px;vertical-align:top}img{max-width:100%}details{margin:16px 0}</style>',
        '<h1>동일 입력 Astra 정답 × LoRA rank 비교</h1><p>76학습 구간, 새8원본24구간. 고정4키, 같은 입력의 사실/동작 재현. 별도 Astra 작성자·검수자, 동일 모델 계열 AI 평가. 기본 경로 미변경.</p>']
    for split,cells in result['splits'].items():
        parts.append(f'<h2>{split}</h2><table><tr><th>조건</th><th>사실</th><th>동작</th><th>오류 구간</th><th>생성 실패</th><th>평균 단어</th></tr>')
        for name,t in cells.items():parts.append(f'<tr><td>{name}</td><td>{t["fact_hits"]}/{t["observable_facts"]}</td><td>{t["action_hits"]}/{t["observable_actions"]}</td><td>{t["error_windows"]}/{t["windows"]}</td><td>{t["generation_failures"]}</td><td>{t["caption_words"]/t["windows"]:.1f}</td></tr>')
        parts.append('</table>')
    for row in result['rows']:
        parts.append(f'<details><summary>{row["id"]} · {row["split"]}</summary><img src="teacher/sheets8/{row["id"]}.png"><ul>')
        for fact in row['facts']:parts.append(f'<li>{escape(fact["text"])} — 4장 관측: {fact["observable_in_four"]}</li>')
        parts.append('</ul><table><tr><th>조건</th><th>실제 출력</th><th>검수</th></tr>')
        for name,cell in row['cells'].items():parts.append(f'<tr><td>{name}</td><td>{escape(cell["output"]["caption"])}</td><td>{escape(cell["review"]["rationale"])}</td></tr>')
        parts.append('</table></details>')
    (run/'report.html').write_text('\n'.join(parts))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze-reference','packet','score']);args=p.parse_args()
    {'freeze-reference':freeze_reference,'packet':packet,'score':score}[args.command](RUN)
