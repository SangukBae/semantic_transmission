"""Cross-author source audit, anonymous grading and paired caption comparison."""
import argparse
from collections import Counter
from copy import deepcopy
from html import escape
import json
from pathlib import Path
import random
from semantic_transmission.qwen35_grounded_eval import RUN,DATA,ARMS,verify,validate_record
from semantic_transmission.qwen35_training_data import read_json,write_json,sha256
from evaluate_qwen35_astra_transfer import digest,validate_source_review
from evaluate_qwen35_oracle_diagnostic import validate_mapping


def validate_reference_audit(packet,doc):
    rows={r['id']:r for r in packet['records']};records={r['id']:r for r in doc['records']}
    if len(records)!=len(doc['records']) or set(rows)!=set(records):raise ValueError('Missing/duplicate source audit')
    if not {r['sheet'] for r in rows.values()}<=set(doc['viewed_sheet_paths']):raise ValueError('Source images not inspected')
    for rid,row in rows.items():
        facts={f['id']:f for f in records[rid]['facts']}
        if len(facts)!=len(records[rid]['facts']) or set(facts)!={f['id'] for f in row['facts']}:raise ValueError('Missing fact audit')
        allowed={f['index'] for f in row['frames']}
        for fact in facts.values():
            if fact['status'] not in ['supported','partial','unsupported'] or not fact['reason'] or not fact['source_indices']:raise ValueError('Invalid source audit')
            if not set(fact['source_indices'])<=allowed:raise ValueError('Audit cites unseen frame')
    return records


def audit_reference():
    files={};records=[]
    for a in ['A','B']:
        path=RUN/f'reference_audit_{a}.json';d=read_json(path);p=RUN/f'reference_packet_{a}.json'
        if d['author']!=a or d['annotator_model']!='gpt-6-astra' or d['packet_sha256']!=sha256(p) or not d['source_only']:raise ValueError('Bad reference provenance')
        packet_obj=read_json(p)
        for row in packet_obj['records']:
            if sha256(row['sheet'])!=row['sheet_sha256']:raise ValueError('Changed source review sheet')
        validate_reference_audit(packet_obj,d);records+=d['records'];files[p.name]=sha256(p);files[path.name]=sha256(path)
    obj=dict(files=files,records=records,policy='Original frozen78 facts remain primary; independently source-supported subset is sensitivity only. Audits precede candidate review.')
    path=RUN/'reference_audit_freeze.json'
    if path.exists() and read_json(path)!=obj:raise ValueError('Changed frozen source audit')
    write_json(path,obj);print('REFERENCE_AUDIT',dict(Counter(f['status'] for r in records for f in r['facts'])))
    return obj


def verify_generation():
    p=verify();f=read_json(RUN/'generation_freeze.json')
    if f['protocol_sha256']!=sha256(RUN/'protocol.json'):raise ValueError('Wrong generation protocol')
    expected={f'records/{r["id"]}.json' for r in p['rows']}
    if set(f['files'])!=expected:raise ValueError('Incomplete generation')
    for row in p['rows']:
        rel=f'records/{row["id"]}.json'
        if sha256(RUN/rel)!=f['files'][rel]:raise ValueError('Changed output')
        validate_record(row,read_json(RUN/rel),f['protocol_sha256'])
    return p


def packet():
    p=verify_generation();audit_reference();rng=random.Random(20261006)
    refs={r['id']:r for s in ['validation','test'] for r in read_json(DATA/f'{s}.json')['records']}
    assignment={};sheets={}
    for a in ['A','B']:
        for row in read_json(RUN/f'reference_packet_{a}.json')['records']:assignment[row['id']]=a;sheets[row['id']]=row
    parts={'A':[],'B':[]};mapping={}
    for row in p['rows']:
        rid=row['id'];cells=read_json(RUN/'records'/f'{rid}.json')['cells'];order=list(ARMS);rng.shuffle(order);dedup={}
        for name in order:
            c=cells[name];identity=(tuple(c['keys']),c['caption'])
            if identity not in dedup:
                label=f'g{len(mapping):03d}';dedup[identity]=label;mapping[label]=[]
                parts[assignment[rid]].append(dict(label=label,row_id=rid,split=row['split'],caption=c['caption'],caption_sha256=digest(c['caption']),
                    keys=c['keys'],facts=refs[rid]['facts'],source_sheet=sheets[rid]['sheet'],source_frames=sheets[rid]['frames']))
            mapping[dedup[identity]].append(dict(row_id=rid,cell=name))
    for a,rows in parts.items():
        obj=dict(author=a,records=rows,generation_freeze_sha256=sha256(RUN/'generation_freeze.json'),reference_audit_freeze_sha256=sha256(RUN/'reference_audit_freeze.json'),
            scope='Opposite original source-label author. Same4 images. Condition identities hidden; exact same captions within a window share one judgment. Same-family AI review, not human truth.')
        path=RUN/f'packet_{a}.json'
        if path.exists() and read_json(path)!=obj:raise ValueError('Changed review packet')
        write_json(path,obj)
    write_json(RUN/'mapping_private.json',mapping);write_json(RUN/'all_packet.json',dict(records=parts['A']+parts['B']))
    print('PACKETS',{a:len(v) for a,v in parts.items()})


def validate_grounded_review(packet,doc):
    scores=validate_source_review(packet,doc)
    if not {r['source_sheet'] for r in packet['records']}<=set(doc['viewed_sheet_paths']):raise ValueError('Missing source inspection')
    for row in packet['records']:
        r=scores[row['label']]
        if set(r['fact_evidence'])!={f['id'] for f in row['facts']}:raise ValueError('Missing fact rationale')
        for fid,e in r['fact_evidence'].items():
            if not e['rationale'] or (e['quote'] and e['quote'] not in row['caption']):raise ValueError('Invalid factual quote')
            if r['fact_hits'][fid] and not e['quote']:raise ValueError('Full hit needs actual caption quote')
        for error in r['errors']:
            if error['kind'] not in ['contradicted','unsupported']:raise ValueError('Invalid error category')
    return scores


def comparison_gates(totals):
    gates={}
    for candidate,others in [('caption',['initial']),('grounded',['initial','caption'])]:
        checks={}
        for split in ['validation','test']:
            a=totals[split][candidate]
            for other in others:
                b=totals[split][other];prefix=f'{split}_vs_{other}'
                checks[prefix+'_noninferior']=a['fact_hits']>=b['fact_hits'] and a['action_hits']>=b['action_hits'] and a['error_windows']<=b['error_windows'] and a['generation_failures']<=b['generation_failures']
                checks[prefix+'_strict_gain']=a['fact_hits']>b['fact_hits'] or a['action_hits']>b['action_hits'] or a['error_windows']<b['error_windows']
        gates[candidate]=dict(passed=all(checks.values()),checks=checks)
    return gates


def apply_adjudication(packet,doc,scores):
    """Apply separately preserved, source-cited missed-error decisions."""
    rows={r['label']:r for r in packet['records']};seen=set();updated=deepcopy(scores)
    for r in doc['records']:
        label=r['label']
        if label in seen or label not in rows:raise ValueError('Invalid adjudication label')
        seen.add(label);source=rows[label]
        if r['decision'] not in ['retain','add_error']:raise ValueError('Unsupported adjudication change')
        if not r['quote'] or r['quote'] not in source['caption'] or not r['reason']:raise ValueError('Adjudication needs exact evidence')
        frames={f['index']:f['path'] for f in source['source_frames']}
        if not r['source_indices'] or not set(r['source_indices'])<=set(frames):raise ValueError('Adjudication cites unseen frame')
        if not {frames[i] for i in r['source_indices']}<=set(doc['viewed_frame_paths']):raise ValueError('Adjudication lacks image inspection')
        if r['decision']=='add_error':
            if r['kind'] not in ['contradicted','unsupported']:raise ValueError('Bad adjudication error kind')
            if any(e['quote']==r['quote'] for e in updated[label]['errors']):raise ValueError('Duplicate adjudicated error')
            updated[label]['errors'].append({k:r[k] for k in ['quote','reason','source_indices','kind']})
            updated[label]['rationale']+=' Source reinspection: '+r['reason']
    return updated


def score():
    p=verify_generation();audit=audit_reference();audits={r['id']:r for r in audit['records']};reviews=[];files={}
    for a in ['A','B']:
        path=RUN/f'review_{a}.json';d=read_json(path);pp=RUN/f'packet_{a}.json'
        if d['author']!=a or d['annotator_model']!='gpt-6-astra' or d['packet_sha256']!=sha256(pp):raise ValueError('Bad judge provenance')
        part=read_json(pp)
        if part['generation_freeze_sha256']!=sha256(RUN/'generation_freeze.json') or part['reference_audit_freeze_sha256']!=sha256(RUN/'reference_audit_freeze.json'):raise ValueError('Review uses a different frozen generation/reference')
        validate_grounded_review(part,d);reviews+=d['records'];files[path.name]=sha256(path);files[pp.name]=sha256(pp)
    pack=read_json(RUN/'all_packet.json');scores=validate_grounded_review(pack,dict(records=reviews,viewed_sheet_paths=[r['source_sheet'] for r in pack['records']]))
    mapping=read_json(RUN/'mapping_private.json');generated={(r['id'],a):c for r in p['rows'] for a,c in read_json(RUN/'records'/f'{r["id"]}.json')['cells'].items()}
    validate_mapping(pack,mapping,generated)
    original_errors={s:{a:0 for a in ARMS} for s in ['validation','test']};split_by_id={r['id']:r['split'] for r in p['rows']}
    for label,entries in mapping.items():
        for e in entries:original_errors[split_by_id[e['row_id']]][e['cell']]+=int(bool(scores[label]['errors']))
    adjudications={}
    for author in ['A','B']:
        path=RUN/f'review_{author}_adjudication.json'
        if not path.exists():continue
        doc=read_json(path)
        if doc['author']!=author or doc['annotator_model']!='gpt-6-astra' or doc['parent_review_sha256']!=sha256(RUN/f'review_{author}.json'):raise ValueError('Changed adjudication parent')
        scores=apply_adjudication(read_json(RUN/f'packet_{author}.json'),doc,scores);adjudications[path.name]=sha256(path)
    if adjudications:
        path=RUN/'adjudication_freeze.json';obj=dict(files=adjudications,original_reviews=files)
        if path.exists() and read_json(path)!=obj:raise ValueError('Changed adjudication')
        write_json(path,obj)
    elif (RUN/'adjudication_freeze.json').exists():raise ValueError('Missing frozen adjudication')
    linked={(e['row_id'],e['cell']):scores[label] for label,entries in mapping.items() for e in entries}
    refs={r['id']:r for s in ['validation','test'] for r in read_json(DATA/f'{s}.json')['records']}
    totals={s:{a:Counter() for a in ARMS} for s in ['validation','test']};supported={s:{a:Counter() for a in ARMS} for s in totals};details=[];source_totals={}
    for row in p['rows']:
        rid=row['id'];split=row['split'];ref=refs[rid];facts={f['id'] for f in ref['facts']};actions={f['id'] for f in ref['facts'] if f['kind']!='state'}
        verified={f['id'] for f in audits[rid]['facts'] if f['status']=='supported'};cells={}
        source_totals.setdefault(row['source_id'],{a:Counter() for a in ARMS})
        for arm in ARMS:
            out=generated[rid,arm];review=linked[rid,arm];success=out['status']=='COMPLETE';hits={k for k,v in review['fact_hits'].items() if v} if success else set()
            partial=set(review.get('partial_fact_ids',[])) if success else set();errors=review['errors'];t=totals[split][arm]
            counts=dict(windows=1,facts=len(facts),fact_hits=len(hits),actions=len(actions),action_hits=len(hits&actions),full_or_partial_hits=len(hits|partial),
                error_windows=int(bool(errors)),error_claims=len(errors),contradicted_windows=int(any(e['kind']=='contradicted' for e in errors)),unsupported_windows=int(any(e['kind']=='unsupported' for e in errors)),
                generation_failures=int(not success),caption_words=len(out['caption'].split()),over_80_words=int(len(out['caption'].split())>80),tokens=out['generated_tokens'],generation_seconds=out['seconds'])
            t.update(counts);source_totals[row['source_id']][arm].update(counts)
            supported[split][arm].update(facts=len(verified),fact_hits=len(hits&verified),actions=len(actions&verified),action_hits=len(hits&actions&verified))
            cells[arm]=dict(output=out,review=review)
        details.append(dict(id=rid,source_id=row['source_id'],split=split,facts=ref['facts'],source_audit=audits[rid],cells=cells))
    result=dict(status='COMPLETE_PAIRED_ASTRA_CAPTION_REVIEW',splits=totals,source_supported_sensitivity=supported,gates=comparison_gates(totals),source_totals=source_totals,rows=details,
        source_counts={s:len({r['source_id'] for r in p['rows'] if r['split']==s}) for s in totals},
        initial_definition=p['initial_definition'],optimizer_steps=0,default_pipeline_changed=False,
        training={a:read_json(DATA/a/'RESULT.json')['state'] for a in ['caption','grounded']},
        shared_encoding_seconds=sum(read_json(RUN/'records'/f'{r["id"]}.json')['encode_seconds'] for r in p['rows']),
        limitations=p['limitations'],timing_policy=p['timing_policy'],review_count=len(reviews),
        original_review_error_windows=original_errors,adjudication_files=adjudications,
        adjudication_scope='Source consistency check requested after parent unmasking; adjudicator kept condition identities hidden. Original reviews retained; missed definite object identity added to both matching captions.')
    frozen=dict(files=files,generation_freeze_sha256=sha256(RUN/'generation_freeze.json'),reference_audit_freeze_sha256=sha256(RUN/'reference_audit_freeze.json'))
    if (RUN/'review_freeze.json').exists() and read_json(RUN/'review_freeze.json')!=frozen:raise ValueError('Changed frozen semantic review')
    write_json(RUN/'review_freeze.json',frozen)
    write_json(RUN/'RESULT.json',result);render(result)
    print(json.dumps({k:result[k] for k in ['splits','source_supported_sensitivity','gates','source_counts','review_count']},indent=2))


def render(result):
    parts=['<!doctype html><meta charset="utf-8"><title>Grounded LoRA caption comparison</title>',
        '<style>body{font:16px system-ui;max-width:1450px;margin:24px auto;padding:16px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #bbb;padding:8px;vertical-align:top}img{max-width:100%}details{margin:18px 0}</style>',
        '<h1>추가 학습 전 / 문장 학습 / 근거 연결 학습</h1><p>같은4키, 새 검증3원본12구간·최종평가4원본16구간. initial은 이전 Astra 정답 rank32 모델. 원본 대조 Astra AI 교차 검수, 사람 정답 아님.</p>']
    if result['adjudication_files']:parts.append('<p>오류 수는 원본 재확인 정정 포함. 같은 구조물을 자전거 거치대로 단정한 두 문장의 누락 오류를 반영했다. 최초 test 오류 initial/caption/grounded=2/1/2, 정정 후=2/2/3. 원래 판정 파일은 보존했다.</p>')
    for split,arms in result['splits'].items():
        parts.append(f'<h2>{split}</h2><table><tr><th>조건</th><th>사실</th><th>동작</th><th>완전+부분</th><th>오류 구간</th><th>초</th></tr>')
        for arm,t in arms.items():parts.append(f'<tr><td>{arm}</td><td>{t["fact_hits"]}/{t["facts"]}</td><td>{t["action_hits"]}/{t["actions"]}</td><td>{t["full_or_partial_hits"]}/{t["facts"]}</td><td>{t["error_windows"]}/{t["windows"]}</td><td>{t["generation_seconds"]:.1f}</td></tr>')
        parts.append('</table>')
    for row in result['rows']:
        parts.append(f'<details><summary>{row["id"]} · {row["split"]}</summary><img src="../teacher/sheets/{row["id"]}.png"><ul>')
        for fact in row['facts']:parts.append(f'<li>{fact["id"]}: {escape(fact["text"])}</li>')
        parts.append('</ul><table><tr><th>조건</th><th>출력</th><th>판정</th></tr>')
        for name,c in row['cells'].items():parts.append(f'<tr><td>{name}</td><td>{escape(c["output"]["caption"])}</td><td>{escape(c["review"]["rationale"])}</td></tr>')
        parts.append('</table></details>')
    (RUN/'report.html').write_text('\n'.join(parts))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['audit-reference','packet','score']);args=p.parse_args()
    {'audit-reference':audit_reference,'packet':packet,'score':score}[args.command]()
