"""Source-grounded action-only review of the post-hoc prompt-transfer probe."""
import argparse
from collections import Counter
import hashlib
from html import escape
import os
from pathlib import Path
from semantic_transmission.qwen35_caption_repair import RUN, DIAG, verify
from semantic_transmission.qwen35_training_data import ROOT,read_json,write_json,sha256
from evaluate_qwen35_oracle_diagnostic import validate_review,validate_mapping


def digest(s):return hashlib.sha256(s.encode()).hexdigest()


def check(root):
    verify(root);out=root/'action_probe';freeze=read_json(out/'freeze.json');p=read_json(out/'protocol.json')
    if freeze['protocol_sha256']!=sha256(out/'protocol.json') or p['parent_result_sha256']!=sha256(root/'RESULT.json'):raise ValueError('Probe parent/protocol changed')
    if p['implementation_sha256']!=sha256(ROOT/'scripts/probe_qwen35_caption_actions.py'):raise ValueError('Probe implementation changed')
    for name,h in freeze['files'].items():
        if sha256(out/name)!=h:raise ValueError('Probe output changed')
    for row in p['rows']:
        record=read_json(out/'records'/f'{row["id"]}.json')
        if row['split']!='regression' or record['protocol_sha256']!=sha256(out/'protocol.json'):raise ValueError('Non-development or stale row')
        if set(record['cells'])!=set(p['cells']):raise ValueError('Incomplete probe')
        if len({c['input_tokens'] for c in record['cells'].values()})!=1:raise ValueError('Unmatched model inputs')
        for c in record['cells'].values():
            if c['keys']!=row['keys'] or c['source_hashes']!=[row['frames'][i]['sha256'] for i in row['keys']]:raise ValueError('Unmatched source keys')
            if c['status'] not in ('COMPLETE','EMPTY_OR_TRUNCATED'):raise ValueError('Unknown generation status')
    return p


def packet(root):
    p=check(root);out=root/'action_probe';records=[];mapping={}
    for row in p['rows']:
        cells=read_json(out/'records'/f'{row["id"]}.json')['cells'];seen={}
        for name,c in cells.items():
            identity=c['caption']
            if identity not in seen:
                label=f'{row["id"][:3]}-{len(seen)+1}';seen[identity]=label;mapping[label]=[]
                records.append(dict(label=label,row_id=row['id'],caption=c['caption'],caption_sha256=digest(c['caption']),keys=c['keys'],facts=[f for f in row['facts'] if f['kind']=='event']))
            mapping[seen[identity]].append(dict(row_id=row['id'],cell=name))
    write_json(out/'review_packet.json',dict(generation_freeze_sha256=sha256(out/'freeze.json'),records=records,scope='Action-only post-hoc development probe, same assistant source/output review'))
    write_json(out/'review_mapping.json',mapping)
    print('ACTION_PACKET',len(records))


def score(root):
    p=check(root);out=root/'action_probe';packet_obj=read_json(out/'review_packet.json');review=read_json(out/'review.json')
    if packet_obj['generation_freeze_sha256']!=sha256(out/'freeze.json') or review['packet_sha256']!=sha256(out/'review_packet.json'):raise ValueError('Probe review changed')
    scores=validate_review(packet_obj,review);mapping=read_json(out/'review_mapping.json')
    generated={(r['id'],n):c for r in p['rows'] for n,c in read_json(out/'records'/f'{r["id"]}.json')['cells'].items()}
    validate_mapping(packet_obj,mapping,generated)
    mapped={(m['row_id'],m['cell']):scores[l] for l,ms in mapping.items() for m in ms}
    totals={n:Counter() for n in p['cells']};details=[]
    for row in p['rows']:
        cells={}
        for name in p['cells']:
            c=generated[row['id'],name];r=mapped[row['id'],name]
            if c['status']!='COMPLETE' and any(r['fact_hits'].values()):raise ValueError('Failed generation receives no complete-event credit in the conservative probe score')
            totals[name].update(generation_failures=int(c['status']!='COMPLETE'),windows=1,event_count=len(r['fact_hits']),event_hits=sum(r['fact_hits'].values()),partial_events=len(r.get('partial_fact_ids',[])),error_windows=int(bool(r['errors'])),error_claims=len(r['errors']),words=len(c['caption'].split()),seconds=c['seconds'])
            cells[name]=dict(c,review=r)
        details.append(dict(id=row['id'],facts=[f for f in row['facts'] if f['kind']=='event'],cells=cells))
    primary=read_json(root/'RESULT.json')['splits']['regression']
    result=dict(status='COMPLETE_PROVISIONAL_AI_REVIEW',post_hoc=True,split='known_development_regression',conditions=totals,rows=details,
        caption_prompt_comparison={n:{k:primary[n][k] for k in ['event_count','event_hits','partial_events','error_windows','windows']} for n in p['cells']},
        failure_policy='No selective retry; failed or truncated answers receive zero complete-event credit and are counted separately. Post-hoc diagnostic scoring policy.',
        limitation='Different task instruction; action answers are not a validated replacement caption pipeline. No new test or model selection. Same assistant non-independent review. No downstream evaluation.',
        review_sha256=sha256(out/'review.json'),freeze_sha256=sha256(out/'freeze.json'))
    write_json(out/'RESULT.json',result)
    lines=['<!doctype html><meta charset="utf-8"><title>Action prompt transfer probe</title><style>body{font:16px system-ui;margin:24px auto;max-width:1400px}td,th{padding:10px;border:1px solid #bbb;vertical-align:top}table{border-collapse:collapse}img{width:100%}</style><h1>동작 질문 직접 추론: 개발 자료 후속 진단</h1>',f'<p>{escape(result["limitation"])}</p>']
    lines.append('<table><tr><th>모델</th><th>일반 캡션 동작</th><th>동작 질문</th><th>질문 답변 오류 구간</th><th>생성 실패</th></tr>')
    for n,t in totals.items():lines.append(f'<tr><td>{n}</td><td>{primary[n]["event_hits"]}/30</td><td>{t["event_hits"]}/30</td><td>{t["error_windows"]}/14</td><td>{t["generation_failures"]}</td></tr>')
    lines.append('</table>')
    for row in details:
        sheet=os.path.relpath(DIAG/'clips'/f'{row["id"]}_candidates.jpg',out)
        lines.append(f'<h2>{row["id"]}</h2><img src="{escape(sheet)}"><ol>'+''.join(f'<li>{escape(f["text"])}</li>' for f in row['facts'])+'</ol><table>')
        for n,c in row['cells'].items():
            r=c['review'];errors=' '.join(f'{e["quote"]}: {e["reason"]}' for e in r['errors'])
            lines.append(f'<tr><td>{n}</td><td>{escape(c["caption"])}</td><td>{sum(r["fact_hits"].values())}/{len(r["fact_hits"])} {escape(r["rationale"])}<p>{escape(errors)}</p></td></tr>')
        lines.append('</table>')
    (out/'report.html').write_text('\n'.join(lines))
    write_json(out/'verification.json',dict(status='PASS',review_records=len(scores),condition_windows=len(generated),result_sha256=sha256(out/'RESULT.json'),evaluator_sha256=sha256(Path(__file__))))
    print({n:dict(t) for n,t in totals.items()})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['packet','score']);parser.add_argument('--run',type=Path,default=RUN);args=parser.parse_args();(packet if args.command=='packet' else score)(args.run)
