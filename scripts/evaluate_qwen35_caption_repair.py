"""Audit fixed-input caption repair and render source/caption evidence."""
import argparse
from collections import Counter
import hashlib
from html import escape
import os
from pathlib import Path

from semantic_transmission.qwen35_caption_repair import RUN, DIAG, verify
from semantic_transmission.qwen35_training_data import ROOT, read_json, write_json, sha256
from evaluate_qwen35_oracle_diagnostic import validate_review, validate_mapping


def digest(text):return hashlib.sha256(text.encode()).hexdigest()


def generation(run):
    p=verify(run);ev=run/'evaluation';f=read_json(ev/'freeze.json')
    if f['protocol_sha256']!=sha256(ev/'protocol.json'):raise ValueError('Generation protocol changed')
    for name,h in f['files'].items():
        if sha256(ev/name)!=h:raise ValueError('Generation changed')
    for item in read_json(ev/'protocol.json')['checkpoints'].values():
        path=Path(item['path'])
        if sha256(path/'complete.json')!=item['sha256']:raise ValueError('Selected checkpoint changed')
        for name,h in read_json(path/'complete.json')['files'].items():
            if sha256(path/name)!=h:raise ValueError('Checkpoint artifact changed')
    rows=[r for s in ('validation','test','regression') for r in read_json(run/f'{s}.json')['records'] if r['origin']!='legacy_replay']
    for row in rows:
        record=read_json(ev/'records'/f'{row["id"]}.json')
        if record['protocol_sha256']!=sha256(ev/'protocol.json'):raise ValueError('Record protocol differs')
        cells=record['cells']
        if set(cells)!={'base','previous_joint','caption_ce','caption_qa'}:raise ValueError('Incomplete comparison')
        if len({c['input_tokens'] for c in cells.values()})!=1:raise ValueError('Unequal text/visual input')
        if len({str(c['grid']) for c in cells.values()})!=1:raise ValueError('Unequal image grids')
        for c in cells.values():
            if c['keys']!=row['keys'] or c['source_hashes']!=[row['frames'][i]['sha256'] for i in row['keys']]:raise ValueError('Unequal source frames')
            if c['status']!='COMPLETE':raise ValueError('Generation failure cannot be silently scored')
    return p,rows


def packet(run):
    p,rows=generation(run);ev=run/'evaluation'
    known={}
    old_result=read_json(DIAG/'RESULT.json')
    for row in old_result['rows']:
        for cell in row['cells'].values():
            known[(row['id'],tuple(cell['keys']),cell['caption'])]=cell['review']
    records=[];mapping={};review=[]
    for row in rows:
        cells=read_json(ev/'records'/f'{row["id"]}.json')['cells'];ids={}
        for name,c in cells.items():
            ident=(row['id'],tuple(c['keys']),c['caption'])
            if ident not in ids:
                label=f'{row["id"][:3]}-{len(ids)+1}';ids[ident]=label
                records.append(dict(label=label,row_id=row['id'],split=row['split'],caption=c['caption'],caption_sha256=digest(c['caption']),keys=c['keys'],facts=row['facts']))
                mapping[label]=[]
                if ident in known:
                    previous=known[ident]
                    review.append(dict(previous,label=label,caption_sha256=digest(c['caption']),
                        review_provenance='Exact same source, keys and caption as prior provisional review; carried forward without treating it as new independent review.'))
            mapping[ids[ident]].append(dict(row_id=row['id'],cell=name))
    write_json(ev/'review_packet.json',dict(generation_freeze_sha256=sha256(ev/'freeze.json'),records=records,
        scope='Same assistant source and caption assessment, not independent or blinded; duplicate outputs share one judgment.'))
    write_json(ev/'review_mapping.json',mapping)
    if not (ev/'review.json').exists():write_json(ev/'review.json',dict(packet_sha256=sha256(ev/'review_packet.json'),records=review))
    write_json(ev/'pending_review.json',{'records':[r for r in records if r['label'] not in {x['label'] for x in review}]})
    print('PACKET',len(records),'carried',len(review),'pending',len(records)-len(review))


def score(run):
    p,rows=generation(run);ev=run/'evaluation';packet_obj=read_json(ev/'review_packet.json');review=read_json(ev/'review.json')
    if review['packet_sha256']!=sha256(ev/'review_packet.json') or packet_obj['generation_freeze_sha256']!=sha256(ev/'freeze.json'):raise ValueError('Review provenance changed')
    scores=validate_review(packet_obj,review);mapping=read_json(ev/'review_mapping.json')
    generated={(r['id'],name):cell for r in rows for name,cell in read_json(ev/'records'/f'{r["id"]}.json')['cells'].items()}
    validate_mapping(packet_obj,mapping,generated)
    mapped={(m['row_id'],m['cell']):scores[label] for label,ms in mapping.items() for m in ms}
    totals={s:{c:Counter() for c in ('base','previous_joint','caption_ce','caption_qa')} for s in ('validation','test','regression')}
    details=[]
    for row in rows:
        cells={}
        for name in totals[row['split']]:
            c=generated[(row['id'],name)];r=mapped[(row['id'],name)];t=totals[row['split']][name]
            t.update(windows=1,keys=4,fact_count=len(row['facts']),fact_hits=sum(r['fact_hits'].values()),
                event_count=sum(f['kind']=='event' for f in row['facts']),
                event_hits=sum(r['fact_hits'][f['id']] for f in row['facts'] if f['kind']=='event'),
                error_windows=int(bool(r['errors'])),error_claims=len(r['errors']),partial_facts=len(r.get('partial_fact_ids',[])),
                partial_events=sum(f['kind']=='event' and f['id'] in r.get('partial_fact_ids',[]) for f in row['facts']),
                caption_words=len(c['caption'].split()),tokens=c['generated_tokens'],caption_seconds=c['seconds'])
            cells[name]=dict(c,review=r)
        details.append(dict(id=row['id'],split=row['split'],facts=row['facts'],keys=row['keys'],cells=cells))
    gates={}
    for arm in ('caption_ce','caption_qa'):
        checks={}
        for split in ('test','regression'):
            for control in ('base','previous_joint'):
                a,b=totals[split][arm],totals[split][control]
                checks[f'{split}_vs_{control}']=a['fact_hits']>=b['fact_hits'] and a['event_hits']>=b['event_hits'] and a['error_windows']<=b['error_windows']
        gates[arm]=dict(passed=all(checks.values()),checks=checks)
    result=dict(status='COMPLETE_PROVISIONAL_AI_REVIEW',splits=totals,gates=gates,rows=details,
        adopted=False,selector_updated=False,joint_training_performed=False,training_arms={a:read_json(run/a/'RESULT.json') for a in gates},
        review_sha256=sha256(ev/'review.json'),generation_freeze_sha256=sha256(ev/'freeze.json'),
        limitations=['Six new test windows from only two purposively selected sources; no broad statistical claim.',
            'Four new validation windows and 16 legacy validation windows select checkpoints by caption CE, not test scores.',
            'The 14-window diagnostic was inspected previously and is development regression only.',
            'Same assistant source/output review; no independent human or GPT6-Astra semantic verification.',
            'Reference-assisted four-key caption experiment, not a deployed candidate/selector pipeline.',
            'Base/previous joint to new arms combines initialization, data and objective changes. Only new CE versus QA isolates auxiliary supervision at matched update count.',
            'No transmission, reconstruction or receiver Added/Missing/Distorted evaluation. Caption timings exclude encoding, model load and downstream processing.'])
    write_json(run/'RESULT.json',result)
    write_json(run/'verification.json',dict(status='PASS',condition_windows=len(rows)*4,review_records=len(scores),
        source_frames=sum(8 for _ in rows),result_sha256=sha256(run/'RESULT.json'),evaluator_sha256=sha256(Path(__file__)),
        checks=['frozen source frames and dependencies','disjoint original sources','exact four-key input matching','generation and checkpoint hashes','all facts reviewed','quoted errors tied to source frames']))
    render(run,result,rows)
    print({s:{c:dict(t) for c,t in v.items()} for s,v in totals.items()});print('GATES',gates)


def render(run,result,rows):
    lines=['<!doctype html><meta charset="utf-8"><title>Qwen caption repair</title>',
        '<style>body{font:16px system-ui;max-width:1500px;margin:24px auto;padding:16px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:8px;vertical-align:top}img{width:100%}.bad{color:#a22}details{margin:12px 0}h2{margin-top:32px}</style>',
        '<h1>고정 키 캡션 보완 실험</h1><p>같은 4장 · 같은 프롬프트 · 선택기 고정. 비독립 AI 검토, 기본 경로 미교체.</p>']
    for split,cells in result['splits'].items():
        lines.append(f'<h2>{split}</h2><table><tr><th>조건</th><th>사실 완전 재현</th><th>동작 재현</th><th>오류 구간</th><th>평균 단어</th></tr>')
        for name,t in cells.items():lines.append(f'<tr><td>{name}</td><td>{t["fact_hits"]}/{t["fact_count"]}</td><td>{t["event_hits"]}/{t["event_count"]}</td><td>{t["error_windows"]}/{t["windows"]}</td><td>{t["caption_words"]/t["windows"]:.1f}</td></tr>')
        lines.append('</table>')
    for detail,row in zip(result['rows'],rows):
        folder=DIAG if row['split']=='regression' else run
        sheet=os.path.relpath(folder/'clips'/f'{row["id"]}_candidates.jpg',run)
        dense=os.path.relpath(folder/'clips'/f'{row["id"]}_dense.jpg',run)
        lines.append(f'<h2>{row["id"]} · {row["split"]} · keys {row["keys"]}</h2><a href="{escape(dense)}">촘촘한 원본 표본</a><img src="{escape(sheet)}">')
        lines.append('<ol>'+''.join(f'<li>{escape(f["text"])}</li>' for f in row['facts'])+'</ol><table><tr><th>조건</th><th>실제 캡션</th><th>판정·근거</th></tr>')
        for name,c in detail['cells'].items():
            r=c['review']; errs=' '.join(f'{e["quote"]}: {e["reason"]}' for e in r['errors'])
            lines.append(f'<tr><td>{name}</td><td>{escape(c["caption"])}</td><td>{sum(r["fact_hits"].values())}/{len(row["facts"])} · {escape(r["rationale"])}<p class="bad">{escape(errs)}</p></td></tr>')
        lines.append('</table>')
    lines.append('<h2>해석 범위</h2><ul>'+''.join(f'<li>{escape(x)}</li>' for x in result['limitations'])+'</ul>')
    (run/'report.html').write_text('\n'.join(lines))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['packet','score']);parser.add_argument('--run',type=Path,default=RUN);args=parser.parse_args()
    (packet if args.command=='packet' else score)(args.run)
