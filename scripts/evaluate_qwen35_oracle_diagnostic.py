"""Build masked review packets and audit paired source-fact diagnostics."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
from html import escape
import json
import os
from pathlib import Path
import random

from semantic_transmission.qwen35_training_data import read_json, write_json, sha256
from semantic_transmission.qwen35_oracle_diagnostic import DEFAULT_RUN, coverage, oracle, verify


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def generation(run):
    p = verify(run)
    freeze = read_json(run/'generation_freeze.json')
    if freeze['protocol_sha256'] != sha256(run/'protocol.json'):
        raise ValueError('Generation used different protocol')
    for name, hashed in freeze['files'].items():
        if sha256(run/name) != hashed:
            raise ValueError('Generation changed')
    return p


def packet(run):
    p = generation(run)
    ensure_sheets(run, p)
    rng = random.Random(20261005)
    records, mapping = [], {}
    for row in p['rows']:
        cells = read_json(run/'records'/f'{row["id"]}.json')['cells']
        names = list(p['cells'])
        rng.shuffle(names)
        identities = {}
        for name in names:
            c = cells[name]
            identity = (tuple(c['keys']), c['caption'])
            if identity not in identities:
                label = f'{row["id"][:3]}-{len(identities)+1}'
                identities[identity] = label
                records.append({'label':label,'row_id':row['id'],'caption':c['caption'],
                    'caption_sha256':digest(c['caption']),'keys':c['keys'],
                    'facts':row['facts'],'source_review_note':row['source_review_note'],
                    'source_sheet':f'clips/{row["id"]}_candidates.jpg',
                    'dense_sheet':f'clips/{row["id"]}_dense.jpg'})
                mapping[label] = []
            mapping[identities[identity]].append({'row_id':row['id'],'cell':name})
    write_json(run/'review_packet.json',{'generation_freeze_sha256':sha256(run/'generation_freeze.json'),
        'scope':'Condition labels masked, but same Codex reviews source and output; not independent/blind ground truth.',
        'records':records})
    write_json(run/'review_mapping.json',mapping)
    print('REVIEW_PACKET',len(records),'distinct source/input/caption records')


def validate_review(packet_obj, review):
    records = {r['label']:r for r in packet_obj['records']}
    scores = {r['label']:r for r in review['records']}
    if len(scores) != len(review['records']) or set(scores) != set(records):
        raise ValueError('Missing or duplicated review rows')
    for label, r in scores.items():
        source=records[label]
        if r['caption_sha256'] != source['caption_sha256']:
            raise ValueError('Review caption changed')
        facts={f['id'] for f in source['facts']}
        if set(r['fact_hits']) != facts or any(type(v) is not bool for v in r['fact_hits'].values()):
            raise ValueError('Every reference fact needs a boolean judgment')
        partial = r.get('partial_fact_ids', [])
        if len(partial) != len(set(partial)) or not set(partial) <= facts or any(r['fact_hits'][f] for f in partial):
            raise ValueError('Partial matches must be unique known facts and cannot also be full hits')
        if not r.get('rationale'):
            raise ValueError('Missing rationale')
        for err in r['errors']:
            if not err['quote'] or err['quote'] not in source['caption'] or not err['reason'] or not err['source_indices']:
                raise ValueError('Error must quote the actual caption and cite source frames')
    return scores


def validate_mapping(packet_obj, mapping, generated):
    source={r['label']:r for r in packet_obj['records']}
    if set(mapping) != set(source):
        raise ValueError('Review mapping labels changed')
    assigned=set()
    for label,entries in mapping.items():
        p=source[label]
        for entry in entries:
            identity=(entry['row_id'],entry['cell'])
            if identity in assigned or entry['row_id']!=p['row_id'] or identity not in generated:
                raise ValueError('Duplicate or invalid condition mapping')
            c=generated[identity]
            if digest(c['caption'])!=p['caption_sha256'] or c['keys']!=p['keys']:
                raise ValueError('Review mapped to different caption or source keys')
            assigned.add(identity)
    if assigned!=set(generated):
        raise ValueError('Not all generated conditions have reviews')


def ensure_sheets(run, protocol):
    from PIL import Image, ImageDraw
    for row in protocol['rows']:
        folder=Path(row['frames'][0]['path']).parent
        metadata=read_json(folder.parent/'source_prepared.json')
        for count,width,cols,label in [(8,480,4,'candidates'),(24,240,6,'dense')]:
            dest=run/'clips'/f'{row["id"]}_{label}.jpg'
            if dest.exists():
                continue
            indices=row['indices'] if count==8 else [round(row['start']+(row['end']-row['start'])*i/(count-1)) for i in range(count)]
            height=round(width*320/576)
            sheet=Image.new('RGB',(cols*width,((count+cols-1)//cols)*(height+24)),'white')
            draw=ImageDraw.Draw(sheet)
            for j,index in enumerate(indices):
                path=folder/f'{index}.png'
                if sha256(path)!=metadata['source_frame_hashes'][str(index)]:
                    raise ValueError('Review source frame changed')
                with Image.open(path) as im:
                    x=j%cols*width;y=j//cols*(height+24)
                    sheet.paste(im.convert('RGB').resize((width,height)),(x,y+24))
                    draw.text((x+4,y+4),f'{row["id"][:3]} i{j} f{index} {index/24:.2f}s',fill='black')
            dest.parent.mkdir(parents=True,exist_ok=True)
            sheet.save(dest,quality=93)


def score(run):
    p = generation(run)
    packet_obj = read_json(run/'review_packet.json')
    review = read_json(run/'review.json')
    if review['packet_sha256'] != sha256(run/'review_packet.json') or packet_obj['generation_freeze_sha256'] != sha256(run/'generation_freeze.json'):
        raise ValueError('Review source changed')
    scores = validate_review(packet_obj,review)
    mapping=read_json(run/'review_mapping.json')
    generated={(row['id'],name):cell for row in p['rows'] for name,cell in
        read_json(run/'records'/f'{row["id"]}.json')['cells'].items()}
    validate_mapping(packet_obj,mapping,generated)
    mapped={}
    for label, entries in mapping.items():
        for entry in entries:
            mapped[(entry['row_id'],entry['cell'])]=scores[label]
    totals={c:Counter() for c in p['cells']}
    by_row=[]
    transitions=[]
    for row in p['rows']:
        cells=read_json(run/'records'/f'{row["id"]}.json')['cells']
        for name,c in cells.items():
            r=mapped[(row['id'],name)]
            counts=totals[name]
            counts['windows']+=1;counts['keys']+=len(c['keys'])
            counts['event_evidence_hits']+=c['coverage']['event_hits']
            counts['event_count']+=c['coverage']['event_count']
            counts['fact_count']+=len(row['facts'])
            counts['fact_hits']+=sum(r['fact_hits'].values())
            counts['event_caption_hits']+=sum(r['fact_hits'][f['id']] for f in row['facts'] if f['kind']=='event')
            counts['error_windows']+=bool(r['errors']); counts['error_claims']+=len(r['errors'])
            counts['ambiguous_claims']+=len(r.get('ambiguous',[]))
            counts['generated_tokens']+=c['generated_tokens'];counts['caption_seconds']+=c['seconds']
            counts['caption_words']+=len(c['caption'].split())
            counts['partial_fact_mentions']+=len(r.get('partial_fact_ids',[]))
            counts['partial_event_mentions']+=sum(f['kind']=='event' and f['id'] in r.get('partial_fact_ids',[]) for f in row['facts'])
            for f in row['facts']:
                available=c['coverage']['facts'][f['id']]
                hit=r['fact_hits'][f['id']]
                counts['fact_evidence_hits']+=available
                counts['missing_despite_evidence']+=available and not hit
                counts['missing_without_evidence']+=not available and not hit
                counts['matches_without_rubric_evidence']+=not available and hit
            c['review']=r
        by_row.append({'id':row['id'],'video_id':row['video_id'],'facts':row['facts'],'cells':cells,
            'oracle':row['oracle'],'unrestricted_oracle':row['unrestricted_oracle']})
        for captioner in ['base','joint']:
            a,b=cells[f'learned_{captioner}'],cells[f'oracle_{captioner}']
            for fact in row['facts']:
                fid=fact['id']
                transitions.append({'row_id':row['id'],'captioner':captioner,'fact_id':fid,'kind':fact['kind'],
                    'text':fact['text'],'learned_evidence':a['coverage']['facts'][fid],
                    'oracle_evidence':b['coverage']['facts'][fid],
                    'learned_caption':a['review']['fact_hits'][fid],'oracle_caption':b['review']['fact_hits'][fid]})
    result={'status':'DIAGNOSTIC_COMPLETE_PROVISIONAL_AI_REVIEW','sources':len(p['sources']),'windows':len(p['rows']),
        'training_performed':False,'adopted':False,'conditions':{k:dict(v) for k,v in totals.items()},
        'oracle_event_ceiling':sum(r['oracle']['coverage']['event_hits'] for r in p['rows']),
        'unrestricted_four_key_event_ceiling':sum(r['unrestricted_oracle']['coverage']['event_hits'] for r in p['rows']),
        'eight_candidate_event_ceiling':sum(coverage(r,range(8))['event_hits'] for r in p['rows']),
        'same_learned_oracle_key_windows':sum(r['cells']['learned_joint']['keys']==r['cells']['oracle_joint']['keys'] for r in by_row),
        'max_allocated_gib':max(read_json(run/'records'/f'{r["id"]}.json')['max_allocated_gib'] for r in p['rows']),
        'generation_freeze_sha256':sha256(run/'generation_freeze.json'),'review_sha256':sha256(run/'review.json'),
        'limitations':['Purposive 8-source diagnostic, not random benchmark. New to pilot, not globally unseen sources.',
            'Same Codex source/output AI review; no independent human or GPT6-Astra verification.',
            'Oracle uses references and is not a deployable selector; only one evidence-optimal tie-broken subset generated.',
            'Some reference items bundle attributes/direction. Full-item recall is strict; partial mentions are supplementary, not fully correct facts.',
            'Observed residuals do not by themselves distinguish encoder recognition, temporal reasoning, and caption omission.',
            'No transmission or reconstructed-video Added/Missing/Distorted evaluation. Timing excludes loading, shared encoding, transmission and reconstruction.'],
        'rows':by_row,'paired_fact_transitions':transitions}
    result['evidence_sensitivity'] = sensitivity(run, p, by_row)
    write_json(run/'RESULT.json',result)
    write_json(run/'verification.json',{'status':'PASS','protocol_sha256':sha256(run/'protocol.json'),
        'reference_frozen_before_generation':True,'source_frames':8*len(p['rows']),
        'review_records':len(scores),'condition_windows':4*len(p['rows']),
        'implementation_sha256':sha256(Path(__file__)),'RESULT_sha256':sha256(run/'RESULT.json'),
        'checks':['source original and normalized hashes','frozen implementation and checkpoint files','reference hash',
                  'generation hashes','complete review and exact error quotes','same-key adapter input matching']})
    render(run,result,p)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','paired_fact_transitions')},ensure_ascii=False,indent=2))


def sensitivity(run, protocol, rows):
    """Post-hoc source-evidence audit, preserving original references and outputs."""
    patches = [
        ('d04','f2',[[0,1,2,3],[4,5,6]],
         'Absence at i7 proves departure but not its direction. A visible intermediate rightward movement is needed for the full directional fact.'),
        ('d04','f3',[[4,5,6],[7]],
         'i6 also shows the man at the right edge; it is a valid alternative before the final empty state.'),
        ('d05','f4',[[3,4,5,6],[7]],
         'Attached wheel at i3/i4 followed by the man visibly walking away holding the detached wheel at i7 is sufficient. Requiring an extra isolated detached-wheel frame was too narrow.'),
        ('d10','f2',[[0,1,2],[4,5,6,7]],
         'The near side view at i6/i7 is also valid later evidence of approaching from the background.'),
        ('d11','f3',[[0,1,2,3,4],[6,7]],
         'Outside at i0/i1/i2 followed by visibly inside at i6/i7 establishes entry without requiring a separate open-door outside frame.'),
    ]
    detail=[]; revised={}; original={r['id']:r for r in protocol['rows']}
    for row in rows:
        ref=deepcopy(original[row['id']])
        for prefix,fid,groups,reason in patches:
            if ref['id'].startswith(prefix+'_'):
                fact=next(f for f in ref['facts'] if f['id']==fid)
                detail.append({'row_id':ref['id'],'fact_id':fid,'before':fact['evidence_groups'],
                    'after':groups,'reason':reason,'source_frames':ref['frames']})
                fact['evidence_groups']=groups
        revised[row['id']]=ref
    counts={}
    for selector in ['learned','oracle']:
        counts[selector]=sum(coverage(revised[r['id']],r['cells'][selector+'_joint']['keys'])['event_hits'] for r in rows)
    result={'scope':'Post-hoc alternative evidence audit only. No captions, original labels, keys, checkpoints or learning changed.',
        'original_learned_event_evidence':sum(r['cells']['learned_joint']['coverage']['event_hits'] for r in rows),
        'original_oracle_event_evidence':sum(r['cells']['oracle_joint']['coverage']['event_hits'] for r in rows),
        'audited_learned_event_evidence':counts['learned'],'audited_original_oracle_event_evidence':counts['oracle'],
        'audited_optimal_four_key_ceiling':sum(oracle(r)['coverage']['event_hits'] for r in revised.values()),
        'event_count':sum(coverage(r,range(8))['event_count'] for r in revised.values()),
        'changes':detail,
        'conclusion':'Do not claim an inherent 29/30 four-key limit from the original strict rubric. Alternative valid state evidence changes the ceiling; learned selection still misses evidence, and unchanged oracle captions still omit events.'}
    write_json(run/'evidence_sensitivity.json',result)
    return {k:v for k,v in result.items() if k!='changes'}


def render(run,result,protocol):
    ensure_sheets(run, protocol)
    h=['<!doctype html><meta charset="utf-8"><title>Qwen failure diagnostic</title>',
       '<style>body{font:16px system-ui;max-width:1500px;margin:auto;padding:20px;background:#f5f5f5}img{max-width:100%}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #ccc;padding:10px;vertical-align:top}section{margin:30px 0}summary{cursor:pointer} .cards{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}.card{background:white;padding:16px}.miss{color:#b32d22}.hit{color:#166534}button{padding:8px;margin:5px}a{color:#175ea8}</style>',
       '<h1>Qwen 선택·캡션 실패 원인 진단</h1><p>8원본 · 14구간 · 네 조건 · 추가 학습 없음. 정답 프레임은 진단용이며 배포 가능한 선택기가 아닙니다. 비독립 Codex AI 검토입니다.</p>',
       '<p><a href="RESULT.json">전체 수치와 판정</a> · <a href="protocol.json">동결 조건·원본 해시</a> · <a href="review.json">원문 인용·검수 이유</a> · <a href="verification.json">검증</a></p>',
       '<table><tr><th>조건</th><th>사건 근거</th><th>캡션 사실</th><th>캡션 동작·변화</th><th>오류 구간</th></tr>']
    for name,c in result['conditions'].items():
        h.append(f'<tr><td>{name}</td><td>{c["event_evidence_hits"]}/{c["event_count"]}</td><td>{c["fact_hits"]}/{c["fact_count"]}</td><td>{c["event_caption_hits"]}/{c["event_count"]}</td><td>{c["error_windows"]}/{c["windows"]}</td></tr>')
    h.append('</table><p>사건 근거 보존과 캡션에 실제로 적힌 사건을 구분합니다. 이전 평가와 원본·주석 정의가 달라 이전 분모와 직접 비교하지 않습니다.</p>')
    s=result['evidence_sensitivity']
    h.append(f'<p>근거 조건 재검토: 유효한 최종 상태를 추가하고 방향 근거를 강화하면 동일 키의 사건 근거는 학습 {s["audited_learned_event_evidence"]}/30, 정답 {s["audited_original_oracle_event_evidence"]}/30입니다. 원래 주석과 출력은 보존했습니다. <a href="evidence_sensitivity.json">항목별 근거 감사</a></p>')
    h.append('<p>정답 항목 일부에는 속성·방향 조건이 포함됩니다. 일부만 언급한 문장은 완전 재현에 포함하지 않으며 별도 표시합니다. 이는 원자적 사실 전체의 객관적 정확도 점수가 아닙니다.</p>')
    for row,source in zip(result['rows'],protocol['rows']):
        h.append(f'<section id="{row["id"]}"><h2>{row["id"]} · {source["start"]/24:.2f}–{source["end"]/24:.2f}s</h2><img src="clips/{row["id"]}_candidates.jpg"><details><summary>24장 원본 검수</summary><img src="clips/{row["id"]}_dense.jpg"></details><div class="cards">')
        for name in protocol['cells']:
            c=row['cells'][name];r=c['review']
            h.append(f'<article class="card"><h3>{name} · 키 {c["keys"]}</h3><p>{escape(c["caption"])}</p><ul>')
            for fact in row['facts']:
                fid=fact['id'];hit=r['fact_hits'][fid];ev=c['coverage']['facts'][fid]
                mark='✓' if hit else ('부분' if fid in r.get('partial_fact_ids',[]) else '✗')
                h.append(f'<li class="{"hit" if hit else "miss"}">{mark} {escape(fact["text"])} (선택 근거 {"충족" if ev else "부족"})</li>')
            h.append('</ul><p>'+escape(r['rationale'])+'</p>')
            for e in r['errors']:
                h.append(f'<p class="miss">오류: “{escape(e["quote"])}” — {escape(e["reason"])}</p>')
            for a in r.get('ambiguous',[]):h.append('<p>보류: '+escape(a)+'</p>')
            h.append('</article>')
        h.append('</div></section>')
    (run/'report.html').write_text('\n'.join(h))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['packet','score'])
    parser.add_argument('--run',type=Path,default=DEFAULT_RUN)
    args=parser.parse_args()
    (packet if args.command=='packet' else score)(args.run)
