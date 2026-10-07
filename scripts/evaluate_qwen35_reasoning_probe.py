"""Blinded same-image diagnostic; observational notes are not hidden reasoning."""
import argparse
from collections import Counter
from html import escape
import json
import random
from semantic_transmission.qwen35_training_data import read_json,write_json,sha256
from semantic_transmission.qwen35_reasoning_probe import RUN,PARENT,parse_observations
from evaluate_qwen35_astra_transfer import digest,validate_source_review
from evaluate_qwen35_oracle_diagnostic import validate_mapping


def verify():
    freeze=read_json(RUN/'freeze.json')
    if sha256(RUN/'protocol.json')!=freeze['protocol_sha256']:raise ValueError('Changed protocol')
    for p,h in freeze['files'].items():
        if sha256(RUN/p)!=h:raise ValueError('Changed generation')
    ref=read_json(PARENT/'reference_freeze.json')
    if sha256(PARENT/'reference.json')!=ref['reference_sha256']:raise ValueError('Changed reference')
    return {r['id']:r for r in read_json(PARENT/'reference.json')['records']}


def normalized_notes(text,keys):
    """Secondary format audit: i2 and integer2 refer to the same source label.

    Does not replace generation, strict parser results, or semantic judgments.
    No index remapping, truncation repair, or unsupported-frame substitution.
    """
    try:
        clean=text.strip()
        if clean.startswith('```'):clean='\n'.join(clean.splitlines()[1:-1])
        obj=json.loads(clean)
        def index(x):
            return int(x[1:]) if isinstance(x,str) and x.startswith('i') and x[1:].isdigit() else x
        for o in obj['observations']:o['evidence']=[index(x) for x in o['evidence']]
        for c in obj['changes']:
            for k in ['before','after']:c[k]=[index(x) for x in c[k]]
        return parse_observations(json.dumps(obj),keys)
    except (ValueError,KeyError,TypeError) as exc:return None,str(exc)


def packet():
    # Release a complete author partition while the other partition generates.
    # Each packet freezes its own exact generation files, then full score()
    # additionally verifies the completed global generation freeze.
    ref_freeze=read_json(PARENT/'reference_freeze.json')
    if sha256(PARENT/'reference.json')!=ref_freeze['reference_sha256']:raise ValueError('Changed reference')
    refs={r['id']:r for r in read_json(PARENT/'reference.json')['records']}
    sheets={};mapping={};packets={'A':[],'B':[]};rng=random.Random(20261006);generation_files={'A':{},'B':{}}
    for author in ['A','B']:
        for r in read_json(PARENT/'teacher'/f'packet8_{author}.json')['records']:sheets[r['id']]=r
    for ordinal,(rid,ref) in enumerate(refs.items()):
        names=['analysis','direct','structured'];rng.shuffle(names)
        path=RUN/'records'/f'{rid}.json'
        if not path.exists():continue
        row=read_json(path);author='B' if ref['split']=='validation' else 'A';generation_files[author][str(path.relative_to(RUN))]=sha256(path)
        for slot,name in enumerate(names):
            c=row['cells'][name];label=f'p{ordinal*3+slot:03d}'
            packets[author].append(dict(label=label,row_id=rid,
                caption=c['caption'],caption_sha256=digest(c['caption']),keys=c['keys'],facts=ref['facts'],
                observable_four_fact_ids=ref['four_frame_fact_ids'],source_sheet=sheets[rid]['sheet'],
                source_frames=sheets[rid]['frames'],generation_status=c['status']))
            mapping[label]=[dict(row_id=rid,cell=name)]
    for author,rows in packets.items():
        if len(rows)!=36:continue
        obj=dict(author=author,records=rows,generation_files=generation_files[author],generation_protocol_sha256=sha256(RUN/'protocol.json'),reference_freeze_sha256=sha256(PARENT/'reference_freeze.json'),
            scope='Known development sources; method names hidden, prose versus JSON format remains visible. Judge factual content of affirmative notes and captions, not uncertain hypotheses as assertions. Same-family AI review.')
        path=RUN/f'packet_{author}.json'
        if path.exists() and read_json(path)!=obj:raise ValueError('Changed packet')
        write_json(path,obj)
    if len(mapping)==72:
        write_json(RUN/'mapping_private.json',mapping)
        write_json(RUN/'all_packet.json',dict(records=packets['A']+packets['B']))
    print('PACKETS',{a:len(r) for a,r in packets.items()})


def score():
    refs=verify();reviews=[];files={}
    for a in ['A','B']:
        p=RUN/f'packet_{a}.json';review_path=RUN/f'review_{a}.json';d=read_json(review_path)
        if d['packet_sha256']!=sha256(p) or d['annotator_model']!='gpt-6-astra' or d['author']!=a:raise ValueError('Bad judge provenance')
        packet_obj=read_json(p)
        if packet_obj['generation_protocol_sha256']!=sha256(RUN/'protocol.json'):raise ValueError('Different generation protocol')
        for rel,h in packet_obj['generation_files'].items():
            if sha256(RUN/rel)!=h:raise ValueError('Changed partition output')
        if not {r['source_sheet'] for r in packet_obj['records']}<=set(d['viewed_sheet_paths']):raise ValueError('Missing source inspection')
        validate_source_review(packet_obj,d);reviews+=d['records'];files[p.name]=sha256(p);files[review_path.name]=sha256(review_path)
    pack=read_json(RUN/'all_packet.json');mapping=read_json(RUN/'mapping_private.json');scores=validate_source_review(pack,dict(records=reviews))
    generated={(rid,c):v for rid in refs for c,v in read_json(RUN/'records'/f'{rid}.json')['cells'].items()}
    validate_mapping(pack,mapping,generated)
    linked={(e['row_id'],e['cell']):scores[label] for label,entries in mapping.items() for e in entries}
    totals={n:Counter() for n in ['direct','analysis','structured']};flow=Counter();details=[];runtime=Counter()
    previous={r['id']:r for r in read_json(PARENT/'RESULT.json')['rows']};regrade=[];identical=0
    for rid,ref in refs.items():
        row=read_json(RUN/'records'/f'{rid}.json');eligible=set(ref['four_frame_fact_ids']);actions={f['id'] for f in ref['facts'] if f['kind']!='state'};hits={};cells={}
        for name,out in row['cells'].items():
            judge=linked[rid,name];hit={f for f,ok in judge['fact_hits'].items() if ok} if out['status']=='COMPLETE' else set();hits[name]=hit&eligible
            totals[name].update(windows=1,observable_facts=len(eligible),fact_hits=len(hit&eligible),observable_actions=len(eligible&actions),action_hits=len(hit&eligible&actions),
                full_or_partial_hits=len((hit|set(judge.get('partial_fact_ids',[])))&eligible),error_windows=bool(judge['errors']),error_claims=len(judge['errors']),
                failures=out['status']!='COMPLETE',generation_seconds=out['seconds'],tokens=out['generated_tokens'],words=len(out['caption'].split()))
            cells[name]=dict(output=out,review=judge)
        normalized,error=normalized_notes(row['cells']['analysis']['caption'],row['cells']['analysis']['keys'])
        runtime.update(strict_json_valid=row['cells']['analysis']['parse_error'] is None,label_normalized_json_valid=error is None,
            shared_encode_seconds=row['encode_seconds'],structured_total_seconds=row['structured_total_seconds'])
        flow.update(observable_facts=len(eligible),notes_full=len(hits['analysis']),notes_full_retained=len(hits['analysis']&hits['structured']),
            notes_full_omitted=len(hits['analysis']-hits['structured']),caption_full_without_notes_full=len(hits['structured']-hits['analysis']),
            neither_full=len(eligible-hits['analysis']-hits['structured']),structured_gain_vs_direct=len(hits['structured']-hits['direct']),
            structured_loss_vs_direct=len(hits['direct']-hits['structured']))
        partial=set(cells['analysis']['review'].get('partial_fact_ids',[]))&eligible
        flow.update(notes_partial=len(partial),notes_unmentioned=len(eligible-hits['analysis']-partial))
        old=previous[rid]['cells']['teacher_r32'];new=cells['direct']
        identical+=int(old['output']['caption']==new['output']['caption'])
        changed=[f for f in eligible if old['review']['fact_hits'][f]!=new['review']['fact_hits'][f]]
        error_change=bool(old['review']['errors'])!=bool(new['review']['errors'])
        if changed or error_change:regrade.append(dict(id=rid,changed_full_fact_ids=changed,error_presence_changed=error_change,
            previous_review=old['review'],current_review=new['review']))
        details.append(dict(id=rid,source_sheet=next(r['source_sheet'] for r in pack['records'] if r['row_id']==rid),facts=ref['facts'],cells=cells,
            notes_full_omitted=sorted(hits['analysis']-hits['structured']),neither_full=sorted(eligible-hits['analysis']-hits['structured']),normalized_parse_error=error))
    result=dict(status='COMPLETE_DEVELOPMENT_ASTRA_REVIEW',totals=totals,fact_flow=flow,runtime=runtime,rows=details,training_started=False,default_pipeline_changed=False,
        baseline_regrade=dict(identical_caption_count=identical,previous_totals=dict(fact_hits=38,action_hits=24,error_windows=3),changes=regrade,
            interpretation='Same24 captions re-reviewed with paired new outputs. Differences are AI judgment variability, not model or weight changes. Do not interpret changed baseline scores as regression.'),
        limitations=['Known24 development windows from8 originals; no new sealed test use.',
            'Notes are observable generated text, not a measurement of internal model reasoning or causal perception ability.',
            'Structured path uses a larger inference token budget and2 calls, direct path1 call; model and4 images are fixed.',
            'Same-family Astra judges with source access; not human or independent model-family truth.',
            'Strict JSON failures retained unchanged. Secondary audit normalizes i-prefixed index strings only.'])
    write_json(RUN/'review_freeze.json',dict(files=files,generation_freeze_sha256=sha256(RUN/'freeze.json')))
    write_json(RUN/'RESULT.json',result)
    html=['<!doctype html><meta charset="utf-8"><title>Observation to caption diagnostic</title><style>body{font:16px system-ui;max-width:1400px;margin:auto;padding:20px}img{max-width:100%}td,th{padding:8px;border:1px solid #bbb}table{border-collapse:collapse;width:100%}pre{white-space:pre-wrap}</style>',
        '<h1>고정4키 관찰문 → 캡션 진단</h1><p>기존 개발24구간. 가중치 고정, 추가 학습 없음. JSON 형식 때문에 조건 완전 맹검은 아님.</p>',f'<pre>{escape(json.dumps(dict(totals=totals,fact_flow=flow,runtime=runtime),indent=2))}</pre>']
    for row in details:
        sheet='../qwen35_astra_transfer_v1/teacher/sheets8/'+row['id']+'.png'
        html += [f'<details><summary>{row["id"]}</summary><img src="{sheet}"><table><tr><th>조건</th><th>실제 출력</th><th>판정</th></tr>']
        for n,c in row['cells'].items():html.append(f'<tr><td>{n}</td><td><pre>{escape(c["output"]["caption"])}</pre></td><td>{escape(c["review"]["rationale"])}</td></tr>')
        html.append('</table></details>')
    (RUN/'report.html').write_text('\n'.join(html));print(json.dumps(dict(totals=totals,fact_flow=flow,runtime=runtime),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['packet','score']);a=p.parse_args();{'packet':packet,'score':score}[a.command]()
