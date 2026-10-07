"""Validate explicit independent reviews and score the frozen caption holdout."""
import argparse
from collections import Counter
from html import escape
import os
import random
from semantic_transmission.qwen35_caption_holdout import RUN,OLD,ARMS,save_frozen,verify_inputs,validate_record
from semantic_transmission.qwen35_training_data import read_json,write_json,sha256
from evaluate_qwen35_astra_transfer import digest
from evaluate_qwen35_grounded import validate_grounded_review
from evaluate_qwen35_oracle_diagnostic import validate_mapping
from evaluate_qwen35_evidence_repair import gate


def validate_author(packet_path, doc):
    if doc['packet_sha256']!=sha256(packet_path) or not doc['source_only']:
        raise ValueError('Wrong packet or outputs seen during source reference')
    a=doc['annotator']
    if a['kind'] not in ['agent','human'] or not a['identity'] or a['prior_outputs_seen']:
        raise ValueError('Missing independent reviewer identity')


def validate_source_labels(packet,doc,audit=False):
    rows={r['id']:r for r in packet['records']}; labels={r['id']:r for r in doc['records']}
    if len(labels)!=len(doc['records']) or set(labels)!=set(rows): raise ValueError('Missing/duplicate source label')
    if not {r['sheet'] for r in rows.values()}<=set(doc['viewed_sheet_paths']): raise ValueError('Uninspected source sheet')
    for rid,row in rows.items():
        r=labels[rid]; allowed={f['index'] for f in row['frames']}
        def evidence(x):
            if not x['reason'] or not x['source_indices'] or not set(x['source_indices'])<=allowed: raise ValueError('Missing/unseen source evidence')
        if audit:
            check=r['caption_review'];evidence(check)
            if check['status'] not in ['supported','partial','unsupported'] or not r['corrected_caption']: raise ValueError('Missing caption audit')
            for name in ['facts','evidence_queries','claims','preference_pairs']:
                expected={x['id'] for x in row[name]} if name=='facts' else set(range(len(row[name])))
                checks=r[name]; key='id' if name=='facts' else 'index'
                if len(checks)!=len(expected) or {x[key] for x in checks}!=expected: raise ValueError('Incomplete label audit')
                for x in checks:
                    evidence(x)
                    if x['status'] not in ['supported','partial','unsupported']: raise ValueError('Invalid audit status')
        else:
            if not 3<=len(r['facts'])<=6 or len({f['id'] for f in r['facts']})!=len(r['facts']): raise ValueError('Need3-6 distinct facts')
            for fact in r['facts']:
                if not fact['text'] or not fact['full_credit'] or fact['kind'] not in ['state','action','change']: raise ValueError('Invalid atomic fact')
                evidence(fact)
                if fact['kind']!='state' and len(set(fact['source_indices']))<2: raise ValueError('Motion/change needs more than one observation')
            if not r['uncertainties'] or not r['camera_review']: raise ValueError('Explicit uncertainty/camera review required (use none if appropriate)')
    return labels


def audit_summary():
    result={}
    for name in ['train','development']:
        pp=RUN/f'source_packet_{name}.json';path=RUN/f'source_review_{name}.json';doc=read_json(path)
        validate_author(pp,doc);labels=validate_source_labels(read_json(pp),doc,True)
        counts={field:Counter() for field in ['caption','facts','evidence_queries','claims','preference_pairs']}
        findings=[]
        for rid,r in labels.items():
            items={'caption':[r['caption_review']],**{k:r[k] for k in counts if k!='caption'}}
            for field,checks in items.items():
                for item in checks:
                    counts[field][item['status']]+=1
                    if item['status']!='supported':findings.append(dict(row_id=rid,field=field,**item))
        result[name]=dict(windows=len(labels),counts=counts,flagged=findings,annotator=doc['annotator'],review_sha256=sha256(path))
    obj=dict(status='COMPLETE_SEPARATE_SOURCE_AUDIT',splits=result,training_scope='17 newly repaired windows; not all120 inherited training windows.',interpretation='Statuses judge supplied annotation correctness, not truth of rejected/contradicted text. Partial is uncertain or partly overstated, not automatically a fully wrong label.',training_data_changed=False,training_updates=0)
    write_json(RUN/'source_audit_summary.json',obj)
    return obj


def validate_cross_audit(packet,reference,doc):
    source={r['id']:r for r in packet['records']};refs={r['id']:r for r in reference['records']}
    rows={r['id']:r for r in doc['records']}
    if len(rows)!=len(doc['records']) or set(rows)!=set(source) or set(refs)!=set(source): raise ValueError('Incomplete cross-audit')
    if doc['annotator']['identity']==reference['annotator']['identity'] or doc['annotator']['prior_outputs_seen'] or not doc['source_only']: raise ValueError('Cross-auditor must be a distinct source-only context')
    if not {r['sheet'] for r in source.values()}<=set(doc['viewed_sheet_paths']): raise ValueError('Cross-auditor did not inspect every source')
    supported={}
    for rid,row in rows.items():
        checks={x['id']:x for x in row['facts']};expected={f['id'] for f in refs[rid]['facts']};allowed={f['index'] for f in source[rid]['frames']}
        if len(checks)!=len(row['facts']) or set(checks)!=expected: raise ValueError('Incomplete fact cross-audit')
        for x in checks.values():
            if x['status'] not in ['supported','partial','unsupported'] or not x['reason'] or not x['source_indices'] or not set(x['source_indices'])<=allowed: raise ValueError('Invalid cross-audit evidence')
        supported[rid]=dict(refs[rid],facts=[f for f in refs[rid]['facts'] if checks[f['id']]['status']=='supported'])
        if not supported[rid]['facts']: raise ValueError('No supported reference facts in window')
    return supported


def eligible_references():
    packet=read_json(RUN/'source_packet_fresh.json');reference=read_json(RUN/'source_review_fresh.json');doc=read_json(RUN/'source_reference_cross_audit.json')
    if doc['source_packet_sha256']!=sha256(RUN/'source_packet_fresh.json') or doc['reference_sha256']!=sha256(RUN/'source_review_fresh.json'): raise ValueError('Cross-audit belongs to different source/reference')
    return validate_cross_audit(packet,reference,doc)


def freeze_reference():
    verify_inputs(); files={};summary={}
    for name in ['fresh','train','development']:
        pp=RUN/f'source_packet_{name}.json';lp=RUN/f'source_review_{name}.json'
        doc=read_json(lp);validate_author(pp,doc);labels=validate_source_labels(read_json(pp),doc,name!='fresh')
        files[lp.name]=sha256(lp);files[pp.name]=sha256(pp)
        summary[name]=dict(rows=len(labels),annotator=doc['annotator'])
    eligible=eligible_references()
    files['source_reference_cross_audit.json']=sha256(RUN/'source_reference_cross_audit.json')
    for name in ['review_provenance.json','source_review_development.initial.json','source_review_development_clarification.json']:
        if (RUN/name).exists(): files[name]=sha256(RUN/name)
    summary['fresh_supported_facts']=sum(len(r['facts']) for r in eligible.values())
    # All labels are finalized before any fresh model output is generated.
    save_frozen(RUN/'reference_freeze.json',dict(files=files,summary=summary,input_freeze_sha256=sha256(RUN/'input_freeze.json'),policy='Separate reviewer contexts, same-family AI if agents; no claim of human truth. Corrections are separate; original training/reference files preserved.'))
    audit_summary()
    print('SOURCE_REVIEWS_FROZEN',summary)


def generation():
    p=verify_inputs(); f=read_json(RUN/'generation_freeze.json'); ph=sha256(RUN/'protocol.json')
    if f['protocol_sha256']!=ph or f['reference_freeze_sha256']!=sha256(RUN/'reference_freeze.json'): raise ValueError('Changed generation protocol/reference')
    expected={f'records/{r["id"]}.json' for r in p['rows']}
    if set(f['files'])!=expected: raise ValueError('Incomplete generation freeze')
    for name,h in read_json(RUN/'reference_freeze.json')['files'].items():
        if sha256(RUN/name)!=h: raise ValueError('Changed source review')
    result={}
    for row in p['rows']:
        name=f'records/{row["id"]}.json'
        if sha256(RUN/name)!=f['files'][name]: raise ValueError('Changed generated caption')
        record=read_json(RUN/name);validate_record(row,record,ph)
        result.update({(row['id'],a):c for a,c in record['cells'].items()})
    return p,result


def packet():
    p,outputs=generation();refs=eligible_references()
    sources={r['id']:r for r in read_json(RUN/'source_packet_fresh.json')['records']}
    rng=random.Random(20261008);records=[];mapping={}
    for row in p['rows']:
        rid=row['id'];order=ARMS[:];rng.shuffle(order);dedup={}
        for arm in order:
            c=outputs[rid,arm];caption=c['caption']
            if caption not in dedup:
                label=f'c{len(records):03d}';dedup[caption]=label;mapping[label]=[]
                records.append(dict(label=label,row_id=rid,caption=caption,caption_sha256=digest(caption),keys=row['keys'],facts=refs[rid]['facts'],source_sheet=sources[rid]['sheet'],source_frames=sources[rid]['frames']))
            mapping[dedup[caption]].append(dict(row_id=rid,cell=arm))
    save_frozen(RUN/'caption_packet.json',dict(records=records,generation_freeze_sha256=sha256(RUN/'generation_freeze.json'),reference_freeze_sha256=sha256(RUN/'reference_freeze.json'),scope='Judge must be a different context from fresh reference author; condition identities withheld.'))
    save_frozen(RUN/'mapping_private.json',mapping);print('CAPTION_PACKET',len(records))


def score():
    p,outputs=generation();pack=read_json(RUN/'caption_packet.json');doc=read_json(RUN/'caption_review.json')
    if doc['packet_sha256']!=sha256(RUN/'caption_packet.json') or pack['generation_freeze_sha256']!=sha256(RUN/'generation_freeze.json') or pack['reference_freeze_sha256']!=sha256(RUN/'reference_freeze.json'): raise ValueError('Changed review inputs')
    if not doc['annotator']['condition_labels_hidden'] or doc['annotator']['identity']==read_json(RUN/'source_review_fresh.json')['annotator']['identity']: raise ValueError('Judge must be separate and masked')
    scores=validate_grounded_review(pack,doc);mapping=read_json(RUN/'mapping_private.json');validate_mapping(pack,mapping,outputs)
    linked={(x['row_id'],x['cell']):scores[label] for label,rr in mapping.items() for x in rr}
    refs=eligible_references();totals={a:Counter() for a in ARMS};sources={};details=[]
    for row in p['rows']:
        rid=row['id'];facts=refs[rid]['facts'];actions={f['id'] for f in facts if f['kind']!='state'}
        sources.setdefault(row['source_id'],{a:Counter() for a in ARMS});cells={}
        for arm in ARMS:
            out=outputs[rid,arm];r=linked[rid,arm];success=out['status']=='COMPLETE';hits={f for f,v in r['fact_hits'].items() if v} if success else set()
            partial=set(r['partial_fact_ids']) if success else set()
            counts=dict(windows=1,facts=len(facts),fact_hits=len(hits),actions=len(actions),action_hits=len(hits&actions),error_windows=int(bool(r['errors'])),error_claims=len(r['errors']),generation_failures=int(not success),full_or_partial_hits=len(hits|partial),action_full_or_partial_hits=len((hits|partial)&actions),generation_seconds=out['seconds'],caption_words=len(out['caption'].split()))
            totals[arm].update(counts);sources[row['source_id']][arm].update(counts);cells[arm]=dict(output=out,review=r,counts=counts)
        details.append(dict(id=rid,source_id=row['source_id'],facts=facts,cells=cells,source_sheet=str(RUN/'sheets'/f'{rid}.png')))
    check=gate(totals['caption'],totals['initial'])
    result=dict(status='COMPLETE_SEPARATE_REVIEWER_HOLDOUT',totals=totals,per_source=sources,gate=dict(check,passed=all(check.values())),source_originals=len(sources),windows=len(details),training_updates=0,default_pipeline_changed=False,
        uncertainty=bootstrap(sources),source_audit=audit_summary(),limitations=[p['scope'],p['limitations'],'Small sample; separate AI contexts are not independent human truth. No learned-key selection, transmission or reconstruction comparison.'],
        reference_policy='Only fully supported facts in separate source cross-audit, finalized before any new model captions.',
        original_facts=sum(len(r['facts']) for r in read_json(RUN/'source_review_fresh.json')['records']),
        scoring_implementation_sha256=sha256(__file__),
        provenance={n:sha256(RUN/n) for n in ['input_freeze.json','reference_freeze.json','generation_freeze.json','caption_packet.json','caption_review.json','mapping_private.json','source_audit_summary.json']})
    write_json(RUN/'paired_details.json',dict(records=details));write_json(RUN/'RESULT.json',result);render(result,details);print(dict(totals))


def bootstrap(sources):
    # Resample originals, not correlated windows/facts, for descriptive uncertainty.
    import numpy as np
    rr=list(sources.values());rng=np.random.default_rng(20261007);result={}
    for metric,denom in [('fact_hits','facts'),('action_hits','actions'),('error_windows','windows')]:
        deltas=[]
        for _ in range(5000):
            sample=[rr[i] for i in rng.integers(0,len(rr),len(rr))];den=sum(x['initial'][denom] for x in sample)
            if den:deltas.append(100*sum(x['caption'][metric]-x['initial'][metric] for x in sample)/den)
        result[metric]=dict(delta_percentage_point_interval_95=np.percentile(deltas,[2.5,97.5]).tolist() if deltas else None,resampling_unit='source original',replicates=5000,scope='Descriptive paired bootstrap; not a superiority proof.')
    return result


def render(result,details):
    html=['<!doctype html><meta charset="utf-8"><title>Fresh caption holdout</title><style>body{font:16px sans-serif;max-width:1200px;margin:auto;padding:24px}img{max-width:100%}td,th{padding:10px;border:1px solid #bbb}table{border-collapse:collapse;width:100%}summary{padding:12px}</style><h1>미사용 ClipShots 원본 캡션 비교</h1><p>같은4키, 초기 모델과 caption epoch2. 독립된 평가 문맥의 AI 검수이며 사람 정답은 아님. 키 선택·복원 평가는 포함하지 않음.</p><table><tr><th>조건</th><th>사실</th><th>동작</th><th>오류 구간</th></tr>']
    names={'initial':'추가 학습 전 — 기존 Astra 정답 Qwen + LoRA32','caption':'보강 캡션 학습 후 — 동일 Qwen + LoRA32, epoch2'}
    for a,t in result['totals'].items():html.append(f'<tr><td>{names[a]}</td><td>{t["fact_hits"]}/{t["facts"]}</td><td>{t["action_hits"]}/{t["actions"]}</td><td>{t["error_windows"]}/{t["windows"]}</td></tr>')
    html.append('</table>')
    html.append('<p>12원본을 단위로 재표집한 변화량의 95% 참고 구간(학습 후−전, 퍼센트포인트). 작은 표본의 참고 구간이며 성능 우위의 확정 근거는 아니다.</p><ul>')
    metric_names={'fact_hits':'사실 재현','action_hits':'동작·변화 재현','error_windows':'오류 구간 비율'}
    for m,d in result['uncertainty'].items():
        ci=d['delta_percentage_point_interval_95']
        html.append(f'<li>{metric_names[m]}: {ci[0]:+.1f} ~ {ci[1]:+.1f}%p</li>' if ci else f'<li>{metric_names[m]}: 산출 불가</li>')
    html.append('</ul><p>추가 학습·선택기 변경·기본 경로 교체는 하지 않았다. 초기 모델은 무학습 Qwen이나 Astra 직접 출력이 아니다. <a href="source_reference_cross_audit.json">새 정답 교차 검수</a> · <a href="source_audit_summary.json">기존 정답 감사</a></p>')
    html.append('<h2>기존 정답의 별도 원본 검수</h2><p>supported는 기존 주석이 지지됨, partial은 일부 과장 또는 불확실, unsupported는 지지되지 않음이다. 반례 문장의 거짓 여부가 아니라 주어진 판정·대조 관계의 정확성을 검수했다.</p><table><tr><th>자료</th><th>항목</th><th>지지됨</th><th>부분·불확실</th><th>지지 안 됨</th></tr>')
    for split,s in result['source_audit']['splits'].items():
        for field,c in s['counts'].items():html.append(f'<tr><td>{split}</td><td>{field}</td><td>{c.get("supported",0)}</td><td>{c.get("partial",0)}</td><td>{c.get("unsupported",0)}</td></tr>')
    html.append('</table>')
    for r in details:
        html.append(f'<details><summary>{escape(r["id"])}</summary><img src="{escape(os.path.relpath(r["source_sheet"],RUN))}"><ul>')
        html.extend(f'<li>{escape(f["text"])}</li>' for f in r['facts']);html.append('</ul><table>')
        for a,c in r['cells'].items():
            review=c['review'];explanations=[escape(review['rationale'])]
            for err in review['errors']:explanations.append('<p><strong>'+escape(err['kind'])+'</strong>: “'+escape(err['quote'])+'” — '+escape(err['reason'])+' (근거 프레임 '+escape(str(err['source_indices']))+')</p>')
            explanations.append('<ul>')
            for f in r['facts']:
                label='완전' if review['fact_hits'][f['id']] else '부분' if f['id'] in review['partial_fact_ids'] else '누락'
                explanations.append(f'<li>{escape(f["id"])} — {label}: {escape(review["fact_evidence"][f["id"]]["rationale"])}</li>')
            explanations.append('</ul>')
            html.append(f'<tr><th>{names[a]}</th><td>{escape(c["output"]["caption"])}</td><td>{"".join(explanations)}</td></tr>')
        html.append('</table></details>')
    (RUN/'report.html').write_text('\n'.join(html))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['freeze-reference','packet','score']);args=p.parse_args()
    {'freeze-reference':freeze_reference,'packet':packet,'score':score}[args.command]()
