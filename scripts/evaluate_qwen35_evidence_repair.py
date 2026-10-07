"""Deduplicate captions for explicit source review, then aggregate judgments."""
import argparse
from collections import Counter
from html import escape
from pathlib import Path
import random

from semantic_transmission.qwen35_evidence_repair_eval import RUN, DATA, ARMS, verify, validate_record
from semantic_transmission.qwen35_training_data import read_json, write_json, sha256
from evaluate_qwen35_astra_transfer import digest
from evaluate_qwen35_grounded import validate_grounded_review
from evaluate_qwen35_oracle_diagnostic import validate_mapping


def generated():
    p = verify(); freeze = read_json(RUN/'generation_freeze.json')
    ph = sha256(RUN/'protocol.json')
    if freeze['protocol_sha256'] != ph or set(freeze['files']) != {f'records/{r["id"]}.json' for r in p['rows']}:
        raise ValueError('Incomplete/changed generation freeze')
    result = {}
    for row in p['rows']:
        name = f'records/{row["id"]}.json'
        if sha256(RUN/name) != freeze['files'][name]:
            raise ValueError('Changed generated caption')
        record = read_json(RUN/name); validate_record(row, record, ph)
        result.update({(row['id'], arm): c for arm, c in record['cells'].items()})
    return p, result


def packet():
    p, outputs = generated(); rng = random.Random(20261007)
    refs = {r['id']: r for r in read_json(RUN/'reference.json')['records']}
    items = []; mapping = {}
    for row in p['rows']:
        rid = row['id']; order = list(ARMS); rng.shuffle(order); dedup = {}
        for arm in order:
            cell = outputs[rid, arm]; text = cell['caption']
            if text not in dedup:
                label = f'r{len(items):03d}'; dedup[text] = label
                ref = refs[rid]
                items.append(dict(label=label, row_id=rid, caption=text, caption_sha256=digest(text),
                    keys=cell['keys'], facts=ref['facts'], source_sheet=ref['source_sheet'], source_frames=ref['source_frames']))
                mapping[label] = []
            mapping[dedup[text]].append(dict(row_id=rid, cell=arm))
    obj = dict(records=items, generation_freeze_sha256=sha256(RUN/'generation_freeze.json'),
        reference_freeze_sha256=sha256(RUN/'reference_freeze.json'),
        policy='Condition labels hidden in packet, but same Codex context built data and knows the experiment. Not an independent/blinded evaluator. Exact duplicates within a window share a review.')
    path = RUN/'packet.json'
    if path.exists() and read_json(path) != obj:
        raise ValueError('Changed review packet')
    write_json(path, obj); write_json(RUN/'mapping_private.json', mapping)
    print('REVIEW_ITEMS', len(items), 'CONDITION_WINDOWS', len(outputs))


def gate(candidate, control):
    return dict(noninferior=candidate['fact_hits'] >= control['fact_hits'] and candidate['action_hits'] >= control['action_hits']
        and candidate['error_windows'] <= control['error_windows'] and candidate['generation_failures'] <= control['generation_failures'],
        strict_gain=candidate['fact_hits'] > control['fact_hits'] or candidate['action_hits'] > control['action_hits'] or candidate['error_windows'] < control['error_windows'])


def score():
    p, outputs = generated(); pack = read_json(RUN/'packet.json'); review = read_json(RUN/'review.json')
    if review['packet_sha256'] != sha256(RUN/'packet.json') or pack['generation_freeze_sha256'] != sha256(RUN/'generation_freeze.json') or pack['reference_freeze_sha256'] != sha256(RUN/'reference_freeze.json'):
        raise ValueError('Review provenance changed')
    scores = validate_grounded_review(pack, review)
    mapping = read_json(RUN/'mapping_private.json'); validate_mapping(pack, mapping, outputs)
    linked = {(x['row_id'], x['cell']): scores[label] for label, xx in mapping.items() for x in xx}
    refs = {r['id']: r for r in read_json(RUN/'reference.json')['records']}
    errata = read_json(RUN/'reference_errata.json')
    if errata['reference_sha256'] != sha256(RUN/'reference.json'):
        raise ValueError('Errata belongs to different references')
    for name, digest_value in errata['source_evidence'].items():
        if sha256(RUN/name) != digest_value:
            raise ValueError('Changed source reinspection evidence')
    totals = {a: Counter() for a in ARMS}; sources = {}; details = []
    for row in p['rows']:
        rid = row['id']; excluded = set(errata['withdrawn_facts'].get(rid, []))
        if not excluded <= {f['id'] for f in refs[rid]['facts']}:
            raise ValueError('Errata refers to an unknown fact')
        facts = [f for f in refs[rid]['facts'] if f['id'] not in excluded]
        valid = {f['id'] for f in facts}; actions = {f['id'] for f in facts if f['kind'] != 'state'}
        sources.setdefault(row['source_id'], {a: Counter() for a in ARMS}); cells = {}
        for arm in ARMS:
            out = outputs[rid, arm]; judged = linked[rid, arm]; good = out['status'] == 'COMPLETE'
            hits = {f for f, v in judged['fact_hits'].items() if v and f in valid} if good else set()
            partial = set(judged['partial_fact_ids']) & valid if good else set(); errors = judged['errors']
            counts = dict(windows=1, facts=len(facts), fact_hits=len(hits), actions=len(actions), action_hits=len(hits & actions),
                full_or_partial_hits=len(hits | partial), action_full_or_partial_hits=len((hits | partial) & actions),
                error_windows=int(bool(errors)), error_claims=len(errors), generation_failures=int(not good),
                contradicted_windows=int(any(e['kind']=='contradicted' for e in errors)),
                unsupported_windows=int(any(e['kind']=='unsupported' for e in errors)),
                caption_words=len(out['caption'].split()), generation_seconds=out['seconds'], generated_tokens=out['generated_tokens'])
            totals[arm].update(counts); sources[row['source_id']][arm].update(counts)
            cells[arm] = dict(output=out, review=judged, counts=counts)
        details.append(dict(id=rid, source_id=row['source_id'], source_sheet=refs[rid]['source_sheet'], facts=facts, cells=cells))
    gates = {}
    for arm in ARMS[1:]:
        controls = ['initial'] if arm == 'caption' else ['initial','caption']
        checks = {control: gate(totals[arm], totals[control]) for control in controls}
        gates[arm] = dict(passed=all(c['noninferior'] and c['strict_gain'] for c in checks.values()), comparisons=checks)
    sequential = {arm: gate(totals[arm], totals[ARMS[i-1]]) for i, arm in enumerate(ARMS) if i>0}
    result = dict(status='COMPLETE_DEVELOPMENT_REVIEW_NOT_INDEPENDENT', totals=totals, per_source=sources,
        gates=gates, incremental_comparisons=sequential, training_audit_path=str(RUN/'training_audit.json'),
        unique_reviewed_captions=len(scores), condition_windows=len(outputs), windows=len(p['rows']), source_originals=len(sources),
        optimizer_steps_this_evaluation=0, backward_calls_this_evaluation=0, independent_final_test=False,
        default_pipeline_changed=False, annotator='Codex assistant source review, not independent Astra or human',
        original_reference_facts=62, withdrawn_reference_facts=4, source_reference_correction=errata,
        provenance={n:sha256(RUN/n) for n in ['protocol.json','generation_freeze.json','reference_freeze.json','reference.json','reference_errata.json','packet.json','review.json','mapping_private.json','training_audit.json']},
        timing_scope='Per-caption preprocessing/generation/decode only; shared visual encoding, loading, judging, transmission and reconstruction excluded.',
        limitations='Four previously observed development originals, single training seed, manual non-independent AI reference/judgment, fixed keys. Not a new final test or full FC-LGVSC performance proof.')
    write_json(RUN/'paired_details.json', dict(records=details)); write_json(RUN/'RESULT.json', result)
    old = read_json(DATA/'RESULT.json')
    if 'preparation_snapshot' not in old:
        old = dict(preparation_snapshot=old)
    old.update(status='COMPLETE_TRAINING_AND_DEVELOPMENT_REVIEW', completed_training_arms=ARMS[1:],
        updates_per_arm=60, training_updates_total=240, analysis_optimizer_steps=0, evaluation_result=str(RUN/'RESULT.json'),
        independent_final_test_ready=False, default_pipeline_changed=False)
    write_json(DATA/'RESULT.json', old)
    render(result, details)
    print({a:{k:v[k] for k in ['fact_hits','facts','action_hits','actions','error_windows','windows']} for a,v in totals.items()})


def render(result, details):
    parts = ['<!doctype html><meta charset="utf-8"><title>Qwen evidence repair development comparison</title>',
        '<style>body{font:16px sans-serif;max-width:1250px;margin:30px auto}table{border-collapse:collapse;width:100%}td,th{border:1px solid #bbb;padding:9px;vertical-align:top}img{max-width:100%}summary{padding:14px;cursor:pointer}</style>',
        '<h1>같은 4키의 캡션 비교 — 개발 4원본·16구간</h1><p>추가 학습 전 Astra 정답 rank32 Qwen과 네 추가 학습 조건. Codex 원본 대조이며 독립 Astra/사람 평가가 아닙니다. 새 최종 평가, 키 선택, 전송, 복원은 미검증입니다.</p><p>도시 장면의 고정 구도 정답 4개는 원본 재확인에서 오류로 판명되어 분자·분모에서 제외했습니다. 원래 정답과 사후 정정 이력은 보존합니다.</p>',
        '<table><tr><th>조건</th><th>사실</th><th>동작</th><th>오류 구간</th><th>생성 시간</th></tr>']
    for a,t in result['totals'].items():
        parts.append(f'<tr><td>{a}</td><td>{t["fact_hits"]}/{t["facts"]}</td><td>{t["action_hits"]}/{t["actions"]}</td><td>{t["error_windows"]}/{t["windows"]}</td><td>{t["generation_seconds"]:.1f}s</td></tr>')
    parts.append('</table>')
    for row in details:
        import os
        parts.append(f'<details><summary>{escape(row["id"])}</summary><img src="{escape(os.path.relpath(row["source_sheet"],RUN))}"><ol>')
        for f in row['facts']: parts.append(f'<li>{f["id"]}: {escape(f["text"])}</li>')
        parts.append('</ol><table><tr><th>조건</th><th>캡션</th><th>판정</th></tr>')
        for a,c in row['cells'].items():
            errors='; '.join(e['quote']+': '+e['reason'] for e in c['review']['errors'])
            parts.append(f'<tr><td>{a}</td><td>{escape(c["output"]["caption"])}</td><td>{escape(c["review"]["rationale"])}<br>{escape(errors)}</td></tr>')
        parts.append('</table></details>')
    (RUN/'report.html').write_text('\n'.join(parts))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=['packet','score'])
    {'packet':packet,'score':score}[parser.parse_args().command]()
