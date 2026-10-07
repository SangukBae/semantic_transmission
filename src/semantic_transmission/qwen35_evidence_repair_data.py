"""Source-reviewed caption/evidence repair data, with no model execution.

Only existing train/development rows are read. Old final-test files and images
are not opened. Raw sources, frozen experiments and checkpoints are read-only.
"""
from collections import Counter
from copy import deepcopy
from html import escape
from itertools import product
from pathlib import Path

from .qwen35_training_data import ROOT, read_json, write_json, sha256

CONFIG = ROOT / 'configs/qwen35_evidence_repair_v3.json'
RUN = ROOT / 'outputs/qwen35_evidence_repair_v3'
VERDICTS = ('supported', 'contradicted', 'insufficient')


def ordered_paths(query):
    groups = [p['positions'] for p in query['phases']]
    return sorted({tuple(p) for p in product(*groups)
                   if all(a < b for a, b in zip(p, p[1:]))})


def informative_queries(row):
    return [q for q in row.get('evidence_queries', [])
            if len(q['phases']) > 1 or len(q['phases'][0]['positions']) < 4]


def expand_reviewed_queries(queries):
    """Expose adjacent transitions already established by a reviewed path.

    Keep complete paths as well. This never labels extra/unseen frames and does
    not relabel a continuously visible event to force a preferred position.
    """
    result = deepcopy(queries)
    for index, q in enumerate(queries):
        if len(q['phases']) < 3:
            continue
        for phase in range(len(q['phases'])-1):
            pair = deepcopy(q['phases'][phase:phase+2])
            result.append(dict(text=f'The shown sequence first depicts: {pair[0]["text"]} It then depicts: {pair[1]["text"]}',
                phases=pair, derived_from_query=index, derived_phase_positions=[phase, phase+1],
                derivation='Adjacent source-reviewed phases; a shown edit is not asserted to be continuous physical action.'))
    return result


def validate_patch(patch, row):
    if patch['id'] != row['id'] or patch['split'] != row['split'] or row['split'] not in ('train', 'validation'):
        raise ValueError('Wrong row/split or forbidden test annotation')
    if patch['source_sha256'] != row['source_sha256'] or patch['frame_sha256'] != [row['frames'][k]['sha256'] for k in row['keys']]:
        raise ValueError('Annotation belongs to different source observations')
    if len(row['keys']) != 4 or not patch['caption'] or len(patch['caption'].split()) > 80:
        raise ValueError('Expected four keys and nonempty caption <=80 words')
    review = patch['review']
    if not review['source_only'] or set(review['viewed_positions']) != set(range(4)):
        raise ValueError('Every source observation must be inspected')
    if not patch['evidence_queries'] or not patch['claims'] or not patch['preference_pairs']:
        raise ValueError('Missing reviewed supervision')
    for q in patch['evidence_queries']:
        if not q['text'] or not 1 <= len(q['phases']) <= 4:
            raise ValueError('Evidence must contain one to four explicit phases')
        for phase in q['phases']:
            positions = phase['positions']
            if not phase['text'] or not positions or len(positions) != len(set(positions)) or any(type(i) is not int or i not in range(4) for i in positions):
                raise ValueError('Invalid visible evidence positions')
        if not ordered_paths(q):
            raise ValueError('No chronological complete evidence path')
    for claim in patch['claims']:
        obs = claim['observations']
        if claim['verdict'] not in VERDICTS or not claim['text'] or not claim['reason']:
            raise ValueError('Claim needs reviewed three-way judgment and reason')
        if not obs or obs != sorted(set(obs)) or any(type(i) is not int or i not in range(4) for i in obs):
            raise ValueError('Claim cites invalid or reordered observations')
    if set(c['verdict'] for c in patch['claims']) != set(VERDICTS):
        raise ValueError('Every reviewed window needs supported/contradicted/insufficient examples')
    for pair in patch['preference_pairs']:
        if pair['kind'] not in ('omission', 'contradiction') or not pair['reason'] or not pair['rejected'] or pair['rejected'] == patch['caption']:
            raise ValueError('Invalid reviewed caption contrast')
    return patch


def shortcut_audit(rows):
    queries = [q for r in rows for q in informative_queries(r)]
    paths = [set(ordered_paths(q)) for q in queries]
    temporal = [p for p in paths if len(next(iter(p))) > 1]
    universe = set().union(*paths) if paths else set()
    best = max((sum(candidate in p for p in paths) for candidate in universe), default=0)
    # Report observation-position rules, not shuffled letter labels.
    by_length = {}
    for size in range(1, 5):
        group = [p for p in paths if len(next(iter(p))) == size]
        candidates = set().union(*group) if group else set()
        by_length[str(size)] = dict(queries=len(group), best_constant_hits=max((sum(c in p for p in group) for c in candidates), default=0))
    return dict(queries=len(paths), temporal_queries=len(temporal),
                constant_AD_hits=sum((0, 3) in p for p in temporal),
                requires_middle_phase=sum(all(any(i in (1, 2) for i in seq) for seq in p) for p in paths),
                multi_phase_three_or_four=sum(len(next(iter(p))) >= 3 for p in paths),
                best_constant_full_path_hits=best, position_shortcut_by_phase_count=by_length,
                claim_counts=dict(Counter(c['verdict'] for r in rows for c in r.get('claims', []))),
                claim_input_sizes=dict(Counter(str(len(c['observations'])) for r in rows for c in r.get('claims', []))))


def config_at(path=CONFIG):
    cfg = read_json(path)
    if cfg['epochs'] <= 0 or cfg['gradient_accumulation'] <= 0 or cfg['lr'] <= 0:
        raise ValueError('Invalid training schedule')
    if (cfg['lora_rank'], cfg['lora_alpha']) != (32, 64):
        raise ValueError('This pilot fixes rank32/alpha64')
    if cfg['arms'] != {'caption': [], 'evidence': ['evidence'], 'judgment': ['evidence', 'judgment'], 'preference': ['evidence', 'judgment', 'preference']}:
        raise ValueError('Keep one-change-at-a-time cumulative comparison arms')
    for key in ('grounding_weight', 'judgment_weight', 'preference_weight', 'preference_margin'):
        if not 0 <= cfg[key] < float('inf'):
            raise ValueError('Nonfinite/negative objective setting')
    return cfg


def allocate_development(rows, source_ids, parent_training):
    """Move whole previously new sources; never reuse a parent's training source."""
    result = deepcopy(rows)
    source_ids = set(source_ids)
    current = {r['source_id'] for r in rows['train']}
    parent_ids = {r['source_id'] for r in parent_training}
    parent_hashes = {r['source_sha256'] for r in parent_training}
    moving = [r for r in result['train'] if r['source_id'] in source_ids]
    if not source_ids <= current or source_ids & parent_ids or any(r['source_sha256'] in parent_hashes or r['origin'] != 'grounded_expansion' for r in moving):
        raise ValueError('Development allocation overlaps the chosen parent training sources')
    result['train'] = [r for r in result['train'] if r['source_id'] not in source_ids]
    for r in moving:
        r['previous_split'] = r['split']; r['split'] = 'validation'
    result['validation'].extend(moving)
    return result


def prepare(run=RUN, config_path=CONFIG):
    if (run / 'freeze.json').exists():
        return verify(run, config_path)
    cfg = config_at(config_path)
    old = ROOT / cfg['source_run']
    # Consult frozen manifests without opening the old test content.
    old_freeze = read_json(old / 'data_freeze.json')
    rows = {}
    for split in ('train', 'validation'):
        name = split + '.json'
        if sha256(old / name) != old_freeze['files'][name]:
            raise ValueError('Parent train/development data changed')
        rows[split] = read_json(old / name)['records']
    parent_training_path = ROOT / 'outputs/qwen35_astra_transfer_v1/train_teacher.json'
    rows = allocate_development(rows, cfg.get('development_source_ids', []), read_json(parent_training_path)['records'])
    source_sets = {s: {r['source_sha256'] for r in rr} for s, rr in rows.items()}
    if source_sets['train'] & source_sets['validation']:
        raise ValueError('Source leakage between train and development')
    usage = read_json(old / 'source_usage.json')
    protected = set(usage['held_out_originals']['test'])
    if any(r['source_id'] in protected for rr in rows.values() for r in rr):
        raise ValueError('Old final-test source reused')
    patches = {r['id']: r for r in cfg['records']}
    known = {r['id']: r for rr in rows.values() for r in rr}
    if len(patches) != len(cfg['records']) or not set(patches) <= set(known):
        raise ValueError('Duplicate/unknown reviewed row')
    sheets = {}
    for patch in patches.values():
        validate_patch(patch, known[patch['id']])
        review = patch['review']
        if sha256(review['sheet']) != review['sheet_sha256']:
            raise ValueError('Changed reviewed image sheet')
        sheets[review['sheet']] = review['sheet_sha256']
    # The parent is the previous Astra-r32 model, never either failed new arm.
    parent = ROOT / cfg['parent_checkpoint']
    for rel, digest in read_json(parent / 'complete.json')['files'].items():
        if sha256(parent / rel) != digest:
            raise ValueError('Changed starting checkpoint')
    source_frames = {}
    enriched = {}
    for split, rr in rows.items():
        enriched[split] = []
        for original in rr:
            r = deepcopy(original)
            for frame in r['frames']:
                if sha256(frame['path']) != frame['sha256']:
                    raise ValueError('Changed train/development source frame')
                source_frames[frame['path']] = frame['sha256']
            if r['id'] in patches:
                patch = patches[r['id']]
                for key in ('caption', 'evidence_queries', 'claims', 'preference_pairs', 'review'):
                    r[key] = deepcopy(patch[key])
                r['evidence_queries'] = expand_reviewed_queries(r['evidence_queries'])
                r['supervision_origin'] = 'Codex source-reviewed repair; original frozen record retained separately'
            else:
                r.update(evidence_queries=[], claims=[], preference_pairs=[], supervision_origin='unchanged previous Astra caption; no unreviewed auxiliary negatives')
            enriched[split].append(r)
        write_json(run / f'{split}.json', dict(records=enriched[split]))
    protocol = dict(config_sha256=sha256(config_path), config_path=str(config_path),
                    parent_checkpoint=str(parent), parent_complete_sha256=sha256(parent / 'complete.json'),
                    source_train_sha256=sha256(old / 'train.json'), source_validation_sha256=sha256(old / 'validation.json'),
                    source_usage_sha256=sha256(old / 'source_usage.json'), source_freeze_sha256=sha256(old / 'data_freeze.json'),
                    parent_training_sha256=sha256(parent_training_path),
                    development_allocation='One source formerly used in the recent failed arms is removed in full from this training set. It was not in the selected older Astra-r32 parent training set. Development only, not a new independent final test.',
                    config=dict(read_json(old / 'protocol.json')['config'], lora_rank=32, lora_alpha=64),
                    settings={k: v for k, v in cfg.items() if k != 'records'},
                    final_test_policy='No new final test exists. Existing test sources remain excluded, not loaded or relabeled. Validation is previously used development data.',
                    auxiliary_policy='Full chronological evidence paths including all intermediate phases; separate supported/contradicted/insufficient claims; uncertainty never treated as false.',
                    input_policy='Caption always uses unchanged four keys; claim-only training uses explicitly reviewed1/2/4-observation subsets, same subsets across relevant arms.',
                    default_pipeline_changed=False)
    write_json(run / 'protocol.json', protocol)
    audit = dict(status='PREPARED_NO_TRAINING', optimizer_steps=0, model_loaded=False,
                 splits={s: dict(originals=len({r['source_id'] for r in rr}), windows=len(rr),
                     reviewed_windows=sum(bool(r['evidence_queries']) for r in rr), **shortcut_audit(rr)) for s, rr in enriched.items()},
                 limitation='Small source-reviewed pilot; no additional original videos or independent teacher. Conditional position shortcuts and source diversity still need measurement; not an accuracy claim.',
                 gpu_training_validated=False, independent_final_test_ready=False)
    write_json(run / 'data_audit.json', audit)
    files = ['protocol.json', 'train.json', 'validation.json', 'data_audit.json']
    write_json(run / 'freeze.json', dict(files={p: sha256(run / p) for p in files},
        annotation_sha256=sha256(config_path), source_frames=source_frames, reviewed_sheets=sheets))
    write_json(run / 'RESULT.json', dict(status='PREPARED_NOT_TRAINED', optimizer_steps=0, model_loaded=False,
        training_arms=list(cfg['arms']), default_pipeline_changed=False, gpu_training_validated=False,
        independent_final_test_ready=False, data_audit_path=str(run / 'data_audit.json')))
    render_review(run, enriched)
    return verify(run, config_path)


def verify(run=RUN, config_path=CONFIG):
    f = read_json(run / 'freeze.json')
    if sha256(config_path) != f['annotation_sha256']:
        raise ValueError('Changed frozen annotations/config; use a new run')
    for rel, digest in f['files'].items():
        if sha256(run / rel) != digest:
            raise ValueError('Changed prepared data')
    for name, digest in {**f['source_frames'], **f['reviewed_sheets']}.items():
        if sha256(name) != digest:
            raise ValueError('Changed source evidence')
    p = read_json(run / 'protocol.json')
    parent = Path(p['parent_checkpoint'])
    if sha256(parent / 'complete.json') != p['parent_complete_sha256']:
        raise ValueError('Changed parent manifest')
    for name, digest in read_json(parent / 'complete.json')['files'].items():
        if sha256(parent / name) != digest:
            raise ValueError('Changed parent checkpoint')
    return p


def render_review(run, splits):
    out = ['<!doctype html><meta charset="utf-8"><title>Evidence repair source review</title>',
           '<style>body{font:16px system-ui;max-width:1200px;margin:auto;padding:24px}img{max-width:100%}details{margin:20px 0}li{margin:8px 0}</style>',
           '<h1>근거·순서·움직임 보강 자료 — 학습 미실행</h1><p>새 주석은 Codex 원본 검수. 기존 Astra 주석·체크포인트 보존. 검증은 기존 개발 자료이며 새 최종 평가가 아닙니다.</p>']
    for split, rows in splits.items():
        for row in rows:
            if not row['evidence_queries']:
                continue
            import os
            sheet = os.path.relpath(row['review']['sheet'], run)
            out.append(f'<details><summary>{escape(split)} / {escape(row["id"])}</summary><img src="{escape(sheet)}"><p>{escape(row["caption"])}</p><ul>')
            for q in row['evidence_queries']:
                paths = ['→'.join('ABCD'[i] for i in path) for path in ordered_paths(q)]
                out.append(f'<li>{escape(q["text"])} — {escape(", ".join(paths))}</li>')
            for c in row['claims']:
                obs = ''.join('ABCD'[i] for i in c['observations'])
                out.append(f'<li>[{obs} / {c["verdict"]}] {escape(c["text"])} — {escape(c["reason"])}</li>')
            out.append('</ul></details>')
    (run / 'review.html').write_text('\n'.join(out))
