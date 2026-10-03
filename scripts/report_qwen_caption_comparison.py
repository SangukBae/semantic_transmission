"""Build an auditable report from completed captions and manual source reviews.

This does not infer correctness from caption similarity or run an automatic judge.
"""
import argparse
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import html
import json
import os
from pathlib import Path
import re
import statistics


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    repo = Path(__file__).resolve().parents[1]
    data = read(root / 'inputs.json')
    cfg = read(root / 'inference_config.json')
    runtime = read(root / 'runtime.json')
    review = read(root / 'review_annotations.json')
    assert sha(root / 'inputs.json') == cfg['inputs_sha256']
    assert sha(repo / 'scripts/run_qwen_caption_comparison.py') == cfg['inference_code_sha256']
    assert sha(repo / 'scripts/qwen35_caption.py') == cfg['helper_code_sha256']
    identity = {k: v for k, v in cfg.items() if k != 'signature'}
    assert hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest() == cfg['signature']
    for name, expected in data['input_files_sha256'].items():
        assert sha(name) == expected, name
    source = Path(data['source_frames'])
    for frame, expected in data['source_png_hashes'].items():
        assert sha(source / f'{frame}.png') == expected, frame
    baseline = read(repo / 'outputs/caption_accuracy_tv_low_08_20261002/RESULT.json')
    base = {x['assistant_segment']: x for x in baseline['annotations']}
    supplemental = {x['segment']: x for x in read(root / 'fc_supplemental_review.json')['annotations']}
    notes = {x['segment']: x for x in review['annotations']}
    assert len(notes) == len(review['annotations']) == len(data['records']) == 78
    assert len(list((root / 'captions').glob('*.json'))) == 78
    rows, captures = [], []
    for item in data['records']:
        i = item['segment']
        cap = read(root / 'captions' / f'{i:03d}.json')
        ann = notes[i]
        for field in ['segment', 'matched_pair', 'start', 'end_exclusive', 'source_indices']:
            assert cap[field] == item[field], (i, field)
        assert cap['signature'] == runtime['signature'] == cfg['signature']
        assert cap['model_id'] == cfg['model_id'] and cap['model_revision'] == cfg['revision']
        assert cap['caption'].strip() and not cap['truncated']
        assert cap['inference_provider'] == 'LOCAL_ONLY'
        assert ann['source_indices'] == item['source_indices'] and ann['directly_reviewed']
        assert ann['qwen']['clear_factual_error'] == bool(ann['qwen']['clear_factual_errors'])
        for error in ann['qwen']['clear_factual_errors']:
            assert error['claim'].lower() in cap['caption'].lower()
        if i in base:
            assert ann['pllava'] == base[i]['pllava']
            assert ann['fc_lgvsc'] == base[i]['assistant_v2']
        else:
            assert ann['fc_lgvsc'] == supplemental[i]['fc_lgvsc']
        captions = dict(qwen=cap['caption'], fc_lgvsc=item['fc_lgvsc_caption'], pllava=item['pllava_caption'])
        rows.append(dict(segment=i, matched_pair=item['matched_pair'],
                         start_seconds=item['start'] / data['fps'],
                         end_seconds=item['end_exclusive'] / data['fps'],
                         source_indices=item['source_indices'], captions=captions, review=ann))
        captures.append(cap)
    matched = [x for x in rows if x['matched_pair'] is not None]
    assert len(matched) == 65

    def stats(items, model):
        n = len(items)
        errors = sum(x['review'][model]['clear_factual_error'] for x in items)
        omissions = sum(bool(x['review'][model]['visible_entity_omission']) for x in items)
        words = [len(x['captions'][model].split()) for x in items]
        return dict(reviewed_segments=n, clear_error_segments=errors,
                    clear_error_segment_percent=100 * errors / n,
                    visible_entity_omission_segments=omissions,
                    visible_entity_omission_percent=100 * omissions / n,
                    mean_words=statistics.mean(words), over_80_words_segments=sum(w > 80 for w in words))

    common = {m: stats(matched, m) for m in ['pllava', 'qwen', 'fc_lgvsc']}
    full = {m: stats(rows, m) for m in ['qwen', 'fc_lgvsc']}
    transitions = {}
    for name, field in [('clear_error', 'clear_factual_error'), ('omission', 'visible_entity_omission')]:
        groups = {k: [] for k in ['resolved', 'remaining', 'new', 'neither']}
        for row in matched:
            before, after = (bool(row['review'][m][field]) for m in ['pllava', 'qwen'])
            key = 'remaining' if before and after else 'resolved' if before else 'new' if after else 'neither'
            groups[key].append(row['segment'])
        before_count = len(groups['remaining']) + len(groups['resolved'])
        net = len(groups['resolved']) - len(groups['new'])
        transitions[name] = dict(segment_ids=groups, counts={k: len(v) for k, v in groups.items()},
                                 net_reduction_segments=net, relative_reduction_percent=100 * net / before_count)
    times = [sum(a['generation_seconds'] for a in c['attempts']) for c in captures]
    finished = max(datetime.fromisoformat(c['created_utc']) for c in captures)
    ready = datetime.fromisoformat(runtime['created_utc'])
    perf = dict(model_load_seconds=runtime['model_load_seconds'],
                generation_seconds_total=sum(times), generation_seconds_mean=statistics.mean(times),
                approximate_load_plus_batch_wall_seconds=(finished - ready).total_seconds() + runtime['model_load_seconds'],
                started_utc_approx=(ready - timedelta(seconds=runtime['model_load_seconds'])).isoformat(),
                finished_utc=finished.isoformat(), generated_tokens=sum(c['generated_tokens'] for c in captures),
                truncated_captions=sum(c['truncated'] for c in captures),
                retried_captions=sum(len(c['attempts']) > 1 for c in captures),
                peak_pytorch_allocated_gib=max(c['peak_allocated_bytes'] for c in captures) / 2**30,
                peak_pytorch_reserved_gib=max(c['peak_reserved_bytes'] for c in captures) / 2**30,
                memory_scope='PyTorch process allocation/reservation; excludes desktop and other processes')
    limitations = [
        'Single development video tv_low_08; adjacent segments are not independent error events.',
        'Manual AI source review, not blinded or independent human ground truth; zero detected contradictions is not 100% accuracy.',
        'Qwen received the FC v2 instruction and four original frames; FC/PLLaVA caption texts and review annotations were not model inputs.',
        'FC v2 was revised after development reconstruction failures; PLLaVA used different prompts/presentation. This compares caption workflows, not model-only performance.',
        'NF4 quantization, greedy single-pass captions. No BF16 comparison, no new PLLaVA inference, and no reconstructed-video quality evaluation.',
        'Clear errors and visible-entity omissions can overlap; do not add the two counts. Unverified notes are examples, not exhaustive unsupported-claim counts.',
    ]
    result = dict(status='LOCAL_QWEN_CAPTION_COMPARISON_COMPLETE_AI_SOURCE_REVIEW',
                  created_utc=datetime.now(timezone.utc).isoformat(), model=cfg['model_id'], revision=cfg['revision'],
                  quantization=cfg['quantization'], source=data['source'], inference_executed=True,
                  local_only=True, gpu_used=True, independent=False, blinded=False,
                  ground_truth=review['ground_truth'], all_fc_segments=78, exact_three_way_segments=65,
                  all_fc_covered_seconds=data['all_fc_covered_seconds'],
                  matched_covered_seconds=data['matched_pllava_covered_seconds'],
                  source_identity_verified=True, input_frames_and_intervals_verified=True,
                  primary_matched_summary=common, full_78_summary=full,
                  pllava_to_qwen_transitions=transitions, performance=perf,
                  limitations=limitations, review_file='review_annotations.json',
                  frozen_baseline_review='outputs/caption_accuracy_tv_low_08_20261002/RESULT.json',
                  protocol='inputs.json', inference_config='inference_config.json')
    write(root / 'RESULT.json', result)
    with (root / 'segments.csv').open('w', newline='') as f:
        fields = ['segment', 'matched_pair', 'start_seconds', 'end_seconds', 'source_indices']
        fields += [f'{m}_{field}' for m in ['pllava', 'qwen', 'fc_lgvsc'] for field in ['caption', 'clear_error', 'omission']]
        out = csv.DictWriter(f, fieldnames=fields)
        out.writeheader()
        for row in rows:
            flat = {k: row[k] for k in fields[:5]}
            for m in ['pllava', 'qwen', 'fc_lgvsc']:
                ann = row['review'][m]
                flat.update({f'{m}_caption': row['captions'][m], f'{m}_clear_error': ann['clear_factual_error'] if ann else '',
                             f'{m}_omission': ann['visible_entity_omission'] if ann else ''})
            out.writerow(flat)
    esc = html.escape
    parts = ['<!doctype html><html lang="ko"><meta charset="utf-8"><title>Qwen 캡션 비교</title>',
             '<style>body{font:16px/1.6 system-ui;margin:2rem auto;max-width:1450px;padding:0 1rem;color:#20242a}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:.5rem}article{border-top:3px solid #999;margin-top:2rem;padding-top:1rem}.frames,.captions{display:grid;gap:12px}.frames{grid-template-columns:repeat(4,1fr)}.captions{grid-template-columns:repeat(3,1fr)}img{width:100%;height:auto}.bad{color:#a71919}.note{color:#755800}button{margin:.4rem;padding:.5rem}@media(max-width:850px){.captions{grid-template-columns:1fr}.frames{grid-template-columns:repeat(2,1fr)}}</style>',
             '<h1>Qwen3.5-9B / PLLaVA / FC-LGVSC 캡션 비교</h1>',
             '<p>tv_low_08 원본 약 60초 · Qwen 로컬 GPU 78개 완료 · 세 모델의 동일 프레임/구간 65개를 주 비교 대상으로 사용.</p>',
             '<p><strong>AI가 원본을 보고 판정한 구간별 발견율이며 독립 정답 평가가 아닙니다.</strong> 오류 사건 수나 모델 정확도로 해석하지 마세요.</p>',
             '<table><tr><th>동일 65구간</th><th>명백한 오류 포함</th><th>사람·개·차량 누락 포함</th><th>평균 단어</th></tr>']
    labels = {'pllava': 'PLLaVA 기존 캡션', 'qwen': 'Qwen3.5-9B NF4', 'fc_lgvsc': 'FC-LGVSC 기존 GPT 작성 v2'}
    for model, v in common.items():
        parts.append(f'<tr><td>{labels[model]}</td><td>{v["clear_error_segments"]}/65 ({v["clear_error_segment_percent"]:.1f}%)</td><td>{v["visible_entity_omission_segments"]}/65 ({v["visible_entity_omission_percent"]:.1f}%)</td><td>{v["mean_words"]:.1f}</td></tr>')
    parts.append('</table><p>확정 오류 순감소: {}구간 ({:.1f}%). 누락 순감소: {}구간 ({:.1f}%). 두 지표는 합산하지 않습니다.</p>'.format(transitions['clear_error']['net_reduction_segments'], transitions['clear_error']['relative_reduction_percent'], transitions['omission']['net_reduction_segments'], transitions['omission']['relative_reduction_percent']))
    parts.append('<p>Qwen 전체 78개: 사실 오류 {}구간, 누락 {}구간. FC 전체: 사실 오류 {}구간, 누락 {}구간. Qwen 생성 평균 {:.2f}초/구간, 모델 로딩 포함 약 {:.1f}분, PyTorch 최대 할당 {:.2f} GiB.</p>'.format(full['qwen']['clear_error_segments'],full['qwen']['visible_entity_omission_segments'],full['fc_lgvsc']['clear_error_segments'],full['fc_lgvsc']['visible_entity_omission_segments'],perf['generation_seconds_mean'],perf['approximate_load_plus_batch_wall_seconds']/60,perf['peak_pytorch_allocated_gib']))
    parts.append('<p>80단어 초과: Qwen {} / 78. 형식 준수 문제는 위의 시각적 오류 수와 별도입니다.</p>'.format(full['qwen']['over_80_words_segments']))
    parts.append('<details><summary>평가 기준·제약·재현 정보</summary><ul>' + ''.join('<li>'+esc(t)+'</li>' for t in review['adjudication_notes'] + limitations) + '</ul><p><a href="RESULT.json">집계 JSON</a> · <a href="review_annotations.json">모든 수동 주석</a> · <a href="segments.csv">CSV</a> · <a href="inference_config.json">추론 설정</a></p></details>')
    parts.append('<p><button onclick="filterRows(\'all\')">전체 78</button><button onclick="filterRows(\'matched\')">동일 입력 65</button><button onclick="filterRows(\'error\')">Qwen 확정 오류</button><button onclick="filterRows(\'new\')">PLLaVA에 없던 Qwen 오류</button></p>')
    new_ids = transitions['clear_error']['segment_ids']['new']
    for row in rows:
        i = row['segment']
        parts.append(f'<article id="segment-{i}" data-matched="{int(row["matched_pair"] is not None)}" data-error="{int(row["review"]["qwen"]["clear_factual_error"])}" data-new="{int(i in new_ids)}"><h2>구간 {i}: {row["start_seconds"]:.3f}–{row["end_seconds"]:.3f}초</h2><div class="frames">')
        for frame in row['source_indices']:
            link = os.path.relpath(source / f'{frame}.png', root)
            parts.append(f'<div>원본 프레임 {frame}<a href="{esc(link)}"><img loading="lazy" src="{esc(link)}" alt="원본 {frame}"></a></div>')
        parts.append('</div><p>원본 관찰: '+esc(row['review']['source_observation_ko'])+'</p><div class="captions">')
        for m in ['pllava', 'qwen', 'fc_lgvsc']:
            ann = row['review'][m]
            parts.append('<section><h3>'+labels[m]+'</h3>')
            if ann is None:
                parts.append('<p>정확히 일치하는 기존 PLLaVA 구간 없음</p></section>')
                continue
            parts.append('<p>'+esc(row['captions'][m])+'</p>')
            parts.append('<p class="bad">'+ ('확정 오류 있음' if ann['clear_factual_error'] else '확정 오류 발견 안 됨')+'</p>')
            for entry in ann['clear_factual_errors']:
                parts.append('<p class="bad">“'+esc(entry['claim'])+'” — '+esc(entry['reason_ko'])+'</p>')
            parts.append('<p>객체 누락: '+esc(ann['visible_entity_omission'] or '발견 안 됨')+'</p>')
            for entry in ann['unverified_details']:
                parts.append('<p class="note">미확인: '+esc(entry.get('claim', ''))+' '+esc(entry['reason_ko'])+'</p>')
            parts.append('</section>')
        parts.append('</div></article>')
    parts.append('<script>function filterRows(mode){document.querySelectorAll("article").forEach(x=>x.hidden=mode!=="all"&&x.dataset[mode]!=="1")}</script></html>')
    (root / 'review.html').write_text('\n'.join(parts))
    write(root / 'artifact_manifest.json', dict(created_utc=result['created_utc'], files={str(p.relative_to(root)): sha(p) for p in [root / name for name in ['inputs.json','inference_config.json','runtime.json','fc_supplemental_review.json','review_annotations.json','RESULT.json','segments.csv','review.html']] + sorted((root / 'captions').glob('*.json'))}, report_code_sha256=sha(__file__)))
    state = read(root / 'STATUS.json')
    state.update(status=result['status'], inference_executed=True, completed_captions=78,
                 reviewed_segments=78, exact_pllava_pairs=65, pending=None,
                 updated_utc=result['created_utc'], skem_resumed=False, report='review.html', result='RESULT.json')
    write(root / 'STATUS.json', state)
    write(root / 'progress.json', dict(status=result['status'], completed=78, total=78,
                                     reviewed=78, updated_utc=result['created_utc']))
    print(json.dumps({k: result[k] for k in ['status','primary_matched_summary','full_78_summary','pllava_to_qwen_transitions','performance']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
