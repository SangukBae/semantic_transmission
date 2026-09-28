#!/usr/bin/env python3
"""Build time-labelled evidence for a single AI source reviewer.

Sampled AI observations are kept outside the frozen benchmark and never satisfy
its independent-human-review gate. Producing contact sheets is not reviewing them.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import html
import json
import math
from pathlib import Path

import cv2
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
BENCHMARK = REPO / 'data/etri_benchmark_v1_20260924'
OUTPUT = REPO / 'data/etri_ai_source_review_20260925'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def build_one(row):
    directory = OUTPUT / 'evidence' / row['id']
    directory.mkdir(parents=True, exist_ok=True)
    metadata = directory / 'sampling.json'
    if metadata.exists():
        cached = json.loads(metadata.read_text())
        if (cached['input_sha256'] == row['processed_sha256']
                and all(digest(OUTPUT / s['path']) == s['sha256'] for s in cached['sheets'])):
            return cached
        raise ValueError(f"Existing evidence differs: {row['id']}")
    if digest(row['processed_path']) != row['processed_sha256']:
        raise ValueError(f"Changed benchmark input: {row['id']}")
    count, fps = row['frames'], row['frame_rate']
    indices = sorted(set(range(0, count, round(2 * fps))) | {count - 1})
    selected = set(indices)
    capture = cv2.VideoCapture(row['processed_path'])
    frames, actual = {}, 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if actual in selected:
                small = cv2.resize(frame, (192, 107), interpolation=cv2.INTER_AREA)
                frames[actual] = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            actual += 1
    finally:
        capture.release()
    if actual != count or set(frames) != selected:
        raise ValueError(f'Incomplete decoded evidence: {row["id"]}: {actual}/{count}')
    sheets = []
    for start in range(0, len(indices), 36):
        group = indices[start:start + 36]
        page = Image.new('RGB', (192 * 6, 32 + 125 * math.ceil(len(group) / 6)), '#202020')
        draw = ImageDraw.Draw(page)
        title = f'{row["id"]} | {row["split"]} | {row["level"]} | page {start // 36 + 1} | source input, 2s sampling'
        draw.text((8, 8), title, fill='white')
        for n, index in enumerate(group):
            x, y = (n % 6) * 192, 32 + (n // 6) * 125
            page.paste(frames[index], (x, y))
            draw.text((x + 3, y + 108), f'{index / fps:06.2f}s  f{index}', fill='white')
        target = directory / f'page_{start // 36 + 1:02d}.jpg'
        page.save(target, quality=94)
        sheets.append({'path':str(target.relative_to(OUTPUT)), 'sha256':digest(target),
                       'frame_indices':group, 'times_sec':[round(i / fps, 6) for i in group]})
    result = {'id':row['id'], 'input_path':row['processed_path'],
              'input_sha256':row['processed_sha256'], 'source_sha256':row['source_sha256'],
              'duration_sec':row['clip_duration_sec'], 'fps':fps, 'total_frames':count,
              'decoded_frames':actual, 'sampled_frames':len(indices),
              'sampling_interval_sec':2, 'sample_fraction':len(indices) / count,
              'review_status':'EVIDENCE_PREPARED_NOT_REVIEWED', 'sheets':sheets}
    save(metadata, result)
    return result


def build(ids=None):
    rows = json.loads((BENCHMARK / 'manifest.json').read_text())
    if ids:
        rows = [r for r in rows if r['id'] in ids]
        if len(rows) != len(set(ids)):
            raise ValueError('Unknown or duplicate input IDs')
    cv2.setNumThreads(1)
    with ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(build_one, rows):
            print(result['id'], result['sampled_frames'], 'frames', flush=True)


def focus(ids):
    """Propose adjacent-frame pairs by pixel change; these are not cut labels."""
    rows = json.loads((BENCHMARK / 'manifest.json').read_text())
    for row in rows:
        if row['id'] not in ids:
            continue
        capture = cv2.VideoCapture(row['processed_path'])
        changes, previous, count, repeats = [], None, 0, 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                small = cv2.resize(frame, (144, 80), interpolation=cv2.INTER_AREA)
                if previous is not None:
                    delta = float(cv2.absdiff(small, previous).mean()) / 255
                    changes.append((delta, count))
                    if delta == 0 and count < row['frame_rate'] * 10:
                        repeats += 1
                previous = small
                count += 1
        finally:
            capture.release()
        chosen = []
        for delta, index in sorted(changes, reverse=True):
            if all(abs(index - other) >= 12 for _, other in chosen):
                chosen.append((delta, index))
            if len(chosen) == 10:
                break
        chosen.sort(key=lambda item:item[1])
        selected = {i for _, f in chosen for i in [f-1, f]}
        capture = cv2.VideoCapture(row['processed_path'])
        frames = {}
        try:
            for index in range(count):
                ok, frame = capture.read()
                if not ok:
                    raise ValueError('Focus decode failed')
                if index in selected:
                    frames[index] = Image.fromarray(cv2.cvtColor(cv2.resize(frame,(288,160)),cv2.COLOR_BGR2RGB))
        finally:
            capture.release()
        page = Image.new('RGB',(1152,32+180*math.ceil(len(chosen)/2)),'#202020')
        draw = ImageDraw.Draw(page)
        draw.text((8,8),f'{row["id"]}: adjacent frames at 10 pixel-change proposals (NOT cut truth)',fill='white')
        for k, (delta, index) in enumerate(chosen):
            for side, f in enumerate([index-1,index]):
                x, y = ((k % 2)*2+side)*288, 32+(k//2)*180
                page.paste(frames[f],(x,y))
                draw.text((x+3,y+162),f'{f/row["frame_rate"]:07.3f}s f{f} change={delta:.3f}',fill='white')
        path=OUTPUT/'focus'/f'{row["id"]}.jpg'
        path.parent.mkdir(exist_ok=True)
        page.save(path,quality=96)
        result={'id':row['id'],'path':str(path.relative_to(OUTPUT)),'sha256':digest(path),
                'input_sha256':row['processed_sha256'],'decoded_frames':count,
                'pairs':[{'before_frame':f-1,'after_frame':f,'time_sec':f/row['frame_rate'],'change_score':delta} for delta,f in chosen],
                'exact_small_frame_repeats_first_10s':repeats,
                'first_10s_comparisons':min(count-1,round(row['frame_rate']*10)-1),
                'status':'UNREVIEWED_PIXEL_CHANGE_PROPOSALS_NOT_SHOT_LABELS'}
        save(path.with_suffix('.json'),result)
        print(row['id'],'focus ready', 'first10s_exact_small_frame_repeats',repeats,flush=True)


def window(ident, start, end):
    rows = json.loads((BENCHMARK / 'manifest.json').read_text())
    row = next(r for r in rows if r['id'] == ident)
    if not 0 <= start < end <= row['clip_duration_sec'] or end - start > 8:
        raise ValueError('Window must cover at most 8 seconds within the input')
    fps = row['frame_rate']
    indices = list(range(round(start*fps), min(round(end*fps)+1, row['frames']), 6))
    capture = cv2.VideoCapture(row['processed_path'])
    page = Image.new('RGB',(1152,32+125*math.ceil(len(indices)/6)),'#202020')
    draw = ImageDraw.Draw(page)
    draw.text((8,8),f'{ident} | {start}-{end}s | every 6th frame, 4 fps',fill='white')
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES,indices[0])
        for index in range(indices[0],indices[-1]+1):
            ok, frame = capture.read()
            if not ok:
                raise ValueError('Incomplete window')
            if index in indices:
                k=indices.index(index)
                x,y=(k%6)*192,32+(k//6)*125
                page.paste(Image.fromarray(cv2.cvtColor(cv2.resize(frame,(192,107)),cv2.COLOR_BGR2RGB)),(x,y))
                draw.text((x+3,y+108),f'{index/fps:07.3f}s f{index}',fill='white')
    finally:
        capture.release()
    path=OUTPUT/'focus'/f'{ident}_{start:g}_{end:g}.jpg'
    path.parent.mkdir(exist_ok=True)
    page.save(path,quality=96)
    save(path.with_suffix('.json'),{'id':ident,'path':str(path.relative_to(OUTPUT)),
         'sha256':digest(path),'input_sha256':row['processed_sha256'],
         'frame_indices':indices,'status':'EVIDENCE_PREPARED_NOT_REVIEWED'})
    print(path)


def audit():
    rows = json.loads((BENCHMARK / 'manifest.json').read_text())
    expected = {r['id']:r for r in rows}
    reviews = json.loads((OUTPUT / 'reviews.json').read_text())
    errors, seen, frame_count, focus_views = [], set(), 0, 0
    for review in reviews:
        ident = review['id']
        if ident in seen or ident not in expected:
            errors.append(f'Duplicate or unknown review: {ident}')
            continue
        seen.add(ident)
        row = expected[ident]
        sampling = json.loads((OUTPUT / 'evidence' / ident / 'sampling.json').read_text())
        if review.get('reviewer_kind') != 'AI' or review.get('independent_ground_truth') is not False:
            errors.append(f'Incorrect AI review provenance: {ident}')
        if review.get('input_sha256') != row['processed_sha256'] or sampling['input_sha256'] != row['processed_sha256']:
            errors.append(f'Input hash mismatch: {ident}')
        expected_pages = [s['path'] for s in sampling['sheets']]
        if review.get('viewed_evidence') != expected_pages:
            errors.append(f'Unconfirmed visual evidence: {ident}')
        for sheet in sampling['sheets']:
            if digest(OUTPUT / sheet['path']) != sheet['sha256']:
                errors.append(f'Changed evidence: {sheet["path"]}')
        for path in review.get('viewed_focus_evidence', []):
            meta = json.loads((OUTPUT / path).with_suffix('.json').read_text())
            if (meta['id'] != ident or meta['input_sha256'] != row['processed_sha256']
                    or digest(OUTPUT / path) != meta['sha256']):
                errors.append(f'Invalid focus evidence: {path}')
            focus_views += 1
        for event in review.get('observations', []):
            if not 0 <= event['start_sec'] <= event['end_sec'] <= row['clip_duration_sec']:
                errors.append(f'Invalid event interval: {ident}')
        if not review.get('observations') or not review.get('limitations'):
            errors.append(f'Missing observations or limitations: {ident}')
        frame_count += sampling['sampled_frames']
    result = {'status':'PASS_AI_SAMPLED_REVIEW_PACKAGE' if not errors and seen == set(expected) else 'INCOMPLETE',
              'expected_inputs':len(rows), 'reviewed_inputs':len(seen), 'sampled_frames_reviewed':frame_count,
              'additional_focus_sheets_reviewed':focus_views,
              'missing_inputs':sorted(set(expected) - seen), 'errors':errors,
              'reviewer_count':1, 'reviewer_kind':'AI', 'independent_human_ground_truth_ready':False,
              'all_frames_visually_reviewed':False, 'reconstruction_error_evaluation_done':False,
              'scope':'Source-only sampled visual review; not continuous playback or event-complete truth.'}
    save(OUTPUT / 'audit.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


def export():
    """Package explicitly written observations, never infer them from model labels."""
    rows = json.loads((BENCHMARK / 'manifest.json').read_text())
    notes = json.loads((OUTPUT / 'notes.partial.json').read_text())
    by_id = {r['id']:r for r in rows}
    result = []
    for note in notes:
        row = by_id[note['id']]
        sampling = json.loads((OUTPUT / 'evidence' / row['id'] / 'sampling.json').read_text())
        result.append({'id':row['id'], 'split':row['split'], 'role':row['role'],
            'provisional_level':row['level'], 'input_sha256':row['processed_sha256'],
            'source_sha256':row['source_sha256'], 'duration_sec':row['clip_duration_sec'],
            'reviewer_id':'Codex_single_AI_session_20260925', 'reviewer_kind':'AI',
            'independent_ground_truth':False, 'blind_to_prior_curation':False,
            'reconstruction_outputs_viewed':False,
            'status':'AI_SAMPLED_VISUAL_REVIEW_COMPLETE',
            'viewed_evidence':[s['path'] for s in sampling['sheets']],
            'viewed_focus_evidence':note.get('focus', []),
            'sampling_interval_sec':2, 'sampled_frames':sampling['sampled_frames'],
            'all_frames_visually_reviewed':False,
            'observations':[{'start_sec':a,'end_sec':b,'description_ko':text,
                             'time_semantics':'coarse_observation_window_not_exact_event_boundary'}
                            for a,b,text in note['observations']],
            'source_flags':note['flags'], 'assessment_ko':note['assessment'],
            'limitations':['Single AI reviewer; no independent human adjudication.',
                '2-second still sampling plus last frame; selected focus evidence only. Brief events can be missed.',
                'Small objects and blur are unresolved when absent from visual evidence; no exhaustive instance tracking.',
                'No audio review, no reconstruction-error labels, no hallucination efficacy claim.']})
    save(OUTPUT/'reviews.json',result)
    audit()
    sections=[]
    for review in sorted(result,key=lambda r:r['id']):
        ident=html.escape(review['id'])
        events=''.join(f'<li>{e["start_sec"]}–{e["end_sec"]}초: {html.escape(e["description_ko"])}</li>' for e in review['observations'])
        evidence=''.join(f'<a href="{html.escape(p)}"><img loading="lazy" src="{html.escape(p)}" alt="{ident} evidence"></a>' for p in review['viewed_evidence']+review['viewed_focus_evidence'])
        sections.append(f'<details><summary>{ident} · {review["split"]} · {review["provisional_level"]} · {review["duration_sec"]}초</summary><p>{html.escape(review["assessment_ko"])}</p><ul>{events}</ul><p>태그: {html.escape(", ".join(review["source_flags"]))}</p>{evidence}</details>')
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ETRI 원본 AI 표본 검수</title><style>body{font:16px/1.65 sans-serif;max-width:1152px;margin:30px auto;padding:0 20px;background:#f5f6f8;color:#17212b}details{background:white;margin:12px 0;padding:16px;border:1px solid #ccd3dc;border-radius:8px}summary{cursor:pointer;font-weight:bold}img{max-width:100%;height:auto}a{color:#175da8}</style>
<h1>ETRI 원본 AI 표본 검수 · 2026-09-25</h1>
<p>66개 입력의 2초 간격 표본과 마지막 프레임을 AI 한 명이 확인한 기록입니다. 일부 의심 구간은 인접 프레임 또는 4 fps로 추가 확인했습니다. 전 프레임 시각 검수·독립 사람 정답·복원 오류 평가를 뜻하지 않습니다.</p>
<p>시간은 관찰 구간이며 사건의 정확한 시작·종료 정답이 아닙니다. 검수자는 기존 분류와 개요를 볼 수 있었습니다. 기존 입력 동결본과 독립 검수 상태는 보존됩니다.</p>
<p><a href="reviews.json">전체 주석 JSON</a> · <a href="audit.json">구조·증거 검사</a> · <a href="REPORT.md">주요 발견과 남은 범위</a></p>'''
    (OUTPUT/'index.html').write_text(page+'\n'.join(sections)+'</html>\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['build', 'focus', 'window', 'export', 'audit'])
    parser.add_argument('--ids', nargs='+')
    parser.add_argument('--start', type=float)
    parser.add_argument('--end', type=float)
    args = parser.parse_args()
    if args.command == 'build':
        build(args.ids)
    elif args.command == 'focus':
        if not args.ids:
            parser.error('focus requires explicit --ids')
        focus(args.ids)
    elif args.command == 'window':
        if not args.ids or len(args.ids) != 1 or args.start is None or args.end is None:
            parser.error('window requires one --ids entry, --start and --end')
        window(args.ids[0], args.start, args.end)
    elif args.command == 'export':
        export()
    else:
        audit()
