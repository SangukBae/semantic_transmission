"""Aggregate source-reviewed annotations; this is not an automatic caption judge."""
import argparse
import csv
import html
import json
import os
from pathlib import Path

import run_qwen35_context as runner
from run_qwen35_context import base, read


def summary(rows, arm):
    n = len(rows)
    errors = [r["segment"] for r in rows if r["review"][arm]["errors"]]
    omissions = [r["segment"] for r in rows if r["review"][arm]["omission"]]
    return dict(reviewed_intervals=n, clear_error_intervals=len(errors),
                clear_error_interval_ids=errors, omission_intervals=len(omissions),
                omission_interval_ids=omissions,
                intervals_with_either=len(set(errors) | set(omissions)))


def transitions(rows, before, after, field):
    groups = {"resolved": [], "remaining": [], "new": [], "neither": []}
    for row in rows:
        a, b = bool(row["review"][before][field]), bool(row["review"][after][field])
        key = "remaining" if a and b else "resolved" if a else "new" if b else "neither"
        groups[key].append(row["segment"])
    before_count = len(groups["resolved"])+len(groups["remaining"])
    reduction = len(groups["resolved"])-len(groups["new"])
    return dict(interval_ids=groups, counts={k:len(v) for k,v in groups.items()},
                net_reduction_intervals=reduction,
                relative_reduction_percent=100*reduction/before_count if before_count else None)


def old_annotation(value):
    return dict(errors=value["clear_factual_errors"], omission=value["visible_entity_omission"],
                uncertain=value.get("unverified_details", []), provenance="Frozen previous AI source review")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--inference-root", type=Path, required=True)
    args = parser.parse_args()
    root, source_root = args.root.resolve(), args.inference_root.resolve()
    protocol = read(root / "protocol.json")
    plan = runner.check_identity(source_root)
    assert protocol["source_plan_sha256"] == base.sha256(source_root / "plan.json")
    baseline_root = Path(plan["inputs_file"]).parent
    old = {r["segment"]:r for r in read(baseline_root / "review_annotations.json")["annotations"]}
    frozen_inputs = {r["segment"]:r for r in read(plan["inputs_file"])["records"]}
    annotations = []
    for name in ["annotations_00_23.json", "annotations_24_47.json", "annotations_48_77.json"]:
        annotations.extend(read(root / name))
    assert [a["segment"] for a in annotations] == list(range(78))
    rows = []
    input_hashes = {str(source_root / "plan.json"):base.sha256(source_root / "plan.json"),
                    str(baseline_root / "review_annotations.json"):base.sha256(baseline_root / "review_annotations.json"),
                    str(baseline_root / "RESULT.json"):base.sha256(baseline_root / "RESULT.json")}
    for ann, sample in zip(annotations, plan["records"]):
        i = ann["segment"]
        assert i == sample["segment"]
        captions = {}
        reviews = {arm:ann[arm] for arm in runner.ARMS}
        for arm in runner.ARMS:
            path = source_root / "captions" / arm / f"{i:03d}.json"
            record = read(path)
            input_hashes[str(path)] = base.sha256(path)
            assert record["signature"] == plan["signature"]
            assert record["start"] == sample["start"] and record["end_exclusive"] == sample["end_exclusive"]
            captions[arm] = record["caption"]
            for error in reviews[arm]["errors"]:
                assert error["claim"].lower() in record["caption"].lower(), (i, arm, error)
        previous_path = baseline_root / "captions" / f"{i:03d}.json"
        captions["old_qwen"] = read(previous_path)["caption"]
        input_hashes[str(previous_path)] = base.sha256(previous_path)
        captions["fc_lgvsc"] = frozen_inputs[i]["fc_lgvsc_caption"]
        captions["pllava"] = frozen_inputs[i]["pllava_caption"]
        reviews["old_qwen"] = old_annotation(old[i]["qwen"])
        reviews["fc_lgvsc"] = old_annotation(old[i]["fc_lgvsc"])
        if sample["matched_pair"] is not None:
            reviews["pllava"] = old_annotation(old[i]["pllava"])
        rows.append(dict(segment=i, matched_pair=sample["matched_pair"],
                         start_seconds=sample["start"]/plan["fps"],
                         end_seconds=sample["end_exclusive"]/plan["fps"],
                         source_indices=sample["original_indices"],
                         direct_source_review=True, evidence_sheet=f"sheets/{i//6:02d}.jpg",
                         source_observation_from_previous_review=old[i]["source_observation_ko"],
                         captions=captions, review=reviews))
    matched = [r for r in rows if r["matched_pair"] is not None]
    assert len(matched) == 65
    full = {arm:summary(rows, arm) for arm in ["old_qwen", "target_only", "context", "fc_lgvsc"]}
    common = {arm:summary(matched, arm) for arm in ["pllava", "old_qwen", "target_only", "context", "fc_lgvsc"]}
    previous = read(baseline_root / "RESULT.json")
    assert full["old_qwen"]["clear_error_intervals"] == previous["full_78_summary"]["qwen"]["clear_error_segments"]
    assert common["pllava"]["clear_error_intervals"] == previous["primary_matched_summary"]["pllava"]["clear_error_segments"]
    changes = {}
    for before, after, subset in [("target_only", "context", rows), ("old_qwen", "context", rows),
                                   ("pllava", "context", matched)]:
        changes[f"{before}_to_{after}"] = {name:transitions(subset,before,after,field)
                                           for name,field in [("clear_error", "errors"), ("omission", "omission")]}
    limitations = [
        "AI review of original source samples, not independent or blinded human ground truth; counts are discovered contradictory-caption intervals, not a calibrated model accuracy rate.",
        "All 78 intervals reviewed from sampled source images; not exhaustive annotation of all 1440 native frames or every possible unsupported claim.",
        "Adjacent intervals are correlated; a repeated mistake can count in multiple intervals. This does not count independent events or atomic errors.",
        "Uncertain claims are excluded from clear errors. Errors and omissions overlap; do not add their counts.",
        "The paired new arms hold model, prompt policy and visual-token budget fixed; context replaces some target samples. This evaluates the allocation strategy.",
        "Older Qwen uses four frames and an older prompt. PLLaVA and revised FC captions have different workflows. Their frozen prior annotations are references, not controlled method-only comparisons.",
        "New review includes additional endpoints to inspect temporal change; frozen older reviews used four samples. Differences in visual review coverage may affect historical comparisons.",
        "One development video, no reconstructed-video evaluation, independent human evaluation, or scene-cut validation.",
    ]
    result = dict(status="ALL_78_INTERVALS_PAIRED_AI_SOURCE_SAMPLE_REVIEW_COMPLETE", created_utc=base.now(),
                  source="tv_low_08", full_intervals=78, matched_pllava_intervals=65,
                  inference_root=str(source_root), independent=False, blinded=False,
                  unit="Intervals containing at least one clear factual error; omissions separately",
                  full_78=full, matched_65=common, transitions=changes, limitations=limitations,
                  reconstructed_video_evaluated=False, annotations_file="review_annotations.json")
    base.write_json(root / "review_annotations.json", dict(protocol=protocol, annotations=rows))
    base.write_json(root / "input_manifest.json", dict(files=input_hashes))
    base.write_json(root / "RESULT.json", result)
    with (root / "comparison.csv").open("w", newline="") as f:
        fields = ["segment", "matched_pair", "start_seconds", "end_seconds"]
        fields += [f"{a}_{k}" for a in ["old_qwen", "target_only", "context", "fc_lgvsc", "pllava"]
                   for k in ["caption", "clear_error", "omission", "error_reasons"]]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            data = {k:r[k] for k in fields[:4]}
            for arm in ["old_qwen", "target_only", "context", "fc_lgvsc", "pllava"]:
                a = r["review"].get(arm)
                data.update({f"{arm}_caption":r["captions"][arm],
                             f"{arm}_clear_error":bool(a["errors"]) if a else "",
                             f"{arm}_omission":a["omission"] if a else "",
                             f"{arm}_error_reasons":json.dumps(a["errors"],ensure_ascii=False) if a else ""})
            writer.writerow(data)
    esc = html.escape
    labels = {"old_qwen":"기존 Qwen: 4프레임·이전 지시문", "target_only":"새 Qwen: 문맥 없음",
              "context":"새 Qwen: 문맥 적용", "fc_lgvsc":"기존 FC-LGVSC", "pllava":"기존 PLLaVA"}
    parts = ['<!doctype html><html lang="ko"><meta charset="utf-8"><title>문맥 캡션 78구간 오류 검토</title>',
             '<style>body{font:16px/1.6 system-ui;margin:2rem auto;max-width:1450px;padding:0 1rem;color:#20242a}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:.6rem}article{border-top:3px solid #aaa;margin-top:2rem;padding-top:1rem}.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:1.5rem}.images{display:grid;grid-template-columns:repeat(4,1fr);gap:.5rem}img{width:100%}.bad{color:#a41111}.note{background:#fff3d2;padding:1rem}pre{white-space:pre-wrap}@media(max-width:800px){.grid{grid-template-columns:1fr}.images{grid-template-columns:repeat(2,1fr)}}</style>',
             '<h1>문맥 캡션: 전체 78구간 AI 원본 샘플 검토</h1>',
             '<p class="note">명백한 사실 오류를 하나 이상 포함한 구간 수입니다. 독립적인 오류 사건 수, 개별 주장 수, 모델 정확도는 아닙니다. 원본 표본을 본 AI 판정이며 독립·블라인드 사람 평가는 아닙니다. 누락은 별도 집계하고 오류와 합산하지 않습니다.</p>']
    for title, stats in [("전체 동일 78구간", full), ("PLLaVA까지 정확히 대응하는 65구간", common)]:
        parts.append(f'<h2>{title}</h2><table><tr><th>조건</th><th>명백 오류 포함</th><th>사람·개·차량 누락 포함</th></tr>')
        for arm,v in stats.items():
            parts.append(f'<tr><td>{labels[arm]}</td><td>{v["clear_error_intervals"]}/{v["reviewed_intervals"]}</td><td>{v["omission_intervals"]}/{v["reviewed_intervals"]}</td></tr>')
        parts.append('</table>')
    c=changes['target_only_to_context']['clear_error']
    parts.append(f'<p>같은 새 조건에서 문맥 적용: 오류 포함 구간 {full["target_only"]["clear_error_intervals"]} → {full["context"]["clear_error_intervals"]}. 기존 오류 구간 중 {c["counts"]["resolved"]}개에서 오류가 없어졌고 {c["counts"]["new"]}개에서는 새 오류가 생겨 순감소는 {c["net_reduction_intervals"]}개입니다. 오류 유형이 바뀌어도 오류가 남으면 해결로 세지 않습니다.</p>')
    parts.append('<p>기존 Qwen/PLLaVA/FC는 이전 동결 판정을 재사용한 참고 비교입니다. 새 Qwen 두 조건과 프롬프트·프레임 수·검토 표본 범위가 다르므로 문맥만의 인과 효과로 해석하지 않습니다.</p>')
    parts.append('<p><a href="comparison.csv">전체 CSV</a> · <a href="RESULT.json">집계 JSON</a> · <a href="review_annotations.json">구간별 판정 JSON</a></p>')
    parts.append('<details><summary>검토 범위와 한계</summary><ul>'+''.join('<li>'+esc(s)+'</li>' for s in limitations)+'</ul></details>')
    for r in rows:
        i=r['segment']
        parts.append(f'<article id="s{i}"><h2>구간 {i:03d} · {r["start_seconds"]:.3f}–{r["end_seconds"]:.3f}초</h2>')
        parts.append(f'<p><a href="{r["evidence_sheet"]}">원본 앞뒤 표본 포함 비교 시트</a></p><details><summary>기존 공통 원본 4개 샘플</summary><div class="images">')
        for index in sorted(set(r['source_indices'])):
            path=os.path.relpath(Path(plan['source_frames'])/f'{index}.png',root)
            parts.append(f'<a href="{esc(path)}"><img loading="lazy" src="{esc(path)}" alt="frame {index}"></a>')
        parts.append('</div></details><div class="grid">')
        for arm in ['target_only','context']:
            a=r['review'][arm]
            verdict='명백 오류 포함' if a['errors'] else '명백 오류 발견 없음'
            parts.append(f'<section><h3>{labels[arm]} · {verdict}</h3><p>{esc(r["captions"][arm])}</p>')
            for e in a['errors']:
                parts.append(f'<p class="bad"><b>{esc(e["claim"])}</b><br>{esc(e["reason_ko"])}</p>')
            if a['omission']:parts.append('<p class="bad">누락: '+esc(a['omission'])+'</p>')
            if a['uncertain']:parts.append('<details><summary>확정 오류에서 제외한 불확실한 사항</summary><ul>'+''.join('<li>'+esc(t)+'</li>' for t in a['uncertain'])+'</ul></details>')
            parts.append('</section>')
        parts.append('</div></article>')
    parts.append('</html>')
    (root/'review.html').write_text('\n'.join(parts))
    base.write_json(root/'artifact_manifest.json',dict(created_utc=base.now(),report_code_sha256=base.sha256(__file__),
                    files={str(p.relative_to(root)):base.sha256(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json'}))
    print(json.dumps(dict(full_78=full,matched_65=common,paired=changes['target_only_to_context'],
                          historical=changes['old_qwen_to_context']),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
