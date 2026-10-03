"""Verify paired extraction and export auditable captions without inventing scores."""
import argparse
import csv
import html
import json
import os
from pathlib import Path
import statistics

import run_qwen35_context as runner
from run_qwen35_context import base, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    plan = runner.check_identity(root)
    oldroot = Path(plan["inputs_file"]).parent
    frozen = read(plan["inputs_file"])
    rows = []
    for row, original in zip(plan["records"], frozen["records"]):
        if row["segment"] != original["segment"]:
            raise ValueError("Interval ordering changed")
        row = dict(row)
        for arm in runner.ARMS:
            cap = read(root / "captions" / arm / f"{row['segment']:03d}.json")
            if cap["signature"] != plan["signature"] or cap["arm"] != arm:
                raise ValueError("Caption identity mismatch")
            for key in ["segment", "start", "end_exclusive"]:
                if cap[key] != row[key]:
                    raise ValueError("Caption interval mismatch")
            expected_context = row["context_indices"] if arm == "context" else []
            expected_target = row["context_target_indices"] if arm == "context" else row["control_target_indices"]
            if cap["supplied_context_indices"] != expected_context or cap["supplied_target_indices"] != expected_target:
                raise ValueError("Caption image sampling mismatch")
            if not runner.policy(cap["caption"], cap["truncated"])["format_passed"]:
                raise ValueError("Caption output policy failed")
            if len(cap["supplied_indices"]) != row["image_budget"]:
                raise ValueError("Paired image budget mismatch")
            row[arm] = cap
        if row["context"]["image_grid_thw"] != row["target_only"]["image_grid_thw"]:
            raise ValueError("Paired visual-token budget differs")
        row["old_qwen"] = read(oldroot / "captions" / f"{row['segment']:03d}.json")["caption"]
        row["fc_lgvsc"] = original["fc_lgvsc_caption"]
        row["pllava"] = original["pllava_caption"]
        rows.append(row)
    performance = {}
    for arm in runner.ARMS:
        records = [r[arm] for r in rows]
        seconds = [sum(a["generation_seconds"] for a in r["attempts"]) for r in records]
        performance[arm] = dict(captions=len(records), mean_words=statistics.mean(r["policy"]["word_count"] for r in records),
                               max_words=max(r["policy"]["word_count"] for r in records),
                               generation_seconds=sum(seconds), mean_generation_seconds=statistics.mean(seconds),
                               retry_segments=sum(len(r["attempts"]) > 1 for r in records),
                               total_attempts=sum(len(r["attempts"]) for r in records),
                               caption_utf8_bytes=sum(len(r["caption"].encode()) for r in records),
                               peak_pytorch_allocated_gib=max(a["peak_allocated_bytes"] for r in records for a in r["attempts"])/2**30,
                               peak_device_used_mib=max(a["device_memory"]["sampled_device_peak_used_mib"] for r in records for a in r["attempts"]))
        bundle = dict(status="CAPTIONS_COMPLETE_VISUAL_ACCURACY_NOT_EXHAUSTIVELY_REVIEWED",
                      source=plan["source"], fps=plan["fps"], arm=arm, model_id=plan["model_id"],
                      model_revision=plan["revision"], quantization=plan["quantization"],
                      signature=plan["signature"], source_frames=plan["source_frames"],
                      hallucination_mitigation_verified=False, reconstruction_started=False,
                      records=[dict(segment=r["segment"], start=r["start"], end_exclusive=r["end_exclusive"],
                                    source_indices=r[arm]["supplied_target_indices"],
                                    context_indices=r[arm]["supplied_context_indices"], text=r[arm]["caption"])
                               for r in rows])
        base.write_json(root / f"{arm}_captions_bundle.json", bundle)
        (root / f"{arm}_captions.txt").write_text("\n\n".join(
            f"[{r['segment']:03d}] {r['start']/plan['fps']:.3f}–{r['end_exclusive']/plan['fps']:.3f} s\n{r[arm]['caption']}"
            for r in rows) + "\n")
    progress = read(root / "progress.json")
    result = dict(status="PAIRED_CAPTION_EXTRACTION_COMPLETE", created_utc=base.now(),
                  model_id=plan["model_id"], model_revision=plan["revision"], quantization=plan["quantization"],
                  source=plan["source"], intervals=len(rows), captions_total=2*len(rows),
                  covered_seconds=plan["covered_seconds"], fps=plan["fps"],
                  context_intervals=sum(bool(r["context_indices"]) for r in rows),
                  no_context_segments=[r["segment"] for r in rows if not r["context_indices"]],
                  scene_starts=plan["scene_starts"], context_seconds=plan["context_seconds"],
                  all_under_80_words=True, all_at_most_six_sentences=True, truncated=0,
                  original_target_samples_retained=True, paired_visual_token_budgets_equal=True,
                  performance=performance, final_session_wall_seconds=progress.get("current_session_seconds"),
                  runtimes=[read(p) for p in sorted((root / "runtimes").glob("*.json"))],
                  format_validation_is_not_accuracy_validation=True, independent_human_evaluation=False,
                  full_78_accuracy_evaluated=False, reconstructed_video_evaluated=False,
                  measured_channel_uses=None, limitations=[
                      "One development video; no independent or blinded human ground truth.",
                      "Prior FC/PLLaVA/Qwen captions have different prompts or frame counts; use the new paired control to study this context configuration.",
                      "Context replaces target samples at equal visual-token budget; this measures that allocation strategy, not an isolated extra-frame effect.",
                      "No detected scene cuts in this video; cut-reset efficacy on high-transition video remains untested.",
                      "UTF-8 caption bytes are payload-size diagnostics, not measured channel uses or total semantic-transmission cost.",
                      "No reconstruction was run, so no hallucination mitigation claim is supported."])
    review_path = root / "diagnostic_review.json"
    if review_path.exists():
        result["diagnostic_review"] = read(review_path)
    base.write_json(root / "RESULT.json", result)
    fields = ["segment", "start_seconds", "end_seconds", "context_indices", "context_target_indices", "control_target_indices",
              "context_caption", "target_only_caption", "old_qwen_caption", "fc_lgvsc_caption", "pllava_caption"]
    with (root / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(segment=row["segment"], start_seconds=row["start"]/plan["fps"],
                                 end_seconds=row["end_exclusive"]/plan["fps"],
                                 **{k:row[k] for k in fields[3:6]},
                                 context_caption=row["context"]["caption"], target_only_caption=row["target_only"]["caption"],
                                 old_qwen_caption=row["old_qwen"], fc_lgvsc_caption=row["fc_lgvsc"], pllava_caption=row["pllava"]))
    esc = html.escape
    parts = ['<!doctype html><html lang="ko"><meta charset="utf-8"><title>Qwen 문맥 캡션 비교</title>',
             '<style>body{font:16px/1.6 system-ui;max-width:1500px;margin:2rem auto;padding:0 1rem;color:#20242a}article{border-top:3px solid #aaa;padding-top:1rem;margin-top:2rem}.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:1.5rem}.frames{display:grid;grid-template-columns:repeat(4,1fr);gap:.5rem}img{width:100%}figure{margin:0}figcaption{font-size:13px}pre{white-space:pre-wrap}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:.5rem}.note{background:#fff6d8;padding:1rem}@media(max-width:850px){.grid{grid-template-columns:1fr}.frames{grid-template-columns:repeat(2,1fr)}}</style>',
             '<h1>Qwen3.5-9B: 직전 문맥 적용 캡션</h1>',
             f'<p>tv_low_08 · {len(rows)}개 동일 구간 · 각 조건 78개 · NF4 · 실제 원본 {plan["covered_seconds"]:.3f}초.</p>',
             '<p>직전 최대 2초의 원본 문맥 최대 4장과 현재 구간 샘플을 구분해 입력했습니다. 비교 조건은 같은 지시문과 같은 수의 현재 구간 이미지만 입력했습니다. 각 구간의 이미지 수·해상도·시각 토큰 예산이 같습니다.</p>',
             '<p class="note">전체 정확도 및 복원 영상의 개선은 검증하지 않았습니다. 단어 수·출력 완결성 검사는 사실 정확도 검사가 아닙니다. 이전 PLLaVA·FC·Qwen은 참고용이며 모델 외 조건이 다릅니다.</p>',
             '<table><tr><th>조건</th><th>평균 단어</th><th>최대 단어</th><th>평균 생성 시간 (재시도 포함)</th><th>재시도 구간</th></tr>']
    for arm, label in [("context", "문맥 적용"), ("target_only", "현재 구간만")]:
        p = performance[arm]
        parts.append(f'<tr><td>{label}</td><td>{p["mean_words"]:.1f}</td><td>{p["max_words"]}</td><td>{p["mean_generation_seconds"]:.2f}초</td><td>{p["retry_segments"]}</td></tr>')
    parts.append('</table><p><a href="context_captions.txt">문맥 캡션 TXT</a> · <a href="context_captions_bundle.json">문맥 캡션 JSON</a> · <a href="comparison.csv">전체 비교 CSV</a> · <a href="RESULT.json">실행 결과</a></p>')
    if review_path.exists():
        review = read(review_path)
        parts.append('<details><summary>8개 진단 구간 AI 검토 (전체 정확도가 아님)</summary><pre>'+esc(json.dumps(review, ensure_ascii=False, indent=2))+'</pre></details>')
    source = Path(plan["source_frames"])
    for row in rows:
        parts.append(f'<article id="s{row["segment"]}"><h2>구간 {row["segment"]:03d} · {row["start"]/plan["fps"]:.3f}–{row["end_exclusive"]/plan["fps"]:.3f}초</h2>')
        for role, indices in [("직전 문맥: 캡션 대상 아님", row["context_indices"]),
                              ("현재 구간: 문맥 적용 조건의 캡션 대상", row["context_target_indices"])]:
            parts.append(f'<details><summary>{role} ({len(indices)}장)</summary><div class="frames">')
            for index in indices:
                relative = os.path.relpath(source / f"{index}.png", root)
                parts.append(f'<figure><a href="{esc(relative)}"><img loading="lazy" src="{esc(relative)}"></a><figcaption>{index} · {index/plan["fps"]:.3f}초</figcaption></figure>')
            parts.append('</div></details>')
        parts.append('<div class="grid">')
        for arm, title in [("context", "문맥 적용"), ("target_only", "현재 구간만 — 동일 조건 비교")]:
            parts.append(f'<section><h3>{title}</h3><p>{esc(row[arm]["caption"])}</p><p>{row[arm]["policy"]["word_count"]} words</p></section>')
        parts.append('</div><details><summary>기존 Qwen · FC-LGVSC · PLLaVA 캡션</summary>')
        for key, title in [("old_qwen", "기존 Qwen 4프레임"), ("fc_lgvsc", "FC-LGVSC"), ("pllava", "PLLaVA 동일 구간")]:
            parts.append(f'<h3>{title}</h3><p>{esc(row[key] or "동일 구간 대응 없음")}</p>')
        parts.append('</details></article>')
    parts.append('</html>')
    (root / "review.html").write_text('\n'.join(parts))
    files = [p for p in root.rglob('*') if p.is_file() and p.name != 'artifact_manifest.json']
    base.write_json(root / "artifact_manifest.json", dict(created_utc=base.now(), files={str(p.relative_to(root)):base.sha256(p) for p in sorted(files)}))
    print(json.dumps({k:result[k] for k in ["status", "intervals", "context_intervals", "performance", "final_session_wall_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
