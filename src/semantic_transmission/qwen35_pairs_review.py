"""Automatic, explicitly non-independent diagnostics for the paired experiment."""
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from . import auto_extraction as ae
from .artifacts import sha256, write_json
from .qwen35_pairs_comparison import ARMS, LABELS, infer, load_protocol, validate_bundle

REVIEW = '''Compare the anonymous candidate with SOURCE frames at the same times.
SOURCE images precede RECONSTRUCTION images; labels are metadata, not scene text.
The candidate captions cover labeled intervals; do not treat a claim about a
different part of an overlapping caption interval as a contradiction here.
The reference facts are fallible AI annotations: use unobservable if the supplied
source samples cannot establish a fact. A fact is reproduced only when the
caption explicitly preserves its visible action, direction and relationship.
Inspect RECONSTRUCTION against SOURCE for Added (unsupported object/event),
Missing (visible source object/event absent), and Distorted (wrong appearance,
position, action, direction or relation). Do not count compression blur alone as
a semantic error. Do not infer events between samples. Only report clear issues;
put ambiguity in uncertainties. This is a diagnostic, not human ground truth.
Return ONLY JSON with exactly these fields:
{"caption_facts":[{"id":"given fact ID","status":"reproduced|missing|unobservable"}],
 "caption_issues":[{"type":"Added|Distorted","claim":"exact candidate caption quote",
 "evidence_frames":[0],"reason":"short source-grounded explanation"}],
 "reconstruction_issues":[{"type":"Added|Missing|Distorted","claim":"short visible difference",
 "evidence_frames":[0],"reason":"short source versus reconstruction explanation"}],
 "uncertainties":["short unresolved ambiguity"]}.
Include every reference fact exactly once. Evidence IDs must be supplied image
frame IDs. At most 4 issues per list. Empty issue lists are valid. Be concise.
INPUT:
'''


def validate_judgment(value, facts, ids, captions):
    if set(value) != {"caption_facts", "caption_issues", "reconstruction_issues", "uncertainties"}:
        raise ValueError("semantic review schema differs")
    expected = [f["id"] for f in facts]
    got = value["caption_facts"]
    if not isinstance(got, list) or sorted(r["id"] for r in got) != sorted(expected):
        raise ValueError("missing/duplicate/unknown reference fact")
    if any(r["status"] not in {"reproduced", "missing", "unobservable"} for r in got):
        raise ValueError("invalid fact status")
    for category in ("caption_issues", "reconstruction_issues"):
        issues = value[category]
        if not isinstance(issues, list) or len(issues) > 4:
            raise ValueError("too many semantic issues")
        for issue in issues:
            kinds = {"Added", "Distorted"} if category == "caption_issues" else {"Added", "Missing", "Distorted"}
            if (issue["type"] not in kinds or not isinstance(issue["claim"], str) or not issue["claim"].strip()
                    or not isinstance(issue["reason"], str) or not issue["reason"].strip()
                    or not isinstance(issue["evidence_frames"], list) or not issue["evidence_frames"]
                    or any(type(i) is not int or i not in ids for i in issue["evidence_frames"])):
                raise ValueError("invalid semantic issue/evidence")
            if category == "caption_issues" and not any(issue["claim"] in r["caption"] for r in captions):
                raise ValueError("semantic issue quote is not in the candidate caption")
    ae.validate_strings(value, "uncertainties")
    return value


def reconstructed_video(root, arm, p):
    run = root / f"reconstruction/{arm}/run"
    audit, quality = (ae.read(run / name) for name in ("output_audit.json", "quality.json"))
    video = Path(audit["video"])
    if (audit["status"] != "PASSED" or quality["status"] != "PASSED"
            or sha256(video) != audit["video_sha256"] or quality["video_sha256"] != audit["video_sha256"]
            or quality["reference_sha256"] != p["source"]["normalized_sha256"]
            or quality["output_source_indices"] != list(range(p["source"]["frames"]))):
        raise ValueError("reconstruction audit/quality/timeline differs")
    return video


def decoded_frames(root, arm, p):
    video = reconstructed_video(root, arm, p)
    folder = root / f"semantic/frames/{arm}"
    receipt = folder / "manifest.json"
    if receipt.exists():
        saved = ae.read(receipt)
        if saved["video_sha256"] != sha256(video) or len(saved["frames"]) != p["source"]["frames"]:
            raise ValueError("review video/frame manifest changed")
        for name, digest in saved["frames"].items():
            if sha256(folder / name) != digest:
                raise ValueError("cached review frame changed")
        return folder
    if folder.exists():
        archive = root / f"failed_attempts/review_frames_{arm}_{time.time_ns()}"
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(folder), str(archive))
    folder.mkdir(parents=True)
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(video), "-vsync", "0",
                    "-start_number", "0", str(folder / "%05d.png")], check=True)
    frames = sorted(folder.glob("*.png"))
    if [f.name for f in frames] != [f"{i:05d}.png" for i in range(p["source"]["frames"])]:
        raise ValueError("decoded reconstruction timeline differs")
    write_json(receipt, dict(video_sha256=sha256(video), frames={f.name: sha256(f) for f in frames}))
    return folder


def summarize(rows):
    result = {}
    for arm in ARMS:
        selected = [r["judgment"] for r in rows if r["arm"] == arm]
        result[arm] = dict(bins=len(selected), facts_reproduced=sum(
            f["status"] == "reproduced" for r in selected for f in r["caption_facts"]),
            facts_missing=sum(f["status"] == "missing" for r in selected for f in r["caption_facts"]),
            facts_unobservable=sum(f["status"] == "unobservable" for r in selected for f in r["caption_facts"]),
            caption_error_bins=sum(bool(r["caption_issues"]) for r in selected),
            reconstruction_error_bins=sum(bool(r["reconstruction_issues"]) for r in selected),
            reconstruction_issue_counts={kind: sum(i["type"] == kind for r in selected
                for i in r["reconstruction_issues"]) for kind in ("Added", "Missing", "Distorted")})
    return result


def semantic_inputs(root, p):
    return dict(protocol=p["signature"], reference_sha256=sha256(Path(p["reference"])),
        videos={arm: sha256(reconstructed_video(root, arm, p)) for arm in ARMS},
        captions={arm: sha256(root / f"captions/{arm}.json") for arm in ARMS})


def review(root):
    from PIL import Image
    from .qwen35_backend import LocalQwen35
    p = load_protocol(root)
    bundles = {arm: validate_bundle(root, arm, p) for arm in ARMS}
    folders = {arm: decoded_frames(root, arm, p) for arm in ARMS}
    reference = ae.read(Path(p["reference"]))
    if reference["frames"] != p["source"]["frames"] or reference["fps"] != 24:
        raise ValueError("reference timeline differs")
    vlm = LocalQwen35(p["caption"])
    rows = []
    for item in reference["bins"]:
        a, b = item["start"], item["end_exclusive"]
        ids = sorted(set([a, a+(b-a)//3, a+2*(b-a)//3, b-1]))
        for arm in (ARMS if item["id"] % 2 == 0 else ARMS[::-1]):
            captions = [r for r in bundles[arm]["captions"] if r["start"] < b and r["end_exclusive"] > a]
            inputs = dict(interval=[a, b], reference_facts=item["facts"],
                candidate_captions=[{k: r[k] for k in ("start", "end_exclusive", "caption")} for r in captions])
            paths = [ae.check_frame(p["source"], i) for i in ids] + [folders[arm] / f"{i:05d}.png" for i in ids]
            labels = [f"{kind} frame {i}, time {i/24:.6f}s" for kind in ("SOURCE", "RECONSTRUCTION") for i in ids]
            identity = dict(protocol=p["signature"], arm=arm, bin=item["id"], task="visual_review",
                frame_hashes=[sha256(path) for path in paths], reference_sha256=sha256(Path(p["reference"])),
                captions_sha256=sha256(root / f"captions/{arm}.json"))
            images = [Image.open(path).convert("RGB") for path in paths]
            try:
                judgment = infer(vlm, root / f"semantic/calls/{arm}/{item['id']:02d}.json", identity,
                    images, labels, REVIEW + json.dumps(inputs, ensure_ascii=False),
                    lambda obj: validate_judgment(obj, item["facts"], ids, captions), tokens=768)
            finally:
                for im in images:
                    im.close()
            rows.append(dict(arm=arm, bin=item["id"], source_indices=ids, judgment=judgment))
            print(f"SEMANTIC_DIAGNOSTIC {arm} {item['id']+1}/{len(reference['bins'])}", flush=True)
    write_json(root / "semantic/RESULT.json", dict(status="COMPLETE_NONINDEPENDENT_QWEN_DIAGNOSTIC",
        inputs=semantic_inputs(root, p),
        protocol=p["signature"], evaluator="Qwen3.5-9B NF4, base model, caption adapter disabled",
        independent=False, human_ground_truth=False, independent_semantic_review="PENDING",
        source_samples_per_second=4, historical_astra_scores_not_comparable=True,
        reference_sha256=sha256(Path(p["reference"])), metrics=summarize(rows), rows=rows,
        limitation="Same model family as captioning; sparse visual samples and fallible existing AI facts. "
                   "Not proof of independent hallucination mitigation or unseen-video generalization."))


def check_common_channel(root, p):
    import numpy as np
    from .wire import unpack
    streams = {}
    for arm in ARMS:
        run = root / f"reconstruction/{arm}/run"
        header, payload = unpack((run / "received/metadata.bin").read_bytes())
        streams[arm] = (header, payload, np.fromfile(run / "transmitter/visual.c64", dtype="<c8"),
                        np.fromfile(run / "received/visual.c64", dtype="<c8"))
    common = sorted(set(p["keys"][ARMS[0]]) & set(p["keys"][ARMS[1]]))
    for i in common:
        records = []
        for arm in ARMS:
            header, payload, sent, received = streams[arm]
            item = next(k for k in header["keyframes"] if k["index"] == i)
            a, n, r = item["complex_offset"], item["complex_count"], item["rate_offset"]
            records.append((sent[a:a+n], received[a:a+n], payload[r:r+item["rate_bytes"]], item["average_power"]))
        if any(not np.array_equal(records[0][j], records[1][j]) for j in (0, 1)) or records[0][2:] != records[1][2:]:
            raise ValueError(f"common transmitted/received frame differs: {i}")
    return dict(status="PASS", common_frames=common, same_sent_and_received_symbols=True,
                law="unit-power complex AWGN, PCG64 seed per source frame")


def comparison_video(root, p):
    inputs = [Path(p["source"]["normalized_video"]), *[reconstructed_video(root, arm, p) for arm in ARMS]]
    target, receipt = root / "comparison.mp4", root / "comparison_video.json"
    hashes = [sha256(v) for v in inputs]
    if receipt.exists():
        old = ae.read(receipt)
        if old["input_sha256"] != hashes or old["sha256"] != sha256(target):
            raise ValueError("comparison video changed")
        return
    temporary = root / "comparison.partial.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-nostdin",
        *[a for v in inputs for a in ("-i", str(v))], "-filter_complex", "[0:v][1:v][2:v]hstack=inputs=3[v]",
        "-map", "[v]", "-an", "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p", str(temporary)], check=True)
    os.replace(temporary, target)
    write_json(receipt, dict(input_sha256=hashes, sha256=sha256(target), order=["source", *ARMS]))


def report(root, p):
    from .qwen35_feedback_eval import event_coverage
    reference = ae.read(Path(p["reference"]))
    metrics = {}
    ready = all((root / f"reconstruction/{arm}/run/quality.json").exists() for arm in ARMS)
    for arm in ARMS:
        bundle = validate_bundle(root, arm, p)
        calls = [ae.unseal(f) for role in ("draft", "review") for f in (root / f"calls/{arm}/{role}").glob("*.json")]
        metrics[arm] = dict(label=LABELS[arm], keys=len(p["keys"][arm]), captions=len(bundle["captions"]),
            event_coverage=event_coverage(p["keys"][arm], reference),
            successful_caption_generation_seconds=sum(c["seconds"] for c in calls),
            caption_failed_attempts=len(list((root / f"calls/{arm}").glob("*/failed_attempts/*.json"))),
            timing_scope="Cached and new successful draft/review calls; excludes load, failures, historical selection.")
        if ready:
            video = reconstructed_video(root, arm, p)
            run = root / f"reconstruction/{arm}/run"
            channel = ae.read(run / "channel_accounting.json")
            metrics[arm].update(video=str(video), quality=ae.read(run / "quality.json"),
                channel=channel, resources={f.stem: ae.read(f) for f in (run.parent / "resources").glob("*.json")})
    paired_channel = check_common_channel(root, p) if ready else None
    if ready:
        comparison_video(root, p)
    semantic_path = root / "semantic/RESULT.json"
    semantic = ae.read(semantic_path) if semantic_path.exists() else None
    if semantic is not None and (semantic["protocol"] != p["signature"] or not ready
            or semantic["inputs"] != semantic_inputs(root, p)):
        raise ValueError("semantic diagnostic protocol/video/captions differ")
    result = dict(status="COMPLETE_AUTO_DIAGNOSTIC_INDEPENDENT_REVIEW_PENDING" if semantic else
        "RECONSTRUCTION_COMPLETE_REVIEW_PENDING" if ready else "CAPTIONS_COMPLETE_RECONSTRUCTION_PENDING",
        protocol=p["signature"], metrics=metrics, common_channel_audit=paired_channel,
        semantic_diagnostic=semantic["metrics"] if semantic else None, independent_semantic_review="PENDING",
        hallucination_mitigation_verified=False, training_executed=False, scope=p["scope"], noise_scope=p["noise_scope"])
    write_json(root / "RESULT.json", result)
    esc = html.escape
    lines = ['<!doctype html><html lang="ko"><meta charset="utf-8"><title>79키 선택 + 보강 캡션 비교</title>',
        '<style>body{font:16px system-ui;margin:30px;max-width:1500px}table{border-collapse:collapse;width:100%}'
        'td,th{border:1px solid #bbb;padding:10px}video{width:100%}img{max-width:24%}pre{white-space:pre-wrap}</style>',
        '<h1>기존 InternVL 선택 / Qwen 사건 쌍 선택 + 같은 보강 캡션</h1>',
        '<p>알려진 TVSum 60초 개발 영상. 두 조건 모두 79키·78캡션, 같은 LoRA·지시문·4장 표본 규칙·수신단.</p>',
        '<p>과거 선택 결과를 재사용합니다. 선택 모델만의 통제 실험이나 새 영상 일반화 검증은 아닙니다.</p>',
        '<p><strong>의미 판정은 로컬 Qwen의 비독립 자동 진단입니다. 독립 검수는 PENDING입니다.</strong></p>',
        '<p><a href="RESULT.json">전체 결과 JSON</a> · <a href="protocol.json">동결 설정·출처</a></p>']
    if ready:
        lines += ['<h2>왼쪽: 원본 / 가운데: InternVL 선택 / 오른쪽: 사건 쌍 선택</h2>',
                  '<video controls preload="metadata" src="comparison.mp4"></video>']
    lines.append('<table><tr><th>조건</th><th>사건 포착</th><th>캡션 생성·재검토</th><th>채널 사용량</th><th>MP4 화질</th></tr>')
    for arm, m in metrics.items():
        ev = m["event_coverage"]
        quality = m.get("quality", {}).get("delivered_mp4", {})
        score = ", ".join(f"{k}={v:.4f}" for k, v in quality.items() if k in {"psnr_db", "ssim", "lpips_vgg", "dists", "clip"})
        lines.append(f'<tr><td>{esc(LABELS[arm])}</td><td>{ev["covered"]}/{ev["total"]}</td>'
            f'<td>{m["successful_caption_generation_seconds"]/60:.1f}분</td>'
            f'<td>{m.get("channel", {}).get("total_complex_channel_uses", "대기")}</td><td>{esc(score or "대기")}</td></tr>')
    lines.append('</table><h2>같은 캡션 모델의 실제 출력</h2><table><tr><th>조건·시간</th><th>원본 표본 4장</th><th>캡션</th></tr>')
    for arm in ARMS:
        for row in validate_bundle(root, arm, p)["captions"]:
            images = ''.join(f'<a href="{esc(os.path.relpath(Path(p["source"]["source_frames"])/f"{i}.png",root))}">'
                f'<img loading="lazy" src="{esc(os.path.relpath(Path(p["source"]["source_frames"])/f"{i}.png",root))}" title="{i}"></a>'
                for i in row["source_indices"])
            lines.append(f'<tr><td>{esc(arm)} {row["start"]/24:.2f}–{row["end_exclusive"]/24:.2f}s</td>'
                         f'<td>{images}</td><td>{esc(row["caption"])}</td></tr>')
    lines.append('</table>')
    if semantic:
        lines += ['<h2>비독립 Qwen 자동 의미 진단</h2>', '<pre>'+esc(json.dumps(semantic["metrics"], ensure_ascii=False, indent=2))+'</pre>',
                  '<p><a href="semantic/RESULT.json">모든 사실 판정·오류·근거 프레임</a></p>']
        for row in semantic["rows"]:
            if not row["judgment"]["caption_issues"] and not row["judgment"]["reconstruction_issues"]:
                continue
            lines.append(f'<details><summary>{esc(row["arm"])} / {row["bin"]}초</summary>')
            for label, paths in (("원본", [Path(p["source"]["source_frames"])/f"{i}.png" for i in row["source_indices"]]),
                                 ("복원", [root/f"semantic/frames/{row['arm']}/{i:05d}.png" for i in row["source_indices"]])):
                lines.append('<p>'+label+'</p>'+''.join(f'<img loading="lazy" src="{esc(os.path.relpath(path,root))}">' for path in paths))
            lines.append('<pre>'+esc(json.dumps(row["judgment"], ensure_ascii=False, indent=2))+'</pre></details>')
    (root / "report.html").write_text('\n'.join(lines)+'</html>')
