"""Run the frozen, diverse five-video WebVid development pilot sequentially."""
import argparse
import csv
import datetime
import fcntl
import hashlib
import html
import json
import math
from pathlib import Path
import uuid

from .artifacts import runtime_state, sha256, write_json
from .cli import repository, settings
from .input_contract import video_config
from .research import main as research
from .resume import completed_runs
from .video_io import probe

CATEGORIES = {"low_motion", "single_subject", "fast_motion", "camera_motion", "multi_object_occlusion"}
METRICS = ("psnr_db", "ssim", "lpips_vgg", "clip", "dists")
BOUNDARIES = ("lossless_frames", "delivered_mp4")


def read_json(path):
    return json.loads(Path(path).read_text())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def load_inputs(repo):
    cohort = read_json(repo / "configs/webvid5_manifest.json")
    videos = cohort["videos"]
    if (len(videos) != 5 or {v["category"] for v in videos} != CATEGORIES
            or len({v["filename"] for v in videos}) != 5):
        raise ValueError("WebVid5 requires five unique videos covering the five frozen categories")
    profile = repo / cohort["profile"]
    if sha256(profile) != cohort["profile_sha256"]:
        raise ValueError("WebVid5 profile changed; explicitly version the cohort before changing the experiment")
    cfg = read_json(profile)
    cfg["models"] = read_json(repo / ".local/model_paths.json")
    data = Path(read_json(repo / ".local/datasets.json")["webvid55"])
    sources = []
    for video in videos:
        name = video["filename"]
        if Path(name).name != name or Path(name).suffix != ".mp4":
            raise ValueError(f"invalid cohort filename: {name}")
        path = (data / "processed" / name).resolve(strict=True)
        if sha256(data / "raw" / name) != video["raw_sha256"]:
            raise ValueError(f"official raw source hash mismatch: {name}")
        digest = sha256(path)
        if digest != video["processed_sha256"]:
            raise ValueError(f"processed source hash mismatch: {name}")
        info = probe(path)
        if any(info[key] != video[key] for key in ("width", "height", "frames", "fps")):
            raise ValueError(f"frozen video dimensions/length changed: {name}")
        video_config(cfg, info)
        sources.append(dict(id=path.stem, path=str(path), sha256=digest, **info))
    return cohort, profile, cfg, sources


def execution_identity(repo, cfg):
    """Conservative resume identity; large weights use file metadata, not rehashing."""
    code = {}
    for directory in ("src/semantic_transmission", "01_data_prep", "02_semantic_encoder",
                      "03_jscc_transmission", "04_semantic_decoder"):
        for path in sorted((repo / directory).rglob("*.py")):
            code[str(path.relative_to(repo))] = sha256(path)
    models = {}
    for root in [*(Path(p) for p in cfg["models"].values()), repo / ".local/checkpoints"]:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                stat = path.stat()
                models[str(path)] = {"resolved": str(path.resolve()), "bytes": stat.st_size,
                                     "mtime_ns": stat.st_mtime_ns}
    local = settings(repo)
    return {"code_sha256": code, "model_file_metadata": models,
            "settings": local, "environment": runtime_state(repo, local)}


def make_report(root, cohort, cfg, sources):
    """Report verified complete videos only; keep manual semantic review pending."""
    batch = read_json(root / "batch_manifest.json")
    records = {r["id"]: r for r in batch["runs"]}
    rows, cards = [], []
    media = root / "review_media"
    media.mkdir(exist_ok=True)
    for selection, source in zip(cohort["videos"], sources):
        name = source["id"]
        run = root / name
        record = records.get(name, {})
        row = {"category": selection["category"], "label": selection["label"], "id": name,
               "status": record.get("status", "NOT_STARTED"), "input_frames": source["frames"],
               "source": source["path"], "review_focus": selection["review_focus"],
               "manual_review_status": "PENDING", "reused_from": record.get("reused_from")}
        if row["status"] == "PASSED":
            try:
                if name not in completed_runs([root], cfg, [source]):
                    raise ValueError("completed run is not verified")
                quality = read_json(run / "quality.json")
                channel = read_json(run / "channel_accounting.json")
                sender = read_json(run / "sender_accounting.json")
                for boundary in BOUNDARIES:
                    for metric in METRICS:
                        value = quality[boundary][metric]
                        if not isinstance(value, (int, float)) or not math.isfinite(value):
                            raise ValueError(f"missing/nonfinite metric: {boundary}/{metric}")
                        row[f"{boundary}_{metric}"] = value
                row.update(output_frames=quality["video"]["frames"],
                           keyframes=read_json(run / "keyframes.json")["indices"],
                           pipeline_seconds=sum(s["seconds"] for s in record["stages"]),
                           serialized_model_input_bytes=sender["serialized_model_input_bytes"],
                           transmission_breakdown=channel["transmission_breakdown"],
                           cbr=channel["cbr_complex_uses_per_source_scalar"],
                           total_complex_channel_uses=channel["total_complex_channel_uses"])
                row["reconstruction"] = str(next((run / "receiver/reconstruction").glob("*.mp4")))
            except (ValueError, KeyError, OSError, StopIteration) as error:
                row.update(status="INVALID_RESULT", error=str(error))
        elif record:
            row["error"] = record.get("error", batch.get("error", "incomplete video"))
        rows.append(row)
        players = []
        for kind, path in (("source", source["path"]), ("reconstruction", row.get("reconstruction"))):
            if path:
                link = media / f"{selection['category']}_{kind}.mp4"
                if not link.is_symlink():
                    link.symlink_to(Path(path).resolve())
                players.append(f'<div>{kind}<br><video controls preload="metadata" '
                               f'src="review_media/{link.name}"></video></div>')
        cards.append(f'<section><h2>{html.escape(selection["label"])} — {html.escape(row["status"])}</h2>'
                     f'<p>{html.escape(selection["review_focus"])}</p>'
                     f'<div class="pair">{"".join(players)}</div>'
                     f'<p>입력 {source["frames"]}프레임 / 출력 {row.get("output_frames", "미완료")}프레임. '
                     '의미 오류 육안 검수: PENDING</p></section>')
    passed = [r for r in rows if r["status"] == "PASSED"]
    complete = len(passed) == 5 and batch["status"] == "PASSED"
    report = {"scope": cohort["scope"], "cohort": cohort["cohort"],
              "status": "PASSED" if complete else "INCOMPLETE",
              "batch_status": batch["status"], "completed_videos": len(passed), "expected_videos": 5,
              "meaning_of_passed": "All nine execution stages and artifact checks passed; no quality acceptance threshold.",
              "manual_review_status": "PENDING", "paper_reproduction": False,
              "evaluation_profile": cfg["evaluation_profile"], "snr_db": cfg["snr_db"],
              "decoder_policy": cfg["decoder_policy"], "concatenation_policy": cfg["concatenation_policy"],
              "mean_over_five_videos": ({f"{b}_{m}": sum(r[f"{b}_{m}"] for r in passed) / 5
                                         for b in BOUNDARIES for m in METRICS} if complete else None),
              "videos": rows}
    if complete:
        report["transmission_totals"] = {
            "complex_channel_uses": sum(r["total_complex_channel_uses"] for r in rows),
            "serialized_model_input_bytes": sum(r["serialized_model_input_bytes"] for r in rows),
            "cbr_over_all_source_scalars": sum(r["total_complex_channel_uses"] for r in rows)
                / sum(s["frames"] * s["width"] * s["height"] * 3 for s in sources)}
    write_json(root / "summary.json", report)
    fields = ["category", "label", "id", "status", "input_frames", "output_frames", "pipeline_seconds",
              "cbr", "total_complex_channel_uses", "serialized_model_input_bytes",
              *(f"{b}_{m}" for b in BOUNDARIES for m in METRICS), "reused_from", "error"]
    with (root / "per_video_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    review = root / "manual_review.csv"
    if not review.exists():
        with review.open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["category", "id", "review_focus", "manual_review_status", "notes"],
                                    extrasaction="ignore")
            writer.writeheader(); writer.writerows(rows)
    (root / "report.html").write_text('<!doctype html><html lang="ko"><meta charset="utf-8">'
        '<title>WebVid5 검증 결과</title><style>body{font-family:sans-serif;margin:2rem;max-width:1250px}'
        '.pair{display:flex;flex-wrap:wrap;gap:1rem}video{width:576px;max-width:100%}'
        'section{border-top:1px solid #bbb;margin-top:2rem}</style>'
        f'<h1>WebVid5: {len(passed)}/5편 실행·평가 완료</h1>'
        '<p>개발용 목적 표집입니다. WebVid 전체 성능이나 의미 보존 성공을 입증하지 않습니다. '
        '점수는 per_video_metrics.csv, 전체 집계와 항목별 전송량은 summary.json에 있습니다.</p>'
        '<p>왼쪽 원본, 오른쪽 복원입니다. 공식 구간 연결은 경계 프레임을 중복하므로 재생 시간에 차이가 '
        '있을 수 있습니다. 화질 평가는 해당 프레임 대응을 적용합니다. '
        '눈에 보이는 오류는 manual_review.csv에 기록하세요.</p>' + ''.join(cards) + '</html>')
    return report


def run(repo, dry_run=False):
    cohort, profile, cfg, sources = load_inputs(repo)
    protocol = {"cohort": cohort, "config": cfg, "execution": execution_identity(repo, cfg)}
    signature = fingerprint(protocol)
    state_path = repo / ".local/webvid5_history.json"
    history = read_json(state_path) if state_path.exists() else {"protocols": {}, "runs": []}
    previous = [Path(r["output"]) for r in reversed(history["runs"]) if r["signature"] == signature]
    reusable = completed_runs(previous, cfg, sources)
    tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    output = repo / "outputs" / ("webvid5_" + tag)
    command = ["--input-dir", str(Path(sources[0]["path"]).parent), "--output", str(output),
               "--profile", str(profile)]
    for source in sources:
        command += ["--video", source["id"]]
    for path in previous:
        command += ["--reuse-completed-from", str(path)]
    pending = [s for s in sources if s["id"] not in reusable]
    comparisons = sum(s["frames"] - 1 for s in pending)
    frames = sum(s["frames"] for s in pending)
    print(f"WebVid5: 576×320, 24fps, 전체 프레임, SKEM stride=1, 10dB, 공식 화질 지표 5개", flush=True)
    for video, source in zip(cohort["videos"], sources):
        action = "완료 결과 재사용" if source["id"] in reusable else "새로 실행"
        print(f"  {video['label']}: {source['frames']}프레임 ({source['duration']:.2f}초), {action}", flush=True)
    calibration = cohort.get("runtime_estimate", {})
    estimate = (comparisons * calibration.get("comparison_seconds", 35.89)
                + frames * calibration.get("other_seconds_per_frame", 1.97)) / 3600
    basis = calibration.get("basis", "기존 측정 기반 추정")
    print(f"생성 {len(pending)}편 / 재사용 {len(reusable)}편. 남은 모델 실행 예상 약 {estimate:.1f}시간 "
          f"({basis}; 실제 시간은 장면·키프레임 수에 따라 달라집니다).", flush=True)
    print(f"결과 폴더: {output}", flush=True)
    if dry_run:
        research(command + ["--dry-run"])
        return
    history["protocols"][signature] = protocol
    history["runs"].append({"output": str(output), "signature": signature})
    write_json(state_path, history)
    report = None
    try:
        research(command)
    finally:
        if (output / "batch_manifest.json").exists():
            write_json(output / "webvid5_protocol.json", dict(signature=signature, **protocol))
            report = make_report(output, cohort, cfg, sources)
            label = "전체 검증 완료" if report["status"] == "PASSED" else "검증 중단"
            print(f"{label} ({report['completed_videos']}/5편 완료): {output / 'report.html'}", flush=True)
    if not report or report["status"] != "PASSED":
        raise RuntimeError(f"five-video verification incomplete: {output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="verify the five full inputs and reuse; no model execution or output writes")
    args = parser.parse_args(argv)
    repo = repository()
    try:
        if args.dry_run:
            run(repo, dry_run=True)
        else:
            with (repo / ".local/webvid5_run.lock").open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise RuntimeError("이미 WebVid5 검증 명령이 실행 중입니다") from None
                run(repo)
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
