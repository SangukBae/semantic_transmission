"""Frozen source-grounded captions v2 and a separate, resumable reconstruction.

The old selection, captions, execution code and receipts remain unchanged.
--check never creates an output directory or starts model inference.
"""
import argparse
import csv
import html
import json
import re
import shutil
import signal
import subprocess
from pathlib import Path

from . import hybrid_reconstruction as hybrid
from .artifacts import sha256, write_json
from .assisted_captions import validate_bundle
from .hybrid_selection import DEFAULT_ROOT
from .webvid5 import fingerprint, read_json
from .webvid_ablation import Stages

REPO = hybrid.REPO
SPEC = REPO / "configs/captions/tv_low_08_faithful_v2.json"


def paths(root):
    return (root / "assistant_captions_v2/captions_bundle.json",
            root / "reconstruction_assistant_captions",
            root / "reconstruction_assistant_captions_v2")


def validate_spec(spec, parent, parent_path):
    for name in ("input_sha256", "selection_freeze_sha256"):
        if spec[name] != parent[name]:
            raise ValueError(f"revision source mismatch: {name}")
    if spec.get("version") != 1 or spec["parent_bundle_sha256"] != sha256(parent_path):
        raise ValueError("revision parent captions changed")
    expected = {str(r["segment"]) for r in parent["records"]}
    if set(spec["texts"]) != expected:
        raise ValueError("revision requires every segment exactly once")
    for text in spec["texts"].values():
        if (not isinstance(text, str) or not text.strip() or "\n" in text
                or len(text.split()) > 80 or not text.endswith((".", "!", "?"))
                or len(re.findall(r"[.!?](?:\s|$)", text)) > 6):
            raise ValueError("revision caption must be complete, at most 80 words and six sentences")
    sheets = {k:v for k,v in parent["evidence"].items() if k.startswith("sheets/")}
    if spec["reviewed_source_sheets"] != sheets:
        raise ValueError("revision must cover the unchanged source observation sheets")


def prepare(root=DEFAULT_ROOT, spec_path=SPEC):
    parent_path = root / "assistant_captions/captions_bundle.json"
    parent = validate_bundle(parent_path, root)
    spec = read_json(spec_path)
    validate_spec(spec, parent, parent_path)
    bundle_path, _, _ = paths(root)
    if bundle_path.exists():
        current = validate_bundle(bundle_path, root)
        if current.get("authored_spec_sha256") != sha256(spec_path):
            raise ValueError("preserve frozen v2 captions; use a new revision for edits")
        return bundle_path
    folder = bundle_path.parent
    folder.mkdir(parents=True, exist_ok=True)
    evidence = {}
    for relative, digest in parent["evidence"].items():
        target = folder / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(parent_path.parent / relative, target)
        if sha256(target) != digest:
            raise ValueError("source observation copy changed")
        evidence[relative] = digest
    shutil.copyfile(spec_path, folder / "authored.json")
    evidence["authored.json"] = sha256(spec_path)
    records = [dict(r, text=spec["texts"][str(r["segment"])]) for r in parent["records"]]
    payload = dict(version=1, revision=spec["revision"],
        status="CAPTIONS_PREPARED_RECONSTRUCTION_NOT_STARTED",
        input_sha256=parent["input_sha256"], selection_freeze_sha256=parent["selection_freeze_sha256"],
        source_frame_hashes=parent["source_frame_hashes"], records=records, author=spec["author"],
        protocol=" ".join(spec["instructions"]), sampling=spec["sampling"],
        scope="Offline assistant captions revised on development source samples after seeing v1 failures. Not blinded or independently validated.",
        parent_bundle_sha256=sha256(parent_path), authored_spec_sha256=sha256(spec_path),
        evidence=evidence, hallucination_mitigation_verified=False, reconstruction_started=False)
    write_json(bundle_path, dict(payload, checksum=fingerprint(payload)))
    validate_bundle(bundle_path, root)
    rows = "".join(f'<tr><td>{r["segment"]}<br>{r["start"]/24:.3f}–{r["end_exclusive"]/24:.3f}s</td>'
        f'<td>{html.escape(old["text"])}</td><td>{html.escape(r["text"])}</td></tr>'
        for old,r in zip(parent["records"],records))
    sheets = "".join(f'<details><summary>원본 표본 {p.stem}</summary><img loading="lazy" src="sheets/{p.name}"></details>'
                     for p in sorted((folder / "sheets").glob("*.jpg")))
    (folder / "review.html").write_text('<!doctype html><html lang="ko"><meta charset="utf-8">'
        '<title>캡션 v2</title><style>body{font:16px/1.6 system-ui;max-width:1200px;margin:24px auto;padding:16px}'
        'table{border-collapse:collapse}td,th{padding:10px;border:1px solid #ccc;vertical-align:top}img{max-width:100%}</style>'
        '<h1>같은 79장 · 수정 캡션 78개</h1><p>객체 위치·크기·부분 가림·변화 시점·원본 질감을 보강한 AI 작성 설명입니다. '
        '원본 표본판 20개를 다시 확인했습니다. 복원·성능 검증은 아직 미실행입니다.</p>'
        '<p><a href="captions_bundle.json">동결 캡션</a> · <a href="authored.json">작성 기준</a></p>'
        '<table><tr><th>구간</th><th>이전 v1</th><th>수정 v2</th></tr>'+rows+'</table>'+sheets+'</html>')
    return bundle_path


def assert_same_settings(old, new):
    # The caption file is the only changed inference setting; its path is provenance.
    drop = lambda cfg: {k:v for k,v in cfg.items() if k != "caption_bundle"}
    if drop(old) != drop(new):
        raise ValueError("non-caption reconstruction settings changed")


def preflight(root=DEFAULT_ROOT):
    bundle, previous, output = paths(root)
    spec = read_json(SPEC)
    parent_path = root / "assistant_captions/captions_bundle.json"
    parent = validate_bundle(parent_path, root)
    validate_spec(spec, parent, parent_path)
    revised = validate_bundle(bundle, root)
    if revised.get("authored_spec_sha256") != sha256(SPEC):
        raise ValueError("authored revision differs from the frozen bundle")
    # Both receipts must still match the untouched original execution code.
    hybrid.check_ready(root, previous, parent_path)
    _, selection, cfg, _, signature, report = hybrid.check_ready(root, output, bundle)
    assert_same_settings(read_json(previous / "run/run_config.json"), cfg)
    result = read_json(previous / "RESULT.json")
    if result["status"] != "PASS_60S_HYBRID_RECONSTRUCTION":
        raise ValueError("v1 reconstruction must be complete")
    old_cfg = read_json(previous / "run/run_config.json")
    if Path(old_cfg["caption_bundle"]) != parent_path:
        raise ValueError("v1 run does not use the revision's parent captions")
    if read_json(previous / "run/keyframes.json")["indices"] != selection["indices"]:
        raise ValueError("v1 keyframes differ from v2")
    source_files = [previous / p for p in (
        "execution_protocol.json", "RESULT.json", "run/run_config.json", "run/keyframes.json",
        "run/metadata_tx.json", "run/received/visual.c64", "run/receiver/metadata.csv",
        "run/quality.json", "run/receiver/reconstruction/sample_0000.mp4")]
    identity = dict(reconstruction_signature=signature,
        runner_sha256=sha256(Path(__file__)), command_sha256=sha256(REPO / "scripts/reconstruct_hybrid_v2.sh"),
        bundle_sha256=sha256(bundle), previous_files={str(p):sha256(p) for p in source_files})
    receipt = output / "revision_protocol.json"
    if receipt.exists() and read_json(receipt) != identity:
        raise ValueError("revision inputs/code changed; preserve this run")
    report.update(revision="faithful_v2", comparison_with=str(previous),
        same_keyframes_and_settings=True, received_inputs_verified_after_run=False,
        expected_comparison=str(output / "caption_comparison.html"))
    return identity, report


def csv_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def verify_pair(previous, output):
    old, new = previous / "run", output / "run"
    assert_same_settings(read_json(old / "run_config.json"), read_json(new / "run_config.json"))
    indices = read_json(old / "keyframes.json")["indices"]
    if read_json(new / "keyframes.json")["indices"] != indices:
        raise ValueError("caption comparison keyframes changed")
    if sha256(old / "received/visual.c64") != sha256(new / "received/visual.c64"):
        raise ValueError("received visual symbols differ; caption-only comparison not established")
    for i in indices:
        relative = f"receiver/frames/sample/key_frames_received/{i}.png"
        if sha256(old / relative) != sha256(new / relative):
            raise ValueError(f"received keyframe changed: {i}")
    a, b = csv_rows(old / "receiver/metadata.csv"), csv_rows(new / "receiver/metadata.csv")
    if len(a) != len(indices)-1 or len(b) != len(a):
        raise ValueError("received metadata segment count differs")
    if [(r["path"],float(r["flow"])) for r in a] != [(r["path"],float(r["flow"])) for r in b]:
        raise ValueError("flow or segment correspondence changed")
    for run, received in ((old,a),(new,b)):
        if [r["text"] for r in received] != [r["text"] for r in read_json(run / "captions.json")]:
            raise ValueError("transmitted caption text changed")
    return dict(received_keyframes_identical=len(indices), received_visual_symbols_identical=True,
        flow_and_segments_identical=True, generation_seed_and_settings_identical=True,
        diffusion_noise_tensor_identity_verified=False,
        scope="Full runs share keys, received images, flow and seed/settings. Initial noise tensors were not recorded; generated history changes with captions.")


def comparison(previous, output):
    paired = verify_pair(previous, output)
    old, new = previous / "run", output / "run"
    videos = [new / "data/normalized.mp4", old / "receiver/reconstruction/sample_0000.mp4",
              new / "receiver/reconstruction/sample_0000.mp4"]
    quality = [read_json(r / "quality.json") for r in (old,new)]
    cfg = read_json(new / "run_config.json")
    for video,q in zip(videos[1:], quality):
        if q["status"] != "PASSED" or sha256(video) != q["video_sha256"]:
            raise ValueError("comparison quality does not match video")
    if quality[0]["source_sha256"] != quality[1]["source_sha256"] or sha256(videos[0]) != quality[0]["source_sha256"]:
        raise ValueError("comparison source changed")
    target = output / "caption_comparison.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(videos[0]),
        "-i", str(videos[1]), "-i", str(videos[2]), "-filter_complex", "[0:v][1:v][2:v]hstack=inputs=3[v]",
        "-map", "[v]", "-an", "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p", str(target)], check=True)
    hybrid.pts_audit(target, cfg["frames"], cfg["fps"])
    channel = [read_json(r / "channel_accounting.json") for r in (old,new)]
    result = dict(status="CAPTION_COMPARISON_COMPLETE_REVIEW_PENDING", pairing=paired,
        comparison_sha256=sha256(target), quality_v1=quality[0]["delivered_mp4"], quality_v2=quality[1]["delivered_mp4"],
        channel_uses_v1=channel[0]["total_complex_channel_uses"], channel_uses_v2=channel[1]["total_complex_channel_uses"],
        hallucination_review="PENDING", hallucination_mitigation_verified=False)
    write_json(output / "CAPTION_COMPARISON.json", result)
    rows = "".join(f'<tr><td>{k}</td><td>{result["quality_v1"][k]:.4f}</td><td>{result["quality_v2"][k]:.4f}</td></tr>'
                   for k in ("psnr_db", "ssim", "lpips_vgg", "clip", "dists"))
    (output / "caption_comparison.html").write_text('<!doctype html><html lang="ko"><meta charset="utf-8">'
        '<title>캡션 v1 / v2 복원 비교</title><style>body{font:16px system-ui;margin:24px}video{width:100%}td,th{padding:10px}</style>'
        '<h1>원본 / 이전 캡션 v1 / 수정 캡션 v2</h1><video controls preload="metadata" src="caption_comparison.mp4"></video>'
        '<p>같은 키프레임 79장·수신 이미지·움직임·생성 시드와 설정을 확인했습니다. 초기 잡음 텐서의 완전 일치는 별도 기록이 없어 미확인입니다. '
        '캡션 변화가 이후 생성 참조에 미치는 영향까지 포함한 60초 전체 비교입니다.</p>'
        '<table><tr><th>지표</th><th>v1</th><th>v2</th></tr>'+rows+'</table>'
        f'<p>전송량: {result["channel_uses_v1"]:,} → {result["channel_uses_v2"]:,} 복소 심볼. 설명 길이에 따른 비용 변화 포함.</p>'
        '<p>의미 오류 검수·완화 효과는 아직 미검증입니다.</p><a href="CAPTION_COMPARISON.json">전체 수치·대응 검사</a>'
        ' · <a href="../assistant_captions_v2/review.html">캡션 변경 내용</a></html>')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--check", action="store_true", help="read-only readiness check; no inference")
    args = parser.parse_args(argv)
    if args.action == "prepare":
        if args.check:
            parser.error("--check applies only to run")
        print(f"PREPARED: {prepare()}")
        return
    identity, report = preflight()
    if args.check:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    bundle, previous, output = paths(DEFAULT_ROOT)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("SIGTERM")))
    with hybrid.lock(REPO / ".local/etri_caption_revision.lock"):
        # Recheck after acquiring the revision lock to prevent concurrent writers.
        identity, _ = preflight()
        write_json(output / "revision_protocol.json", identity)
        hybrid.execute(DEFAULT_ROOT, output, bundle)
        products = ["caption_comparison.mp4", "caption_comparison.html", "CAPTION_COMPARISON.json"]
        stages = Stages(output, fingerprint(identity))
        stages.step("caption-revision-comparison", products, products,
                    lambda _: comparison(previous,output))
    print(f"COMPLETE CAPTION V2: {output / 'caption_comparison.html'}", flush=True)


if __name__ == "__main__":
    main()
