"""Separate, resumable collision-repair comparison using completed FP32 inputs."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import tempfile

from . import condition_collision as fix, precision_reconstruction as precision
from . import precision_text_cache as text, generation_noise as noise, tail_reference as tail
from .artifacts import sha256, write_json
from .precision_review import metric_rows
from .webvid5 import fingerprint, read_json
from .webvid_ablation import Stages, environment, snapshot

REPO = tail.REPO
MODULE = "semantic_transmission.condition_repair"
CODE = ("src/semantic_transmission/condition_collision.py", "src/semantic_transmission/condition_repair.py",
        "scripts/etri_condition_fix_decoder.py", "scripts/reconstruct_condition_fix.sh",
        "src/semantic_transmission/precision_review.py", "src/semantic_transmission/reconstruction_console.py")
INPUTS = (*tail.INPUTS, text.REPORT)
VIDEOS = ("person_walk", "tv_low_08", "single_subject", "candle_flowers")


def source_for(video):
    if video not in VIDEOS:
        raise ValueError("unknown prepared video")
    if video == "tv_low_08":
        return precision.default_output("fp32")
    return REPO / "outputs/etri_latest_short3_20261001" / video / "reconstruction_fp32"


def preflight(source, output):
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("repair output must be separate from the source")
    old = source / "run"
    protocol = read_json(source / "execution_protocol.json")
    if any(sha256(REPO / p) != value for p, value in protocol["code"].items()):
        raise ValueError("frozen source code changed")
    policy = read_json(old / "receiver_policy.json")
    report = read_json(old / text.REPORT)
    keys = read_json(old / "keyframes.json")["indices"]
    if policy.get("condition_collision_policy") or policy["policy"] != tail.POLICY:
        raise ValueError("an unchanged tail17 baseline is required")
    if report["contract"]["compute_dtype"] != "fp32":
        raise ValueError("a completed FP32 baseline is required")
    text.validate_usage(old)
    if any(sha256(Path(entry["path"])) != entry["sha256"] for entry in report["entries"]):
        raise ValueError("stored T5 tensors changed")
    contract = policy["noise_contract"]
    noise.validate_trace(read_json(old / noise.TRACE), contract, len(keys)-1, contract["steps"])
    tail.validate_trace(read_json(old / "receiver/tail_reference_trace.json"), len(keys)-2)
    quality = read_json(old / "quality.json")
    if (quality["status"] != "PASSED" or quality["video_sha256"] != sha256(old / "receiver/reconstruction/sample_0000.mp4")
            or quality["source_sha256"] != sha256(old / "data/normalized.mp4")):
        raise ValueError("source quality/video identity differs")
    if (old / "receiver/decoder_config.py").read_text() != tail.decoder_config(old):
        raise ValueError("source decoder configuration differs from the frozen runner")
    plan = fix.preview(keys, fix.upstream_functions(REPO))
    identity = dict(version=1, policy=fix.POLICY, source=str(source), inputs=snapshot(old, INPUTS),
        source_evidence=snapshot(source, ["execution_protocol.json", "run/receiver_policy.json",
            "run/quality.json", "run/receiver/decoder_config.py", "run/"+noise.TRACE,
            "run/receiver/tail_reference_trace.json", "run/"+text.TRACE]),
        # This contract identifies baseline random draws, not the revised code.
        baseline_noise_contract=contract, plan=plan,
        baseline_code=protocol["code"], repair_code={p:sha256(REPO/p) for p in CODE})
    signature = fingerprint(identity)
    existing = output / "execution_protocol.json"
    if existing.exists():
        if read_json(existing) != dict(identity, signature=signature):
            raise ValueError("repair inputs/code changed; choose another output directory")
    elif output.exists():
        raise ValueError("unrecognized existing repair output")
    return identity, signature


def initialize(output, identity):
    source = Path(identity["source"])
    tail.initialize(source, output, snapshot(source / "run", tail.INPUTS))
    shutil.copyfile(source / "run" / text.REPORT, output / "run" / text.REPORT)
    if snapshot(output / "run", INPUTS) != identity["inputs"]:
        raise ValueError("receiver input copy changed")
    policy = read_json(source / "run/receiver_policy.json")
    reference = source / "run" / noise.TRACE
    policy.update(condition_collision_policy=fix.POLICY, paired_source=str(source),
        noise_reference=str(reference), noise_reference_sha256=sha256(reference))
    write_json(output / "run/receiver_policy.json", policy)


def reconstruct(run):
    from .decoder_runner import run as decode
    policy = read_json(run / "receiver_policy.json")
    source = Path(policy["paired_source"])
    if sha256(Path(policy["noise_reference"])) != policy["noise_reference_sha256"]:
        raise ValueError("baseline noise changed")
    config = run / "receiver/decoder_config.py"
    config.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / "run/receiver/decoder_config.py", config)
    for p in ("receiver/reference_trace.json", "receiver/tail_reference_trace.json", fix.TRACE):
        write_json(run / p, [])
    env = dict(os.environ, PYTHONPATH=str(REPO / ".local/vendor/Open-Sora"),
        ETRI_REFERENCE_TRACE=str(run / "receiver/reference_trace.json"),
        ETRI_TAIL_REFERENCE_TRACE=str(run / "receiver/tail_reference_trace.json"),
        ETRI_PRECISION_RUN=str(run))
    decode(run / "receiver/metadata.csv", run / "receiver/reconstruction", "key_frames_received",
        run / "receiver", decoder=REPO / "scripts/etri_condition_fix_decoder.py", config=config, environment=env)


def audit(run):
    precision.audit(run)
    identity = read_json(run.parent / "execution_protocol.json")
    if snapshot(run, INPUTS) != identity["inputs"]:
        raise ValueError("received inputs or prepared text changed")
    fix.validate_trace(read_json(run / fix.TRACE), identity["plan"])


def comparison(output):
    run = output / "run"
    audit(run)
    policy = read_json(run / "receiver_policy.json")
    source = Path(policy["paired_source"])
    quality = [read_json(p / "run/quality.json") for p in (source, output)]
    videos = [run / "data/normalized.mp4", source / "run/receiver/reconstruction/sample_0000.mp4",
              run / "receiver/reconstruction/sample_0000.mp4"]
    if any(q["status"] != "PASSED" or q["source_sha256"] != sha256(videos[0]) or
           q["video_sha256"] != sha256(v) for q, v in zip(quality, videos[1:])):
        raise ValueError("comparison quality identity mismatch")
    target = output / "comparison.mp4"
    precision.subprocess.run(["ffmpeg", "-v", "error", "-nostdin",
        *[x for v in videos for x in ("-i", str(v))], "-filter_complex", "[0:v][1:v][2:v]hstack=inputs=3[v]",
        "-map", "[v]", "-an", "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p", str(target)], check=True)
    cfg = read_json(run / "run_config.json")
    tail.hybrid.pts_audit(target, cfg["frames"], cfg["fps"])
    records = read_json(run / fix.TRACE)
    result = dict(status="PASS_CONDITION_COLLISION_REPAIR_REVIEW_PENDING", policy=fix.POLICY,
        previous=str(source), frames=cfg["frames"], fps=cfg["fps"],
        checked_segments=len(records), repaired_segments=sum(r[0]["repaired"] for r in records),
        received_inputs_identical=True, prepared_text_identical=True, noise_reference_verified=True,
        quality_before=quality[0]["delivered_mp4"], quality_after=quality[1]["delivered_mp4"],
        video_sha256=sha256(videos[2]), comparison_sha256=sha256(target), additional_channel_uses=0,
        hallucination_review="PENDING", hallucination_mitigation_verified=False,
        scope="Full sequential replay; only colliding endpoint placement changes. Later outputs can change through history.")
    write_json(output / "RESULT.json", result)
    (output / "review.html").write_text('<!doctype html><html lang="ko"><meta charset="utf-8">'
        '<title>키프레임 조건 충돌 수정</title><style>body{font:16px system-ui;margin:24px}video{width:100%}td,th{padding:10px}</style>'
        '<h1>원본 / 수정 전 / 조건 충돌 수정</h1><video controls preload="metadata" src="comparison.mp4"></video>'
        '<p>수신 자료·T5 저장값·VAE 및 생성 잡음 일치 확인. 충돌 구간만 조건 위치 변경.</p>'
        f'<p>검사 {len(records)}구간, 직접 수정 {result["repaired_segments"]}구간. 이전 출력 참조로 뒤 구간의 영상도 달라질 수 있습니다.</p>'
        '<table><tr><th>지표</th><th>수정 전</th><th>수정 후</th></tr>'
        + metric_rows(result["quality_before"], result["quality_after"]) + '</table>'
        '<p>화면 붕괴·새 왜곡의 검수와 완화 효과 확인은 별도입니다.</p><a href="RESULT.json">결과 기록</a></html>')


def execute(output, identity, signature):
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "execution_protocol.json", dict(identity, signature=signature))
    stages = Stages(output, signature)
    required = ["run/"+p for p in INPUTS] + ["run/receiver_policy.json"]
    stages.step("initialize", ["run"], required, lambda _: initialize(output, identity))
    env = environment(identity["baseline_noise_contract"]["seed"])
    env["PYTHONPATH"] = str(REPO / "src")
    jobs = [("reconstruct", MODULE, ["receiver/reconstruction", "receiver/decoder_config.py",
        "receiver/reference_trace.json", "receiver/tail_reference_trace.json", text.TRACE, noise.TRACE, fix.TRACE]),
        ("audit", MODULE, ["output_audit.json"]),
        ("evaluate", "semantic_transmission.research_quality", ["quality.json", "quality_delivered_mp4.csv", "quality_lossless_frames.csv"])]
    for name, module, products in jobs:
        args = ["--worker", name, "--run-dir", str(output / "run")] if module == MODULE else [name, str(output / "run")]
        command = [sys.executable, "-m", module, *args]
        resource = f"resources/{name}.json"
        required = ["run/"+p for p in products] + [resource]
        stages.step(name, required, required,
            lambda log, cmd=command, res=resource: tail.hybrid.launch(cmd, log, env, output / res))
    products = ["RESULT.json", "review.html", "comparison.mp4"]
    stages.step("comparison", products, products, lambda _: comparison(output))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", choices=VIDEOS, default="person_walk")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true", help="CPU readiness only; no reconstruction")
    parser.add_argument("--execute", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker", choices=("reconstruct", "audit"), help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        if not args.run_dir or args.check:
            parser.error("worker needs --run-dir and cannot use --check")
        return {"reconstruct": reconstruct, "audit": audit}[args.worker](args.run_dir.resolve())
    if args.run_dir:
        parser.error("--run-dir requires --worker")
    source = source_for(args.video).resolve()
    output = (args.output or source.with_name(source.name + "_condition_fix")).resolve()
    identity, signature = preflight(source, output)
    if args.check:
        print(json.dumps(dict(status="READY_FOR_USER_EXECUTION", video=args.video,
            segments=len(identity["plan"]), repairs=[r for r in identity["plan"] if r["repaired"]],
            output=str(output), quality_effect_verified=False, model_inference_started=False), ensure_ascii=False, indent=2))
        return 0
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("SIGTERM")))
    if args.execute:
        with tail.hybrid.lock(REPO / ".local/etri_60s_check.lock"):
            execute(output, identity, signature)
        return 0
    from .reconstruction_console import run_command
    # Initialize under the same stage rules so the console uses this video's
    # segment count instead of its legacy tv_low_08 fallback.
    with tail.hybrid.lock(REPO / ".local/etri_60s_check.lock"):
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "execution_protocol.json", dict(identity, signature=signature))
        required = ["run/"+p for p in INPUTS] + ["run/receiver_policy.json"]
        with contextlib.redirect_stdout(io.StringIO()):
            Stages(output, signature).step("initialize", ["run"], required, lambda _: initialize(output, identity))
    folder = REPO / ".local/reconstruction_console"
    folder.mkdir(parents=True, exist_ok=True)
    session = Path(tempfile.mkdtemp(prefix="condition-fix-", dir=folder))
    command = [sys.executable, "-m", MODULE, "--execute", "--video", args.video, "--output", str(output)]
    code = run_command(command, output, session / "console.log", session / "progress.json")
    if code == 0:
        print(f"COMPLETE CONDITION FIX: {output / 'review.html'}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
