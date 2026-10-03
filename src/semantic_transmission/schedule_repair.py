"""Resumable paired schedule repair from a completed collision-fixed run."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile

from . import condition_repair as base, short_schedule as fix
from .artifacts import sha256, write_json
from .precision_review import metric_rows
from .temporal import segment_lengths
from .webvid5 import fingerprint, read_json
from .webvid_ablation import Stages, environment, snapshot

REPO, INPUTS = base.REPO, base.INPUTS
MODULE = "semantic_transmission.schedule_repair"
CODE = ("src/semantic_transmission/short_schedule.py", "src/semantic_transmission/schedule_repair.py",
        "scripts/etri_schedule_fix_decoder.py", "scripts/reconstruct_schedule_fix.sh",
        ".local/vendor/Open-Sora/opensora/schedulers/rf/rectified_flow.py")


def preflight(source, output):
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("schedule output must be separate from the source")
    policy = read_json(source / "run/receiver_policy.json")
    if (policy.get("condition_collision_policy") != base.fix.POLICY
            or policy.get("short_schedule_policy")):
        raise ValueError("a completed collision-fixed baseline is required")
    # This also checks frozen baseline/repair code and the existing identity.
    original = Path(policy["paired_source"])
    baseline, _ = base.preflight(original, source)
    for parent in (original, source):
        receipts = {}
        for p in (parent / "stages").glob("*.json"):
            row = read_json(p)
            if row["status"] != "PASSED" or snapshot(parent, row["required"]) != row["artifacts"]:
                raise ValueError(f"completed baseline artifacts changed: {p}")
            receipts[p.stem] = row
        for row in receipts.values():
            if any(fingerprint(receipts[k]) != value for k, value in row["dependencies"].items()):
                raise ValueError("baseline stage dependencies changed")
        audit_stage = "output-audit" if parent == original else "audit"
        if not {"initialize", "reconstruct", audit_stage, "evaluate", "comparison"} <= set(receipts):
            raise ValueError("incomplete baseline stages")
    run = source / "run"
    keys = read_json(run / "keyframes.json")["indices"]
    contract = policy["noise_contract"]
    base.text.validate_usage(run)
    base.noise.validate_trace(read_json(run / base.noise.TRACE), contract, len(keys)-1, contract["steps"])
    base.fix.validate_trace(read_json(run / base.fix.TRACE), baseline["plan"])
    base.tail.validate_trace(read_json(run / "receiver/tail_reference_trace.json"), len(keys)-2)
    cfg = read_json(run / "run_config.json")
    lengths = segment_lengths(keys)
    code = sorted(set(baseline["baseline_code"]) | set(baseline["repair_code"]) | set(CODE))
    identity = dict(version=1, policy=fix.POLICY, source=str(source), original=str(original),
        inputs=snapshot(run, INPUTS), baseline_noise_contract=contract, plan=baseline["plan"],
        schedule_plan=[dict(loop=i, frames=n, repaired=1<n<17) for i,n in enumerate(lengths)],
        frames=cfg["frames"], fps=cfg["fps"],
        source_evidence=snapshot(source, ["execution_protocol.json", "stages", "RESULT.json",
            "run/receiver_policy.json", "run/receiver/decoder_config.py"]),
        code={name:sha256(REPO/name) for name in code})
    signature = fingerprint(identity)
    p = output / "execution_protocol.json"
    if p.exists():
        if read_json(p) != dict(identity, signature=signature):
            raise ValueError("schedule inputs/code changed; choose another output directory")
    elif output.exists():
        raise ValueError("unrecognized existing schedule output")
    return identity, signature


def initialize(output, identity):
    source = Path(identity["source"])
    base.tail.initialize(source, output, snapshot(source / "run", base.tail.INPUTS))
    shutil.copyfile(source / "run" / base.text.REPORT, output / "run" / base.text.REPORT)
    if snapshot(output / "run", INPUTS) != identity["inputs"]:
        raise ValueError("receiver input copy differs")
    policy = read_json(source / "run/receiver_policy.json")
    reference = source / "run" / base.noise.TRACE
    policy.update(short_schedule_policy=fix.POLICY, paired_source=str(source),
        noise_reference=str(reference), noise_reference_sha256=sha256(reference))
    write_json(output / "run/receiver_policy.json", policy)


def reconstruct(run):
    from .decoder_runner import run as decode
    policy = read_json(run / "receiver_policy.json")
    if sha256(Path(policy["noise_reference"])) != policy["noise_reference_sha256"]:
        raise ValueError("baseline noise modified")
    config = run / "receiver/decoder_config.py"
    shutil.copyfile(Path(policy["paired_source"]) / "run/receiver/decoder_config.py", config)
    for p in ("receiver/reference_trace.json", "receiver/tail_reference_trace.json", base.fix.TRACE, fix.TRACE):
        write_json(run / p, [])
    env = dict(os.environ, PYTHONPATH=str(REPO / ".local/vendor/Open-Sora"),
        ETRI_REFERENCE_TRACE=str(run / "receiver/reference_trace.json"),
        ETRI_TAIL_REFERENCE_TRACE=str(run / "receiver/tail_reference_trace.json"), ETRI_PRECISION_RUN=str(run))
    decode(run / "receiver/metadata.csv", run / "receiver/reconstruction", "key_frames_received",
        run / "receiver", decoder=REPO / "scripts/etri_schedule_fix_decoder.py", config=config, environment=env)


def audit(run):
    base.precision.audit(run)
    identity = read_json(run.parent / "execution_protocol.json")
    if snapshot(run, INPUTS) != identity["inputs"]:
        raise ValueError("received inputs/prepared text differ")
    base.fix.validate_trace(read_json(run / base.fix.TRACE), identity["plan"])
    fix.validate_trace(read_json(run / fix.TRACE), read_json(run / "keyframes.json")["indices"],
                       identity["baseline_noise_contract"]["steps"])


def comparison(output):
    run = output / "run"
    audit(run)
    identity = read_json(output / "execution_protocol.json")
    source = Path(identity["source"])
    quality = [read_json(p / "run/quality.json") for p in (source, output)]
    videos = [run / "data/normalized.mp4", source / "run/receiver/reconstruction/sample_0000.mp4",
              run / "receiver/reconstruction/sample_0000.mp4"]
    if any(q["status"] != "PASSED" or q["source_sha256"] != sha256(videos[0]) or
           q["video_sha256"] != sha256(v) for q,v in zip(quality,videos[1:])):
        raise ValueError("comparison quality/video mismatch")
    target = output / "comparison.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", *[x for v in videos for x in ("-i",str(v))],
        "-filter_complex", "[0:v][1:v][2:v]hstack=inputs=3[v]", "-map", "[v]", "-an", "-c:v", "libx264",
        "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p", str(target)], check=True)
    base.tail.hybrid.pts_audit(target, identity["frames"], identity["fps"])
    records = read_json(run / fix.TRACE)
    result = dict(status="PASS_SHORT_SCHEDULE_RECONSTRUCTION_REVIEW_PENDING", policy=fix.POLICY,
        previous=str(source), frames=identity["frames"], fps=identity["fps"], checked_segments=len(records),
        repaired_segments=[r["loop"] for r in records if r["repaired"]],
        received_inputs_identical=True, prepared_text_identical=True, noise_reference_verified=True,
        condition_placement_identical=True, quality_before=quality[0]["delivered_mp4"],
        quality_after=quality[1]["delivered_mp4"], video_sha256=sha256(videos[2]),
        comparison_sha256=sha256(target), additional_channel_uses=0,
        hallucination_review="PENDING", hallucination_mitigation_verified=False,
        scope="Only short-segment scheduling changes; later generated references change through sequential replay.")
    write_json(output / "RESULT.json", result)
    (output / "review.html").write_text('<!doctype html><html lang="ko"><meta charset="utf-8">'
        '<title>짧은 구간 생성 시간표 수정</title><style>body{font:16px system-ui;margin:24px}video{width:100%}td,th{padding:10px}</style>'
        '<h1>원본 / 조건 충돌 수정 / 생성 시간표 추가 수정</h1><video controls preload="metadata" src="comparison.mp4"></video>'
        '<p>같은 수신 자료·T5·조건 위치·잡음. 시간표를 수정한 첫 구간의 영향이 이전 영상 참조를 통해 뒤 구간으로 전달됩니다.</p>'
        '<table><tr><th>지표</th><th>시간표 수정 전</th><th>수정 후</th></tr>'
        + metric_rows(result["quality_before"],result["quality_after"]) + '</table>'
        '<p>독립 의미 오류 검수와 할루시네이션 완화 검증은 별도입니다.</p><a href="RESULT.json">결과 기록</a></html>')


def execute(output, identity, signature):
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "execution_protocol.json", dict(identity, signature=signature))
    stages = Stages(output, signature)
    products = ["run/"+p for p in INPUTS] + ["run/receiver_policy.json"]
    stages.step("initialize", ["run"], products, lambda _: initialize(output, identity))
    env = environment(identity["baseline_noise_contract"]["seed"])
    env["PYTHONPATH"] = str(REPO / "src")
    jobs = [("reconstruct", MODULE, ["receiver/reconstruction", "receiver/decoder_config.py", "receiver/reference_trace.json",
        "receiver/tail_reference_trace.json", base.text.TRACE, base.noise.TRACE, base.fix.TRACE, fix.TRACE]),
        ("audit", MODULE, ["output_audit.json"]),
        ("evaluate", "semantic_transmission.research_quality", ["quality.json", "quality_delivered_mp4.csv", "quality_lossless_frames.csv"])]
    for name,module,products in jobs:
        args = ["--worker",name,"--run-dir",str(output / "run")] if module == MODULE else [name,str(output / "run")]
        command = [sys.executable,"-m",module,*args]
        resource = f"resources/{name}.json"
        required = ["run/"+p for p in products]+[resource]
        stages.step(name, required, required,
            lambda log,cmd=command,res=resource: base.tail.hybrid.launch(cmd,log,env,output/res))
    products = ["RESULT.json","review.html","comparison.mp4"]
    stages.step("comparison",products,products,lambda _: comparison(output))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", choices=base.VIDEOS, default="person_walk")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--execute", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker", choices=("reconstruct","audit"), help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        if not args.run_dir or args.check:
            parser.error("worker requires --run-dir without --check")
        return {"reconstruct":reconstruct,"audit":audit}[args.worker](args.run_dir.resolve())
    if args.run_dir:
        parser.error("--run-dir requires --worker")
    original = base.source_for(args.video)
    source = original.with_name(original.name+"_condition_fix").resolve()
    output = (args.output or source.with_name(source.name+"_schedule_fix")).resolve()
    identity,signature = preflight(source,output)
    if args.check:
        print(json.dumps(dict(status="READY",output=str(output),plan=identity["schedule_plan"],model_inference_started=False),indent=2))
        return 0
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("SIGTERM")))
    if args.execute:
        with base.tail.hybrid.lock(REPO / ".local/etri_60s_check.lock"):
            execute(output,identity,signature)
        return 0
    from .reconstruction_console import run_command
    with base.tail.hybrid.lock(REPO / ".local/etri_60s_check.lock"):
        output.mkdir(parents=True,exist_ok=True)
        write_json(output / "execution_protocol.json",dict(identity,signature=signature))
        products=["run/"+p for p in INPUTS]+["run/receiver_policy.json"]
        with contextlib.redirect_stdout(io.StringIO()):
            Stages(output,signature).step("initialize",["run"],products,lambda _:initialize(output,identity))
    folder=REPO/".local/reconstruction_console"
    folder.mkdir(parents=True,exist_ok=True)
    session=Path(tempfile.mkdtemp(prefix="schedule-fix-",dir=folder))
    command=[sys.executable,"-m",MODULE,"--execute","--video",args.video,"--output",str(output)]
    code=run_command(command,output,session/"console.log",session/"progress.json")
    if code==0:
        print(f"COMPLETE SCHEDULE FIX: {output/'review.html'}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
