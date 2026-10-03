"""FC-LGVSC default: prepared FP32 inputs with both receiver repairs."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

from . import condition_repair as base, schedule_repair as paired
from .artifacts import sha256, write_json
from .precision_review import metric_rows
from .temporal import segment_lengths
from .webvid5 import fingerprint, read_json
from .webvid_ablation import Stages, environment, snapshot

REPO, INPUTS = base.REPO, base.INPUTS
MODULE = "semantic_transmission.fc_lgvsc"
POLICY = "fc_lgvsc_default_v1"
CODE = ("src/semantic_transmission/fc_lgvsc.py", "scripts/reconstruct_fc_lgvsc.sh")
# Reuse the exact decoder and audits validated on the full person_walk run.
reconstruct, audit = paired.reconstruct, paired.audit


def preflight(source, output):
    source, output = source.resolve(), output.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("default output must be separate from the source")
    # The frozen validator also checks a collision-only destination. Give it
    # an empty temporary destination, then validate our own protocol below.
    # It does not initialize or reconstruct anything in this directory.
    with tempfile.TemporaryDirectory(prefix="fc-lgvsc-check-") as scratch:
        baseline, _ = base.preflight(source, Path(scratch) / "unused")
    receipts = {}
    for p in (source / "stages").glob("*.json"):
        row = read_json(p)
        if row["status"] != "PASSED" or snapshot(source, row["required"]) != row["artifacts"]:
            raise ValueError(f"completed baseline artifacts changed: {p}")
        receipts[p.stem] = row
    required = {"initialize", "reconstruct", "evaluate", "comparison"}
    if not required <= receipts.keys() or not {"audit", "output-audit"} & receipts.keys():
        raise ValueError("incomplete FP32 baseline")
    for row in receipts.values():
        if any(k not in receipts or fingerprint(receipts[k]) != value
               for k, value in row["dependencies"].items()):
            raise ValueError("baseline stage dependencies changed")
    run = source / "run"
    cfg = read_json(run / "run_config.json")
    keys = read_json(run / "keyframes.json")["indices"]
    code = sorted(set(baseline["baseline_code"]) | set(baseline["repair_code"]) | set(paired.CODE) | set(CODE))
    identity = dict(version=1, policy=POLICY, source=str(source), inputs=baseline["inputs"],
        baseline_noise_contract=baseline["baseline_noise_contract"], plan=baseline["plan"],
        condition_collision_policy=base.fix.POLICY, short_schedule_policy=paired.fix.POLICY,
        t5_compute_dtype="fp32", frames=cfg["frames"], fps=cfg["fps"],
        schedule_plan=[dict(loop=i, frames=n, repaired=1<n<17) for i,n in enumerate(segment_lengths(keys))],
        source_evidence=snapshot(source, ["execution_protocol.json", "stages", "RESULT.json",
            "run/receiver_policy.json", "run/receiver/decoder_config.py"]),
        code={name:sha256(REPO/name) for name in code})
    signature = fingerprint(identity)
    protocol = output / "execution_protocol.json"
    if protocol.exists():
        if read_json(protocol) != dict(identity, signature=signature):
            raise ValueError("default inputs/code changed; choose another output directory")
    elif output.exists():
        raise ValueError("unrecognized existing default output")
    return identity, signature


def initialize(output, identity):
    base.initialize(output, identity)
    path = output / "run/receiver_policy.json"
    policy = read_json(path)
    policy.update(default_method=POLICY, short_schedule_policy=paired.fix.POLICY)
    write_json(path, policy)


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
    collisions = read_json(run / base.fix.TRACE)
    schedules = read_json(run / paired.fix.TRACE)
    result = dict(status="PASS_FC_LGVSC_RECONSTRUCTION_REVIEW_PENDING", policy=POLICY,
        previous=str(source), frames=identity["frames"], fps=identity["fps"], checked_segments=len(schedules),
        condition_repaired_segments=[i for i,r in enumerate(collisions) if r[0]["repaired"]],
        schedule_repaired_segments=[r["loop"] for r in schedules if r["repaired"]],
        received_inputs_identical=True, prepared_text_identical=True, noise_reference_verified=True,
        quality_before=quality[0]["delivered_mp4"], quality_after=quality[1]["delivered_mp4"],
        video_sha256=sha256(videos[2]), comparison_sha256=sha256(target), additional_channel_uses=0,
        hallucination_review="PENDING", hallucination_mitigation_verified=False,
        scope="Both endpoint collision and short-segment schedule repairs; later references change through sequential replay.")
    write_json(output / "RESULT.json", result)
    (output / "review.html").write_text('<!doctype html><html lang="ko"><meta charset="utf-8">'
        '<title>FC-LGVSC 기본 복원</title><style>body{font:16px system-ui;margin:24px}video{width:100%}td,th{padding:10px}</style>'
        '<h1>원본 / 수정 전 FP32 / FC-LGVSC 기본</h1><video controls preload="metadata" src="comparison.mp4"></video>'
        '<p>같은 수신 자료·캡션 v2·T5 FP32 저장값·잡음. 조건 충돌 방지와 짧은 구간 시간표 수정 적용.</p>'
        '<table><tr><th>지표</th><th>수정 전</th><th>기본 방식</th></tr>'
        + metric_rows(result["quality_before"],result["quality_after"]) + '</table>'
        '<p>복원 완료와 할루시네이션 완화 검증은 별도입니다.</p><a href="RESULT.json">결과 기록</a></html>')


def initialize_stage(output, identity, signature):
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "execution_protocol.json", dict(identity, signature=signature))
    stages = Stages(output, signature)
    products = ["run/"+p for p in INPUTS] + ["run/receiver_policy.json"]
    stages.step("initialize", ["run"], products, lambda _: initialize(output, identity))
    return stages


def execute(output, identity, signature):
    stages = initialize_stage(output, identity, signature)
    env = environment(identity["baseline_noise_contract"]["seed"])
    env["PYTHONPATH"] = str(REPO / "src")
    jobs = [("reconstruct", MODULE, ["receiver/reconstruction", "receiver/decoder_config.py", "receiver/reference_trace.json",
        "receiver/tail_reference_trace.json", base.text.TRACE, base.noise.TRACE, base.fix.TRACE, paired.fix.TRACE]),
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
    parser.add_argument("--video", choices=base.VIDEOS, default="tv_low_08")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true", help="check prepared inputs without model inference")
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
    source = base.source_for(args.video).resolve()
    output = (args.output or source.with_name(source.name+"_fc_lgvsc_default_v1")).resolve()
    identity,signature = preflight(source,output)
    if args.check:
        print(json.dumps(dict(status="READY",video=args.video,output=str(output),policy=POLICY,
            t5_compute_dtype="fp32",segments=len(identity["plan"]),
            condition_collision_policy=identity["condition_collision_policy"],
            short_schedule_policy=identity["short_schedule_policy"],
            condition_repairs=[r for r in identity["plan"] if r["repaired"]],
            schedule_repairs=[r for r in identity["schedule_plan"] if r["repaired"]],
            model_inference_started=False),ensure_ascii=False,indent=2))
        return 0
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("SIGTERM")))
    if args.execute:
        with base.tail.hybrid.lock(REPO / ".local/etri_60s_check.lock"):
            execute(output,identity,signature)
        return 0
    from .reconstruction_console import run_command
    with base.tail.hybrid.lock(REPO / ".local/etri_60s_check.lock"):
        with contextlib.redirect_stdout(io.StringIO()):
            initialize_stage(output,identity,signature)
    folder=REPO/".local/reconstruction_console"
    folder.mkdir(parents=True,exist_ok=True)
    session=Path(tempfile.mkdtemp(prefix="fc-lgvsc-",dir=folder))
    command=[sys.executable,"-m",MODULE,"--execute","--video",args.video,"--output",str(output)]
    code=run_command(command,output,session/"console.log",session/"progress.json")
    if code==0:
        print(f"COMPLETE FC-LGVSC: {output/'review.html'}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
