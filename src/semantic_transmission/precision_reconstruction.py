"""Prepared T5 FP32 reconstruction with tail17 and independently fixed noise."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess

from . import caption_revision, generation_noise as noise, precision_text_cache as text, tail_reference as tail
from .artifacts import sha256, write_json
from .webvid5 import fingerprint, read_json
from .webvid_ablation import Stages, environment, snapshot

REPO, SOURCE = tail.REPO, tail.SOURCE
PREVIOUS = tail.T5_OUTPUT.with_name(tail.T5_OUTPUT.name + "_console")
MODULE = "semantic_transmission.precision_reconstruction"
CODE = (*tail.CODE, *text.CODE, "src/semantic_transmission/precision_reconstruction.py",
        "src/semantic_transmission/generation_noise.py", "scripts/etri_precision_decoder.py",
        "scripts/reconstruct_fp32.sh")


def default_output(precision="fp32"):
    return SOURCE.with_name(SOURCE.name + f"_tail17_t5{precision}_fixednoise")


def preflight(output, precision, noise_reference=None):
    tail.validate_destination(output, SOURCE)
    caption_revision.preflight()
    receipts = {}
    for path in sorted((SOURCE / "stages").glob("*.json")):
        record = read_json(path)
        if record["status"] != "PASSED" or snapshot(SOURCE, record["required"]) != record["artifacts"]:
            raise ValueError(f"frozen v2 artifacts changed: {path}")
        receipts[path.name] = sha256(path)
    expected = {f"{name}.json" for name in ("initialize", *[s for s, _, _ in tail.hybrid.job_plan(True)],
                                           "caption-revision-comparison")}
    if set(receipts) != expected or read_json(SOURCE / "RESULT.json")["status"] != "PASS_60S_HYBRID_RECONSTRUCTION":
        raise ValueError("complete frozen v2 input required")
    cfg = read_json(SOURCE / "run/run_config.json")
    inputs = snapshot(SOURCE / "run", tail.INPUTS)
    keys = read_json(SOURCE / "run/keyframes.json")["indices"]
    decoder = read_json(SOURCE / "run/receiver/decoder_inputs.json")["decoder"]
    code_names = sorted(set(CODE) | set(read_json(SOURCE / "execution_protocol.json")["code"]))
    code = {name: sha256(REPO / name) for name in code_names}
    contract = dict(policy=noise.POLICY, seed=cfg["seed"], segments=len(keys)-1, steps=decoder["steps"],
                    inputs=fingerprint(inputs), code=fingerprint(code), reference_policy=tail.POLICY)
    identity = dict(version=1, policy="t5_precision_with_scoped_noise_v1", source=str(SOURCE),
        source_receipts=receipts, inputs=inputs, code=code, precision=precision,
        t5_device="cpu" if precision == "fp32" else "cuda", noise_contract=contract,
        model_inventory=text.legacy.model_inventory(cfg["models"]["t5"]),
        packages={n: importlib.metadata.version(n) for n in ("torch", "transformers", "diffusers", "tokenizers")},
        noise_reference=str(noise_reference) if noise_reference else None)
    if noise_reference:
        reference = read_json(noise_reference)
        noise.validate_trace(reference, contract, len(keys)-1, decoder["steps"])
        identity["noise_reference_sha256"] = sha256(noise_reference)
        # A completed reference run also supplies the paired quality comparison.
        other = noise_reference.parents[2]
        if noise_reference != other / "run" / noise.TRACE or not (other / "run/quality.json").is_file():
            raise ValueError("noise reference must belong to a completed, evaluated precision run")
        if other == output:
            raise ValueError("noise reference cannot be this output")
    signature = fingerprint(identity)
    protocol = output / "execution_protocol.json"
    if protocol.exists() and read_json(protocol) != dict(identity, signature=signature):
        raise ValueError("precision run inputs/code changed; preserve it and choose a new --output")
    report = dict(status="READY_FOR_USER_EXECUTION", video="tv_low_08", frames=cfg["frames"], fps=cfg["fps"],
        keyframes=len(keys), captions=len(keys)-1, steps=decoder["steps"], reference_policy=tail.POLICY,
        t5_compute_dtype=precision, t5_device=identity["t5_device"], decoder_dtype="bf16",
        persistent_text_cache=True, noise_policy=noise.POLICY, noise_reference=identity["noise_reference"],
        legacy_noise_recovered=False, quality_effect_verified=False, output=str(output),
        expected_review=str(output / "review.html"), model_inference_started=False)
    return identity, signature, report


def initialize(output, identity):
    tail.initialize(SOURCE, output, identity["inputs"])
    policy = read_json(output / "run/receiver_policy.json")
    policy.update(text_encoder_policy=text.POLICY, t5_precision=identity["precision"],
                  noise_contract=identity["noise_contract"], noise_reference=identity["noise_reference"],
                  noise_reference_sha256=identity.get("noise_reference_sha256"))
    write_json(output / "run/receiver_policy.json", policy)


def prepare(run):
    policy = read_json(run / "receiver_policy.json")
    text.prepare(run, REPO, tail.decoder_config(run), policy["t5_precision"])


def reconstruct(run):
    from .decoder_runner import run as decode
    policy = read_json(run / "receiver_policy.json")
    report = read_json(run / text.REPORT)
    if report["contract"]["compute_dtype"] != policy["t5_precision"] or report["policy"] != text.POLICY:
        raise ValueError("incorrect T5 precision cache")
    if policy["noise_reference"] and sha256(Path(policy["noise_reference"])) != policy["noise_reference_sha256"]:
        raise ValueError("noise reference changed after preflight")
    config = run / "receiver/decoder_config.py"
    config.write_text(tail.decoder_config(run))
    trace, tail_trace = (run / f"receiver/{name}.json" for name in ("reference_trace", "tail_reference_trace"))
    write_json(trace, [])
    write_json(tail_trace, [])
    env = dict(os.environ, PYTHONPATH=str(REPO / ".local/vendor/Open-Sora"),
        ETRI_REFERENCE_TRACE=str(trace), ETRI_TAIL_REFERENCE_TRACE=str(tail_trace), ETRI_PRECISION_RUN=str(run))
    decode(run / "receiver/metadata.csv", run / "receiver/reconstruction", "key_frames_received",
        run / "receiver", decoder=REPO / "scripts/etri_precision_decoder.py", config=config, environment=env)


def audit(run):
    tail.hybrid.hybrid_output_audit(run)
    policy = read_json(run / "receiver_policy.json")
    contract = policy["noise_contract"]
    tail.validate_trace(read_json(run / "receiver/tail_reference_trace.json"), contract["segments"]-1)
    text.validate_usage(run)
    actual = read_json(run / noise.TRACE)
    noise.validate_trace(actual, contract, contract["segments"], contract["steps"])
    if policy["noise_reference"]:
        if sha256(Path(policy["noise_reference"])) != policy["noise_reference_sha256"]:
            raise ValueError("noise reference modified")
        reference = read_json(policy["noise_reference"])
        if not actual["matched_reference"] or any(actual[k] != reference[k] for k in ("records", "runtime", "contract")):
            raise ValueError("paired VAE/initial/sampling noise differs")


def comparison(output):
    run = output / "run"
    policy = read_json(run / "receiver_policy.json")
    previous = Path(policy["noise_reference"]).parents[2] if policy["noise_reference"] else PREVIOUS
    paired = caption_revision.verify_pair(previous, output)
    if snapshot(previous / "run", tail.INPUTS) != snapshot(run, tail.INPUTS):
        raise ValueError("received inputs differ between comparisons")
    controlled = bool(policy["noise_reference"])
    if controlled:
        a, b = [read_json(root / "run" / text.REPORT) for root in (previous, output)]
        unchanged = lambda report: {k: v for k, v in report["contract"].items() if k not in {"compute_dtype", "device"}}
        if unchanged(a) != unchanged(b) or a["prompts"] != b["prompts"]:
            raise ValueError("paired T5 model, prompt, or non-precision configuration changed")
    paired.update(diffusion_noise_tensor_identity_verified=controlled, vae_noise_identity_verified=controlled,
        scope="Same tail17 and verified VAE/initial/RFLOW noise; T5 CPU FP32 vs GPU BF16 changes compute device as well."
        if controlled else "Historical comparison only: legacy noise was not recorded. No T5-only causal claim.")
    quality = [read_json(root / "run/quality.json") for root in (previous, output)]
    videos = [run / "data/normalized.mp4", previous / "run/receiver/reconstruction/sample_0000.mp4",
              run / "receiver/reconstruction/sample_0000.mp4"]
    if any(q["status"] != "PASSED" or q["source_sha256"] != sha256(videos[0])
           or q["video_sha256"] != sha256(video) for q, video in zip(quality, videos[1:])):
        raise ValueError("quality artifacts do not match comparison videos")
    target = output / "comparison.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", *[x for v in videos for x in ("-i", str(v))],
        "-filter_complex", "[0:v][1:v][2:v]hstack=inputs=3[v]", "-map", "[v]", "-an", "-c:v", "libx264",
        "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p", str(target)], check=True)
    cfg = read_json(run / "run_config.json")
    tail.hybrid.pts_audit(target, cfg["frames"], cfg["fps"])
    report = read_json(run / text.REPORT)
    result = dict(status="PASS_T5_PRECISION_RECONSTRUCTION_REVIEW_PENDING", pairing=paired,
        precision=policy["t5_precision"], t5_device=report["contract"]["device"], reference_policy=tail.POLICY,
        frames=cfg["frames"], fps=cfg["fps"], previous=str(previous), quality_before=quality[0]["delivered_mp4"],
        quality_after=quality[1]["delivered_mp4"], video_sha256=sha256(videos[2]), comparison_sha256=sha256(target),
        noise_trace_sha256=sha256(run / noise.TRACE), noise_reference_verified=controlled,
        channel_uses=read_json(run / "channel_accounting.json")["total_complex_channel_uses"],
        additional_channel_uses=0, hallucination_review="PENDING", hallucination_mitigation_verified=False)
    write_json(output / "RESULT.json", result)
    rows = ''.join(f'<tr><td>{key}</td><td>{result["quality_before"][key]:.4f}</td>'
                   f'<td>{result["quality_after"][key]:.4f}</td></tr>' for key in result["quality_after"])
    note = "두 실행의 VAE·초기 생성·반복 생성 잡음 해시 일치를 확인했습니다." if controlled else \
        "이전 결과에는 잡음 기록이 없어 잡음이 같은 비교가 아닙니다. 이 결과만으로 T5의 영향을 확정할 수 없습니다."
    (output / "review.html").write_text('<!doctype html><html lang="ko"><meta charset="utf-8">'
        '<title>T5 정밀도 복원</title><style>body{font:16px system-ui;margin:24px}video{width:100%}'
        'td,th{padding:10px}</style><h1>원본 / 비교 기준 / 이번 복원</h1>'
        '<video controls preload="metadata" src="comparison.mp4"></video>'
        f'<p>T5 {policy["t5_precision"].upper()} · 마지막 17프레임 참조 · 생성 30단계.</p><p>{note}</p>'
        '<p>FP32는 CPU, BF16은 GPU에서 사전 계산합니다. 영상 생성기는 기존 BF16을 유지합니다.</p>'
        '<table><tr><th>지표</th><th>비교 기준</th><th>이번 복원</th></tr>'+rows+
        '</table><p>의미 오류 검수·할루시네이션 완화 입증은 미완료입니다.</p><a href="RESULT.json">검증 기록</a></html>')


def execute(output, identity, signature, prepare_only=False):
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "execution_protocol.json", dict(identity, signature=signature))
    stages = Stages(output, signature)
    stages.step("initialize", ["run"], [f"run/{p}" for p in tail.INPUTS] + ["run/receiver_policy.json"],
                lambda _: initialize(output, identity))
    env = environment(identity["noise_contract"]["seed"])
    env["PYTHONPATH"] = str(REPO / "src")
    python = tail.hybrid.settings(REPO)["python"]
    jobs = [("prepare-text", MODULE, [text.REPORT]),
        ("reconstruct", MODULE, ["receiver/decoder_config.py", "receiver/reconstruction",
          "receiver/reference_trace.json", "receiver/tail_reference_trace.json", text.TRACE, noise.TRACE]),
        ("audit", MODULE, ["output_audit.json"]),
        ("evaluate", "semantic_transmission.research_quality",
         ["quality.json", "quality_delivered_mp4.csv", "quality_lossless_frames.csv"])]
    for name, module, products in jobs:
        args = ["--worker", name, "--run-dir", str(output / "run")] if module == MODULE else [name, str(output / "run")]
        command = [python, "-m", module, *args]
        resource = f"resources/{name}.json"
        required = [f"run/{p}" for p in products] + [resource]
        stages.step(name, required, required,
                    lambda log, cmd=command, res=resource: tail.hybrid.launch(cmd, log, env, output / res))
        if name == "prepare-text" and prepare_only:
            print(f"PREPARED T5 ONLY: {output / 'run' / text.REPORT}", flush=True)
            return
    products = ["comparison.mp4", "RESULT.json", "review.html"]
    stages.step("comparison", products, products, lambda _: comparison(output))
    print(f"COMPLETE T5 PRECISION: {output / 'review.html'}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="read-only readiness check; no inference")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--t5-precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--noise-reference", type=Path, help="completed precision run's run/receiver/generation_noise.json")
    parser.add_argument("--prepare-text-only", action="store_true")
    parser.add_argument("--worker", choices=("prepare-text", "reconstruct", "audit"), help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        if args.run_dir is None or args.check:
            parser.error("worker requires --run-dir and cannot use --check")
        return {"prepare-text": prepare, "reconstruct": reconstruct, "audit": audit}[args.worker](args.run_dir.resolve())
    if args.run_dir is not None:
        parser.error("--run-dir requires --worker")
    output = (args.output or default_output(args.t5_precision)).resolve()
    reference = args.noise_reference.resolve() if args.noise_reference else None
    if args.check:
        _, _, report = preflight(output, args.t5_precision, reference)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("SIGTERM")))
    with tail.hybrid.lock(REPO / ".local/etri_60s_check.lock"):
        identity, signature, _ = preflight(output, args.t5_precision, reference)
        execute(output, identity, signature, args.prepare_text_only)


if __name__ == "__main__":
    main()
