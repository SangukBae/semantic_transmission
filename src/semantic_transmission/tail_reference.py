"""Receiver-only tail17 revision, reusing frozen v2 received inputs.

No inference or writes occur with --check. Old runners and receipts stay valid.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess

from . import caption_revision, hybrid_reconstruction as hybrid
from .artifacts import sha256, write_json
from .temporal import conditioning_indices
from .webvid5 import fingerprint, read_json
from .webvid_ablation import Stages, environment, snapshot

REPO = hybrid.REPO
SOURCE = caption_revision.paths(caption_revision.DEFAULT_ROOT)[2]
OUTPUT = SOURCE.with_name(SOURCE.name + "_tail17")
T5_OUTPUT = SOURCE.with_name(SOURCE.name + "_tail17_t5gpu")
MODULE = "semantic_transmission.tail_reference"
POLICY = "last_17_frames_before_vae"
INPUTS = (
    "run_config.json", "keyframes.json", "input_audit.json", "captions.json",
    "caption_provenance.json", "metadata_tx.json", "channel_accounting.json",
    "data/normalized.mp4", "received/visual.c64", "received/metadata.bin",
    "receiver/metadata.csv", "receiver/decoder_inputs.json", "receiver/frames",
)
CODE = (
    "src/semantic_transmission/tail_reference.py", "scripts/etri_tail17_decoder.py",
    "scripts/reconstruct_tail17.sh", "scripts/reconstruct.sh",
    "configs/official_opensora.py", "src/semantic_transmission/caption_revision.py",
    ".local/vendor/Open-Sora/opensora/utils/inference_utils.py",
    ".local/vendor/Open-Sora/opensora/models/vae/vae.py",
    ".local/vendor/Open-Sora/opensora/models/vae/vae_temporal.py",
    ".local/vendor/Open-Sora/opensora/models/vae/utils.py",
    ".local/vendor/Open-Sora/opensora/schedulers/rf/__init__.py",
)


def append_tail17(original, vae, video, refs, strategies, loop_i, length, edit):
    """Crop pixels BEFORE encoding; leave the existing mask/alignment unchanged."""
    if length != 5 or video.ndim != 5:
        raise ValueError("tail17 requires a B,C,T,H,W video and five reference latents")
    indices = conditioning_indices(video.shape[2], 17)
    tail = video[:, :, indices]
    before = [len(row or []) for row in refs]
    result = original(vae, tail, refs, strategies, loop_i, length, edit)
    after = [len(row) for row in result[0]]
    shapes = [list(row[-1].shape) for row in result[0]]
    if after != [n + 1 for n in before] or any(shape[1] != 5 for shape in shapes):
        raise ValueError("tail17 reference must append exactly one five-latent block per batch")
    record = dict(loop=loop_i, input_frames=video.shape[2], source_indices=indices,
                  encoded_frames=tail.shape[2], latent_shapes=shapes, policy=POLICY)
    return result, record


def validate_trace(records, transfers):
    if [row["loop"] for row in records] != list(range(1, transfers + 1)):
        raise ValueError("missing tail17 transfer records")
    for row in records:
        if (row["policy"] != POLICY or row["encoded_frames"] != 17
                or row["source_indices"] != conditioning_indices(row["input_frames"], 17)
                or not row["latent_shapes"] or any(s[1] != 5 for s in row["latent_shapes"])):
            raise ValueError("reference is not the last 17 frames encoded as five latents")


def validate_destination(output, source):
    # Keep this prepared-input revision in its own sibling folder.
    if output.parent != source.parent or output == source:
        raise ValueError("output must be a separate sibling of the frozen v2 result")
    path = output / "execution_protocol.json"
    if output.exists() and not path.exists():
        raise ValueError("unrecognized existing output; choose a new --output")


def preflight(output=OUTPUT, t5_cache=False):
    validate_destination(output, SOURCE)
    caption_revision.preflight()
    result = read_json(SOURCE / "RESULT.json")
    if result["status"] != "PASS_60S_HYBRID_RECONSTRUCTION":
        raise ValueError("complete v2 reconstruction required")
    receipts = {}
    for path in sorted((SOURCE / "stages").glob("*.json")):
        record = read_json(path)
        if record["status"] != "PASSED" or snapshot(SOURCE, record["required"]) != record["artifacts"]:
            raise ValueError(f"frozen v2 artifacts changed: {path}")
        receipts[path.name] = sha256(path)
    if set(receipts) != {f"{name}.json" for name in (
            "initialize", *[s for s, _, _ in hybrid.job_plan(True)], "caption-revision-comparison")}:
        raise ValueError("incomplete v2 stage receipts")
    old = read_json(SOURCE / "execution_protocol.json")
    code = sorted(set(old["code"]) | set(CODE))
    identity = dict(version=1, policy=POLICY, source=str(SOURCE), source_receipts=receipts,
        source_result_sha256=sha256(SOURCE / "RESULT.json"),
        inputs=snapshot(SOURCE / "run", INPUTS),
        code={name: sha256(REPO / name) for name in code})
    if t5_cache:
        import importlib.metadata
        from . import text_embedding_cache as cache
        cfg = read_json(SOURCE / "run/run_config.json")
        identity["text_encoder"] = dict(policy=cache.POLICY, device="cuda", dtype="bf16",
            model_inventory=cache.model_inventory(cfg["models"]["t5"]),
            packages={n: importlib.metadata.version(n) for n in
                      ("torch", "transformers", "tokenizers", "ftfy", "beautifulsoup4")})
        identity["code"].update({name: sha256(REPO / name) for name in cache.CODE})
    signature = fingerprint(identity)
    path = output / "execution_protocol.json"
    if path.exists() and read_json(path) != dict(identity, signature=signature):
        raise ValueError("tail17 inputs/code changed; preserve this run and choose a new --output")
    cfg = read_json(SOURCE / "run/run_config.json")
    report = dict(status="READY_FOR_USER_EXECUTION", reference_policy=POLICY,
        frames=cfg["frames"], fps=cfg["fps"], keyframes=result["keyframes"],
        reused_inputs="v2 received keyframes, captions, motion and channel symbols",
        runs_selector=False, runs_caption_model=False, reruns_transmission=False,
        output=str(output), expected_review=str(output / "review.html"),
        alignment=5, first_segment_collision_fixed=False,
        diffusion_noise_tensor_identity_verified=False, quality_effect_verified=False)
    if t5_cache:
        report.update(text_encoder_policy=cache.POLICY, t5_prepared_before_decoder=True,
            t5_model_loaded_in_decoder=False, legacy_cpu_fp32_equivalence_verified=False)
    return identity, signature, report


def initialize(source, output, expected, t5_cache=False):
    if snapshot(source / "run", INPUTS) != expected:
        raise ValueError("v2 receiver inputs changed before initialization")
    # Copy the small metadata and received frames; link only immutable large inputs.
    for name in INPUTS:
        src, dest = source / "run" / name, output / "run" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dest)
        elif name in {"data/normalized.mp4", "received/visual.c64"}:
            dest.symlink_to(src.resolve())
        else:
            shutil.copyfile(src, dest)
    if snapshot(output / "run", INPUTS) != expected:
        raise ValueError("copied receiver inputs differ from v2")
    policy = dict(policy=POLICY, reference_frames=17, reference_latents=5, alignment=5,
                  source=str(source), additional_channel_uses=0)
    if t5_cache:
        from .text_embedding_cache import POLICY as text_policy
        policy["text_encoder_policy"] = text_policy
    write_json(output / "run/receiver_policy.json", policy)


def decoder_config(run):
    from .codec_transport import decoder_config_text
    cfg = read_json(run / "run_config.json")
    if read_json(run / "receiver_policy.json")["policy"] != POLICY:
        raise ValueError("tail17 receiver policy missing")
    text = decoder_config_text(cfg, REPO, read_json(run / "receiver/decoder_inputs.json"))
    text += "\ncpu_video_storage=True\nconditioning_alignment='official_release'\nalign=5\n"
    text += f"reference_policy={POLICY!r}\n"
    return text


def prepare_text(run):
    from .text_embedding_cache import prepare, POLICY as text_policy
    if read_json(run / "receiver_policy.json").get("text_encoder_policy") != text_policy:
        raise ValueError("GPU T5 preparation not enabled for this run")
    prepare(run, REPO, decoder_config(run))


def reconstruct(run):
    from .decoder_runner import run as decode
    text = decoder_config(run)
    config = run / "receiver/decoder_config.py"
    config.write_text(text)
    trace = run / "receiver/reference_trace.json"
    tail_trace = run / "receiver/tail_reference_trace.json"
    write_json(trace, [])
    write_json(tail_trace, [])
    env = dict(os.environ, PYTHONPATH=str(REPO / ".local/vendor/Open-Sora"),
               ETRI_REFERENCE_TRACE=str(trace), ETRI_TAIL_REFERENCE_TRACE=str(tail_trace))
    decoder = REPO / "scripts/etri_tail17_decoder.py"
    if read_json(run / "receiver_policy.json").get("text_encoder_policy"):
        from .text_embedding_cache import REPORT, TRACE, POLICY as text_policy
        if read_json(run / REPORT)["policy"] != text_policy:
            raise ValueError("GPU T5 preparation missing")
        env.update(ETRI_TEXT_EMBEDDINGS=str(run / REPORT), ETRI_TEXT_EMBEDDING_TRACE=str(run / TRACE))
        decoder = REPO / "scripts/etri_t5_cached_decoder.py"
    decode(run / "receiver/metadata.csv", run / "receiver/reconstruction", "key_frames_received",
           run / "receiver", decoder=decoder, config=config, environment=env)


def audit(run):
    hybrid.hybrid_output_audit(run)
    keys = read_json(run / "keyframes.json")["indices"]
    validate_trace(read_json(run / "receiver/tail_reference_trace.json"), len(keys) - 2)
    if read_json(run / "receiver_policy.json").get("text_encoder_policy"):
        from .text_embedding_cache import validate_usage
        validate_usage(run)


def comparison(output):
    run, old = output / "run", SOURCE / "run"
    paired = caption_revision.verify_pair(SOURCE, output)
    if sha256(old / "receiver/metadata.csv") != sha256(run / "receiver/metadata.csv"):
        raise ValueError("tail17 comparison captions/motion changed")
    paired.update(reference_policy_changed=True, scope="Same received inputs and seed; full-run noise is not replayed.")
    quality = [read_json(r / "quality.json") for r in (old, run)]
    videos = [run / "data/normalized.mp4", old / "receiver/reconstruction/sample_0000.mp4",
              run / "receiver/reconstruction/sample_0000.mp4"]
    if quality[0]["source_sha256"] != quality[1]["source_sha256"]:
        raise ValueError("quality sources differ")
    for video, q in zip(videos[1:], quality):
        if q["status"] != "PASSED" or sha256(video) != q["video_sha256"]:
            raise ValueError("comparison quality does not match video")
    cfg = read_json(run / "run_config.json")
    target = output / "comparison.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", *[x for v in videos for x in ("-i", str(v))],
        "-filter_complex", "[0:v][1:v][2:v]hstack=inputs=3[v]", "-map", "[v]", "-an",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p", str(target)], check=True)
    hybrid.pts_audit(target, cfg["frames"], cfg["fps"])
    channel = read_json(run / "channel_accounting.json")
    result = dict(status="PASS_TAIL17_RECONSTRUCTION_REVIEW_PENDING", policy=POLICY, pairing=paired,
        frames=cfg["frames"], fps=cfg["fps"], quality_before=quality[0]["delivered_mp4"],
        quality_after=quality[1]["delivered_mp4"], video_sha256=quality[1]["video_sha256"],
        comparison_sha256=sha256(target), channel_uses=channel["total_complex_channel_uses"],
        additional_channel_uses=0, first_segment_collision_fixed=False,
        hallucination_review="PENDING", hallucination_mitigation_verified=False)
    extra_note = ""
    text_policy = read_json(run / "receiver_policy.json").get("text_encoder_policy")
    if text_policy:
        from .text_embedding_cache import REPORT
        report = read_json(run / REPORT)
        result.update(text_encoder_policy=text_policy, text_cache_hits=report["cache_hits"],
                      text_cache_misses=report["cache_misses"], legacy_cpu_fp32_equivalence_verified=False)
        paired["text_encoder_precision_changed"] = True
        paired["scope"] = "Same received inputs; tail17 references and GPU BF16 T5 changed. Noise is not replayed."
        extra_note = '<p>T5를 GPU BF16으로 사전 계산해 재사용했습니다. 기존 CPU FP32와 수치·화질 동일성은 미검증입니다.</p>'
    write_json(output / "RESULT.json", result)
    rows = ''.join(f'<tr><td>{k}</td><td>{result["quality_before"][k]:.4f}</td>'
                   f'<td>{result["quality_after"][k]:.4f}</td></tr>'
                   for k in ("psnr_db", "ssim", "lpips_vgg", "clip", "dists"))
    (output / "review.html").write_text('<!doctype html><html lang="ko"><meta charset="utf-8">'
        '<title>v2 참조 구간 변경</title><style>body{font:16px system-ui;margin:24px}video{width:100%}'
        'td,th{padding:10px}</style><h1>원본 / 기존 v2 / 마지막 17프레임 참조</h1>'
        '<video controls preload="metadata" src="comparison.mp4"></video>'
        '<p>키프레임·캡션·움직임·수신 자료 동일, 추가 전송량 0. '
        '참조 변경에 따라 난수 소비가 달라져 초기 생성 잡음의 완전 일치는 보장하지 않습니다.</p>'
        '<p>첫 구간 조건 충돌은 별도 문제로 남아 있습니다. 의미 오류 검수·완화 효과는 미검증입니다.</p>'+extra_note+
        '<table><tr><th>지표</th><th>기존 v2</th><th>마지막 17프레임</th></tr>'+rows+
        '</table><a href="RESULT.json">수치·검증 범위</a></html>')


def execute(output, identity, signature, prepare_only=False):
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "execution_protocol.json", dict(identity, signature=signature))
    stages = Stages(output, signature)
    t5_cache = "text_encoder" in identity
    stages.step("initialize", ["run"], [f"run/{p}" for p in INPUTS] + ["run/receiver_policy.json"],
                lambda _: initialize(SOURCE, output, identity["inputs"], t5_cache=t5_cache))
    cfg = read_json(output / "run/run_config.json")
    local = hybrid.settings(REPO)
    env = environment(cfg["seed"])
    env["PYTHONPATH"] = str(REPO / "src")
    jobs = [
        ("reconstruct", MODULE, ["receiver/decoder_config.py", "receiver/reconstruction",
                                 "receiver/reference_trace.json", "receiver/tail_reference_trace.json"]),
        ("audit", MODULE, ["output_audit.json"]),
        ("evaluate", "semantic_transmission.research_quality",
         ["quality.json", "quality_delivered_mp4.csv", "quality_lossless_frames.csv"]),
    ]
    if t5_cache:
        from .text_embedding_cache import REPORT, TRACE
        jobs.insert(0, ("prepare-text", MODULE, [REPORT]))
        jobs[1][2].append(TRACE)
    elif prepare_only:
        raise ValueError("--prepare-text-only requires --t5-cache")
    for name, module, products in jobs:
        args = ["--worker", name, "--run-dir", str(output / "run")] if module == MODULE else [name, str(output / "run")]
        command = [local["python"], "-m", module, *args]
        resource = f"resources/{name}.json"
        required = [f"run/{p}" for p in products] + [resource]
        stages.step(name, required, required,
            lambda log, cmd=command, res=resource: hybrid.launch(cmd, log, env, output / res))
        if name == "prepare-text" and prepare_only:
            print(f"PREPARED T5 ONLY: {output / 'run' / REPORT}", flush=True)
            return
    products = ["comparison.mp4", "RESULT.json", "review.html"]
    stages.step("comparison", products, products, lambda _: comparison(output))
    print(f"COMPLETE TAIL17: {output / 'review.html'}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="read-only readiness check; no model execution")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--t5-cache", action="store_true", help="prepare T5 on GPU and reuse saved text conditions")
    parser.add_argument("--prepare-text-only", action="store_true", help="prepare text cache without video generation")
    parser.add_argument("--worker", choices=("prepare-text", "reconstruct", "audit"), help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        if args.check or args.run_dir is None:
            parser.error("worker requires --run-dir and cannot be combined with --check")
        return {"prepare-text": prepare_text, "reconstruct": reconstruct, "audit": audit}[args.worker](args.run_dir.resolve())
    if args.run_dir is not None:
        parser.error("--run-dir requires --worker")
    if args.prepare_text_only and not args.t5_cache:
        parser.error("--prepare-text-only requires --t5-cache")
    output = (args.output or (T5_OUTPUT if args.t5_cache else OUTPUT)).resolve()
    def check():
        return preflight(output, t5_cache=True) if args.t5_cache else preflight(output)
    if args.check:
        _, _, report = check()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("SIGTERM")))
    with hybrid.lock(REPO / ".local/etri_60s_check.lock"):
        identity, signature, _ = check()
        execute(output, identity, signature, prepare_only=args.prepare_text_only)


if __name__ == "__main__":
    main()
