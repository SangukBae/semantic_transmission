"""FC-LGVSC with Qwen3.5 proposals/captions, InternVL SKEM, and repaired decoding.

Prepared full-duration source videos are selected by inventory ID. Extraction
and reconstruction are separate resumable commands and use separate environments.
"""
import argparse
import json
import os
from pathlib import Path
import signal

from . import auto_extraction as extraction, qwen35_conditions as conditions
from . import condition_collision as collision, decoder_memory, generation_noise as noise
from . import precision_text_cache as text_cache, short_schedule, tail_reference as tail
from .artifacts import sha256, write_json
from .cli import settings
from .etri_60s_check import lock
from .hybrid_reconstruction import initialize as initialize_source, initialization_products, launch
from .qwen35_backend import MODEL_ID, REVISION
from .webvid_ablation import Stages, environment

REPO = extraction.REPO
MODULE = "semantic_transmission.fc_lgvsc_qwen35"
ETRI = "semantic_transmission.etri_60s"
DEFAULT_ROOT = REPO / "outputs/fc_lgvsc_qwen35_caption_repair_20261007"
POLICY = "fc_lgvsc_qwen35_nf4_v1"


def build_config(config, source, dest, root):
    template = extraction.read(Path(config["prepared_root"]) /
        "webvid/webvid_011_65f0e6cf/extraction/protocol.json")["config"]
    cfg = dict(template)
    for key in ("selector_checkpoint", "caption_checkpoint", "caption_bundle", "hybrid_selection_root"):
        cfg.pop(key, None)
    cfg.update(profile=POLICY, selector="qwen35_proposals_internvl_skem",
        caption_provider=conditions.PROVIDER, condition_extraction_root=str(root),
        condition_video_id=source["id"], condition_signature=config["signature"],
        condition_captions_sha256=sha256(dest / "captions.json"),
        input=source["normalized_video"], input_sha256=source["normalized_sha256"],
        frames=source["frames"], max_frames=source["frames"], fps=source["fps"],
        width=source["width"], height=source["height"], cpu_video_storage=True,
        semantic_clip_policy="frame_exact", official_semantic_clips=True,
        concatenation_policy="endpoint_exact")
    if config.get("caption_adapter"):
        cfg["caption_adapter"] = config["caption_adapter"]
    return cfg


def job_plan():
    transport = "semantic_transmission.codec_transport"
    return [
        ("input-audit", ETRI, ["input_audit.json"]),
        ("semantic-clips", ETRI, ["semantic_clips_audit.json", "data/clips"]),
        ("caption", MODULE, ["captions.json", "caption_sampling.json", "caption_provenance.json"]),
        ("flow", "semantic_transmission.workers", ["metadata_tx.json", "flow_sampling.json"]),
        ("send", transport, ["transmitter", "sender_accounting.json"]),
        ("channel", transport, ["received", "channel_accounting.json"]),
        ("receive", transport, ["receiver/frames", "receiver/metadata.csv", "receiver/decoder_inputs.json", "receiver_accounting.json"]),
        ("prepare-text", MODULE, ["receiver_policy.json", text_cache.REPORT]),
        ("reconstruct", MODULE, ["receiver/decoder_config.py", "receiver/reconstruction",
            "receiver/reference_trace.json", "receiver/tail_reference_trace.json",
            text_cache.TRACE, noise.TRACE, collision.TRACE, short_schedule.TRACE, decoder_memory.TRACE]),
        ("output-audit", MODULE, ["output_audit.json"]),
        ("evaluate", "semantic_transmission.research_quality", ["quality.json", "quality_delivered_mp4.csv", "quality_lossless_frames.csv"]),
        ("comparison", ETRI, [])]


def preflight(root, video_id, output):
    config, source, dest, bundle = conditions.validated(root, video_id)
    # Output must be outside all source/extraction trees.
    protected = [root, Path(config["prepared_root"]), Path(source["source_path"]),
                 Path(source["source_frames"]), Path(source["normalized_video"])]
    if any(output == p.resolve() or output in p.resolve().parents or p.resolve() in output.parents
           for p in protected):
        raise ValueError("reconstruction output overlaps protected source/extraction paths")
    keys = extraction.read(dest / "keyframes.json")["indices"]
    plan = collision.preview(keys, collision.upstream_functions(REPO))
    cfg = build_config(config, source, dest, root)
    for model in ("stdit", "vae", "t5", "vae2d"):
        if not Path(cfg["models"][model]).is_dir():
            raise FileNotFoundError(cfg["models"][model])
    for field in ("python", "channel_python"):
        if not Path(settings(REPO)[field]).is_file():
            raise FileNotFoundError(field)
    code = sorted(set(tail.CODE) | set(text_cache.CODE) | {
        "src/semantic_transmission/fc_lgvsc_qwen35.py", "src/semantic_transmission/qwen35_conditions.py",
        "src/semantic_transmission/condition_collision.py", "src/semantic_transmission/short_schedule.py",
        "src/semantic_transmission/decoder_memory.py",
        "src/semantic_transmission/generation_noise.py", "scripts/etri_precision_decoder.py",
        "scripts/etri_condition_fix_decoder.py", "scripts/etri_schedule_fix_decoder.py"})
    identity = dict(policy=POLICY, config=cfg, selection_sha256=sha256(dest / "selection.json"),
        source_metadata_sha256=next(v["prepared_sha256"] for v in config["videos"] if v["id"] == video_id),
        condition_plan=plan, code={p: sha256(REPO / p) for p in code})
    signature = extraction.digest(identity)
    receipt = output / "execution_protocol.json"
    if receipt.exists():
        if extraction.read(receipt) != dict(identity, signature=signature):
            raise ValueError("reconstruction inputs/code changed; choose a new output")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("unrecognized nonempty reconstruction output")
    return source, keys, identity, signature


def prepare_text(run):
    cfg = extraction.read(run / "run_config.json")
    keys = extraction.read(run / "keyframes.json")["indices"]
    contract = dict(policy=noise.POLICY, seed=cfg["seed"], segments=len(keys)-1, steps=cfg["steps"],
        inputs=extraction.digest({p: sha256(run / p) for p in
            ("receiver/metadata.csv", "receiver/decoder_inputs.json", "received/visual.c64")}),
        code=extraction.read(run.parent / "execution_protocol.json")["signature"], reference_policy=tail.POLICY)
    write_json(run / "receiver_policy.json", dict(policy=tail.POLICY, default_method=POLICY,
        reference_frames=17, reference_latents=5, alignment=5, text_encoder_policy=text_cache.POLICY,
        t5_precision="fp32", noise_contract=contract, noise_reference=None,
        condition_collision_policy=collision.POLICY, short_schedule_policy=short_schedule.POLICY))
    text_cache.prepare(run, REPO, tail.decoder_config(run), "fp32")


def reconstruct(run):
    from .decoder_runner import run as decode
    config = run / "receiver/decoder_config.py"
    config.write_text(tail.decoder_config(run))
    for p in ("receiver/reference_trace.json", "receiver/tail_reference_trace.json", collision.TRACE, short_schedule.TRACE):
        write_json(run / p, [])
    env = dict(os.environ, PYTHONPATH=str(REPO / ".local/vendor/Open-Sora"),
        ETRI_REFERENCE_TRACE=str(run / "receiver/reference_trace.json"),
        ETRI_TAIL_REFERENCE_TRACE=str(run / "receiver/tail_reference_trace.json"), ETRI_PRECISION_RUN=str(run))
    decode(run / "receiver/metadata.csv", run / "receiver/reconstruction", "key_frames_received",
        run / "receiver", decoder=REPO / "scripts/etri_schedule_fix_decoder.py", config=config, environment=env)


def audit(run):
    from .etri_60s import output_audit
    output_audit(run)
    keys = extraction.read(run / "keyframes.json")["indices"]
    policy = extraction.read(run / "receiver_policy.json")
    tail.validate_trace(extraction.read(run / "receiver/tail_reference_trace.json"), len(keys)-2)
    text_cache.validate_usage(run)
    noise.validate_trace(extraction.read(run / noise.TRACE), policy["noise_contract"], len(keys)-1, policy["noise_contract"]["steps"])
    collision.validate_trace(extraction.read(run / collision.TRACE),
                             extraction.read(run.parent / "execution_protocol.json")["condition_plan"])
    short_schedule.validate_trace(extraction.read(run / short_schedule.TRACE), keys, policy["noise_contract"]["steps"])
    result = extraction.read(run / "output_audit.json")
    result.update(automatic_keyframe_insertion=True, automatic_keyframe_insertion_stage="qwen35_internvl_selection",
                  caption_provider=conditions.PROVIDER)
    write_json(run / "output_audit.json", result)


def execute(root, video_id, output, stop_after=None):
    source, keys, identity, signature = preflight(root, video_id, output)
    output.mkdir(parents=True, exist_ok=True)
    with lock(REPO / ".local/etri_60s_check.lock"), lock(output / ".execution.lock"):
        receipt = output / "execution_protocol.json"
        if not receipt.exists():
            write_json(receipt, dict(identity, signature=signature))
        stages = Stages(output, signature)
        stages.step("initialize", ["run"], initialization_products(source["frames"]),
            lambda _: initialize_source(output, source, dict(indices=keys, selector=identity["config"]["selector"]), identity["config"]))
        local, run = settings(REPO), output / "run"
        env = dict(environment(identity["config"]["seed"]), PYTHONPATH=str(REPO / "src"), ETRI_RUN_SIGNATURE=signature)
        for stage, module, products in job_plan():
            python = local["channel_python"] if stage == "channel" else local["python"]
            args = ["--worker", stage, "--run-dir", str(run)] if module in (MODULE, ETRI) else [stage, str(run)]
            artifacts = [f"run/{p}" for p in products] if stage != "comparison" else ["comparison.mp4"]
            required = [*artifacts, f"resources/{stage}.json"]
            worker_env = dict(env, CUDA_VISIBLE_DEVICES="-1") if stage == "channel" else env
            stages.step(stage, required, required,
                lambda log, cmd=[python, "-m", module, *args], e=worker_env, s=stage:
                    launch(cmd, log, e, output / f"resources/{s}.json"))
            if stage == stop_after:
                print(json.dumps(dict(status="PREPARATION_COMPLETE", through=stage, output=str(output))))
                return
        audit_record, quality = extraction.read(run / "output_audit.json"), extraction.read(run / "quality.json")
        if (audit_record["status"] != "PASSED" or quality["status"] != "PASSED"
                or quality["video_sha256"] != audit_record["video_sha256"]):
            raise ValueError("reconstruction/quality evidence mismatch")
        result = dict(status="PASS_QWEN35_FC_LGVSC_RECONSTRUCTION", policy=POLICY,
            model_id=MODEL_ID, model_revision=REVISION, quantization="nf4",
            video_id=video_id, frames=source["frames"], fps=source["fps"], keyframes=len(keys), captions=len(keys)-1,
            video=audit_record["video"], metrics=quality["delivered_mp4"],
            independent_semantic_review="PENDING", hallucination_mitigation_verified=False,
            extraction_signature=identity["config"]["condition_signature"])
        write_json(output / "RESULT.json", result)
        print(json.dumps(result, ensure_ascii=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("init", "extract", "status", "check", "reconstruct"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--prepared", type=Path, default=extraction.DEFAULT_PREPARED)
    parser.add_argument("--base-caption", action="store_true",
        help="init only: explicitly use the historical base-Qwen caption draft")
    parser.add_argument("--video-id", action="append")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stop-after", choices=("caption", "receive", "prepare-text"))
    parser.add_argument("--worker", choices=("caption", "prepare-text", "reconstruct", "output-audit"))
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args(argv)
    if args.base_caption and args.action != "init":
        parser.error("--base-caption is only valid for init; existing configs are frozen")
    if args.worker:
        if not args.run_dir:
            parser.error("worker requires --run-dir")
        return {"caption": conditions.import_captions, "prepare-text": prepare_text,
                "reconstruct": reconstruct, "output-audit": audit}[args.worker](args.run_dir.resolve())
    if not args.action:
        parser.error("choose init, extract, status, check or reconstruct")
    root = args.root.resolve()
    if args.action == "init":
        from .qwen35_caption_adapter import default_adapter
        return extraction.initialize(root, args.prepared.resolve(), args.video_id,
            MODEL_ID, REVISION, "nf4", backend="qwen35",
            caption_adapter=None if args.base_caption else default_adapter())
    config = extraction.read(root / "run_config.json")
    if config.get("backend") != "qwen35":
        raise ValueError("this command requires a Qwen3.5 extraction root")
    if args.action == "status":
        print(json.dumps(extraction.read(root / "status.json"), ensure_ascii=False, indent=2))
        return
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("SIGTERM")))
    if args.action == "extract":
        try:
            extraction.run(root, config)
        except BaseException as exc:
            extraction.report(root, config, status="STOPPED", error=str(exc))
            raise
        if extraction.read(root / "status.json")["status"] != "COMPLETE":
            raise RuntimeError("extraction incomplete; inspect per-video errors")
        return
    if not args.video_id or len(args.video_id) != 1:
        parser.error("check/reconstruct requires exactly one --video-id")
    video_id = args.video_id[0]
    output = (args.output or root.with_name(root.name + "_reconstruction") / video_id).resolve()
    if args.action == "check":
        source, keys, _, _ = preflight(root, video_id, output)
        print(json.dumps(dict(status="READY", frames=source["frames"], keyframes=len(keys),
            model_id=MODEL_ID, caption_provider=conditions.PROVIDER, output=str(output), model_inference_started=False)))
        return
    execute(root, video_id, output, args.stop_after)


if __name__ == "__main__":
    main()
