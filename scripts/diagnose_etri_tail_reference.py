"""Compare full previous references with their last 17 frames, retaining align=5."""
import argparse
import os
from pathlib import Path
import shutil

from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.cli import settings
from semantic_transmission.etri_60s_check import lock, run_command
from semantic_transmission.webvid5 import execution_identity, fingerprint, read_json
from semantic_transmission.webvid_ablation import Stages, environment, snapshot

from diagnose_etri_conditioning import WINDOWS

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "outputs/etri_conditioning_diagnosis_20260928"
OUTPUT = REPO / "outputs/etri_tail_reference_20260928"
CASES = ("baseline", "tail17")
SCRIPTS = ["scripts/diagnose_etri_tail_reference.py", "scripts/etri_tail_reference_probe.py",
           "scripts/diagnose_etri_conditioning.py", "scripts/etri_conditioning_evidence.py"]


def verify_source(source):
    if read_json(source / "RESULT.json")["status"] != "PAIRED_DIAGNOSTICS_COMPLETE":
        raise ValueError("previous diagnostic run is incomplete")
    receipts = {}
    for path in sorted((source / "stages").glob("*.json")):
        value = read_json(path)
        if value["status"] != "PASSED" or snapshot(source, value["required"]) != value["artifacts"]:
            raise ValueError(f"previous diagnostic artifacts changed: {path}")
        receipts[path.stem] = sha256(path)
    if len(receipts) != 13:
        raise ValueError("expected all 13 previous diagnostic stages")
    return receipts


def initialize(root, source):
    previous = read_json(source / "pairs.json")
    pairs = []
    for name in WINDOWS:
        origin = source / name / "baseline"
        shared = root / "inputs" / name
        shared.mkdir(parents=True)
        shutil.copyfile(origin / "data/normalized.mp4", shared / "source.mp4")
        record = next(p for p in previous if p["window"] == name and p["case"] == "baseline")
        for case in CASES:
            run = root / name / case
            for relative in ("data", "receiver/frames"):
                shutil.copytree(origin / relative, run / relative)
            for relative in ("keyframes.json", "receiver/metadata.csv", "receiver/decoder_inputs.json"):
                shutil.copyfile(origin / relative, run / relative)
            cfg = dict(read_json(origin / "run_config.json"), profile="etri_tail_reference_v1",
                       input=str(shared / "source.mp4"), diagnostic_align=5, reference_policy=case)
            write_json(run / "run_config.json", cfg)
            pairs.append(dict(record, case=case, align=5, reference_policy=case))
    write_json(root / "pairs.json", pairs)


def reconstruct(run):
    from semantic_transmission.codec_transport import decoder_config_text
    from semantic_transmission.decoder_runner import run as decode
    cfg = read_json(run / "run_config.json")
    text = decoder_config_text(cfg, REPO, read_json(run / "receiver/decoder_inputs.json"))
    text += "\ncpu_video_storage=True\nconditioning_alignment='official_release'\nalign=5\n"
    path = run / "receiver/decoder_config.py"
    path.write_text(text)
    env = dict(os.environ, PYTHONPATH=str(REPO / ".local/vendor/Open-Sora"),
        ETRI_CONDITION_TRACE=str(run / "receiver/conditioning_trace.json"),
        ETRI_REFERENCE_MODE="record" if run.name == "baseline" else "tail17",
        ETRI_REFERENCE_BANK=str(run.parent / "baseline/receiver/replay"))
    decode(run / "receiver/metadata.csv", run / "receiver/reconstruction", "key_frames_received",
           run / "receiver", decoder=REPO / "scripts/etri_tail_reference_probe.py", config=path, environment=env)


def validate_pair(a, b, key_count):
    if len(a["mask"]) != key_count-1 or len(b["mask"]) != key_count-1:
        raise ValueError("incomplete paired mask trace")
    for x, y in zip(a["mask"], b["mask"]):
        if x["align"] != 5 or y["align"] != 5:
            raise ValueError("alignment changed")
        for field in ("loop", "noise_sha256", "sampling_rng_sha256"):
            if x[field] != y[field]:
                raise ValueError(f"paired {field} differs")
        rx = lambda t: [v["reference_sha256"] for v in t["active"] if v["reference"] < key_count]
        if rx(x) != rx(y):
            raise ValueError("received keyframe latents differ")
    if len(a["references"]) != key_count-2 or len(b["references"]) != key_count-2:
        raise ValueError("incomplete reference trace")
    for x, y in zip(a["references"], b["references"]):
        if x["reference_rng_sha256"] != y["reference_rng_sha256"] or x["loop"] != y["loop"]:
            raise ValueError("reference RNG differs")
        if y["encoded_pixel_shape"][2] != 17 or y["latent_shape"][1] != 5:
            raise ValueError("tail reference is not a single 17-frame/5-latent block")


def summarize(root, source):
    rows = []
    for name, keys in WINDOWS.items():
        traces = {c: read_json(root / name / c / "receiver/conditioning_trace.json") for c in CASES}
        validate_pair(traces["baseline"], traces["tail17"], len(keys))
        values = {c: read_json(root / name / c / "quality.json")["delivered_mp4"] for c in CASES}
        relative = Path(name) / "baseline/receiver/reconstruction/sample_0000.mp4"
        rows.append({"window": name, "metrics": values,
            "initial_noise_and_sampling_rng_matched": True, "reference_rng_matched": True,
            "received_keyframe_latents_matched": True, "align": 5,
            "baseline_mp4_matches_previous_experiment": sha256(root / relative) == sha256(source / relative),
            "delta_after_minus_before": {k: values['tail17'][k]-values['baseline'][k]
                for k in ("psnr_db", "ssim", "lpips_vgg", "clip", "dists")}})
    write_json(root / "RESULT.json", {"status": "PAIRED_TAIL_REFERENCE_COMPLETE", "windows": rows,
        "intervention": "Previous reference cropped to its last 17 frames before VAE encoding; align=5 unchanged.",
        "random_control": "Baseline noise tensors and RNG states replayed at reference encoding and sampling.",
        "scope": "Three restarted short contexts from one development video/seed; not full60s mitigation.",
        "additional_channel_uses": 0, "independent_semantic_review": "PENDING"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--worker", choices=["reconstruct"])
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if args.worker:
        reconstruct(args.run_dir.resolve())
        return
    root, source = args.output.resolve(), args.source.resolve()
    receipts = verify_source(source)
    cfg = read_json(source / "people_10s/baseline/run_config.json")
    identity = {"source": str(source), "source_receipts": receipts,
                "windows": WINDOWS, "cases": CASES, "execution": execution_identity(REPO, cfg),
                "scripts": {p: sha256(REPO / p) for p in SCRIPTS}}
    signature = fingerprint(identity)
    root.mkdir(parents=True, exist_ok=True)
    with lock(REPO / ".local/etri_60s_check.lock"), lock(root / ".lock"):
        path = root / "protocol.json"
        if path.exists() and read_json(path)["signature"] != signature:
            raise ValueError("experiment changed; choose a new output directory")
        if not path.exists():
            write_json(path, dict(identity, signature=signature))
        pipeline = Stages(root, signature)
        required = lambda: [str(p.relative_to(root)) for d in [root / 'inputs', *[root / w for w in WINDOWS]]
                            for p in d.rglob('*') if p.is_file()] + ['pairs.json']
        pipeline.step("initialize", ["inputs", "pairs.json", *WINDOWS], required, lambda _: initialize(root, source))
        local, env = settings(REPO), environment(cfg["seed"])
        for window in WINDOWS:
            for case in CASES:
                relative, run = f"{window}/{case}", root / window / case
                for stage in ("reconstruct", "evaluate"):
                    name = f"{window}.{case}.{stage}"
                    command = ([local['python'], str(Path(__file__).resolve()), '--worker', 'reconstruct', '--run-dir', str(run)]
                               if stage == 'reconstruct' else [local['python'], '-m', 'semantic_transmission.research_quality', 'evaluate', str(run)])
                    files = (["receiver/reconstruction", "receiver/decoder_config.py", "receiver/conditioning_trace.json"]
                             + (["receiver/replay"] if case == "baseline" else []) if stage == 'reconstruct'
                             else ["quality.json", "quality_lossless_frames.csv", "quality_delivered_mp4.csv"])
                    paths = [f"{relative}/{p}" for p in files]
                    metrics, timing, progress = f"resources/{name}.json", f"resources/{name}.time.txt", f"progress/{name}.json"
                    pipeline.step(name, paths + [metrics, timing, progress], paths + [metrics, timing],
                        lambda log, cmd=command, m=metrics, p=progress: run_command(REPO, cmd, log, env, root/m, root/p))
        summarize(root, source)
        from etri_conditioning_evidence import build
        for name in WINDOWS:
            build(root, name, variant="tail17", variant_label="TAIL 17 - restarted context")
        print(f"PAIRED_TAIL_REFERENCE_COMPLETE: {root}", flush=True)


if __name__ == "__main__":
    main()
