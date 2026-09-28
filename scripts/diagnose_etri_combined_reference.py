"""Four-condition short diagnostic: reference tail cropping x mask rounding."""
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
from diagnose_etri_tail_reference import validate_pair

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "outputs/etri_tail_reference_20260928"
OUTPUT = REPO / "outputs/etri_combined_reference_20260928"
CASES = ("baseline", "no_rounding", "tail17", "combined")
NEW_CASES = ("no_rounding", "combined")
SCRIPTS = ["scripts/diagnose_etri_combined_reference.py", "scripts/etri_combined_reference_probe.py",
           "scripts/etri_combined_reference_evidence.py", "scripts/etri_tail_reference_probe.py",
           "scripts/diagnose_etri_tail_reference.py", "scripts/diagnose_etri_conditioning.py",
           "scripts/etri_conditioning_evidence.py", "scripts/check_etri_combined_reference.sh"]


def verify_source(source):
    if read_json(source / "RESULT.json")["status"] != "PAIRED_TAIL_REFERENCE_COMPLETE":
        raise ValueError("tail-reference source is incomplete")
    receipts = {}
    for path in sorted((source / "stages").glob("*.json")):
        value = read_json(path)
        if value["status"] != "PASSED" or snapshot(source, value["required"]) != value["artifacts"]:
            raise ValueError(f"source artifacts changed: {path}")
        receipts[path.stem] = sha256(path)
    if len(receipts) != 13:
        raise ValueError("expected 13 source stages")
    for name, keys in WINDOWS.items():
        validate_pair(*(read_json(source / name / c / "receiver/conditioning_trace.json")
                        for c in ("baseline", "tail17")), len(keys))
    return receipts


def initialize(root, source):
    previous = read_json(source / "pairs.json")
    shutil.copytree(source / "inputs", root / "inputs")
    pairs = []
    for name in WINDOWS:
        origin = source / name / "baseline"
        record = next(p for p in previous if p["window"] == name and p["case"] == "baseline")
        for case in CASES:
            run = root / name / case
            if case not in NEW_CASES:
                # Independent copies: never write into the completed control runs.
                shutil.copytree(source / name / case, run)
            else:
                for relative in ("data", "receiver/frames"):
                    shutil.copytree(origin / relative, run / relative)
                for relative in ("keyframes.json", "receiver/metadata.csv", "receiver/decoder_inputs.json"):
                    shutil.copyfile(origin / relative, run / relative)
                cfg = dict(read_json(origin / "run_config.json"), profile="etri_combined_reference_v1",
                           input=str(root / "inputs" / name / "source.mp4"), diagnostic_align=None,
                           reference_policy="tail17" if case == "combined" else "full")
                write_json(run / "run_config.json", cfg)
            pairs.append(dict(record, case=case, align=None if case in NEW_CASES else 5,
                reference_policy="tail17" if case in {"combined", "tail17"} else "full",
                reused_from=str(source / name / case) if case not in NEW_CASES else None))
    write_json(root / "pairs.json", pairs)


def reconstruct(run):
    from semantic_transmission.codec_transport import decoder_config_text
    from semantic_transmission.decoder_runner import run as decode
    if run.name not in NEW_CASES:
        raise ValueError("completed controls must be reused")
    cfg = read_json(run / "run_config.json")
    text = decoder_config_text(cfg, REPO, read_json(run / "receiver/decoder_inputs.json"))
    # Keep official_release: endpoint_exact would silently crop the full-reference control.
    text += "\ncpu_video_storage=True\nconditioning_alignment='official_release'\nalign=None\n"
    path = run / "receiver/decoder_config.py"
    path.write_text(text)
    env = dict(os.environ, PYTHONPATH=str(REPO / ".local/vendor/Open-Sora"),
        ETRI_CONDITION_TRACE=str(run / "receiver/conditioning_trace.json"),
        ETRI_FACTORIAL_CASE=run.name,
        ETRI_REFERENCE_BANK=str(run.parent / "baseline/receiver/replay"))
    decode(run / "receiver/metadata.csv", run / "receiver/reconstruction", "key_frames_received",
           run / "receiver", decoder=REPO / "scripts/etri_combined_reference_probe.py", config=path, environment=env)


def validate_conditions(traces, key_count):
    baseline = traces["baseline"]
    for case in CASES:
        trace = traces[case]
        if len(trace["mask"]) != key_count-1 or len(trace["references"]) != key_count-2:
            raise ValueError("incomplete factorial trace")
        align = None if case in NEW_CASES else 5
        for loop, (a, b) in enumerate(zip(baseline["mask"], trace["mask"])):
            if b["align"] != align or b["loop"] != loop:
                raise ValueError("alignment or loop differs from factorial design")
            for field in ("noise_shape", "noise_sha256", "sampling_rng_sha256"):
                if a[field] != b[field]:
                    raise ValueError(f"unmatched {field}: {case}")
            keys = lambda t: [(v["reference"], v["reference_sha256"]) for v in t["active"]
                              if v["reference"] < key_count]
            if keys(a) != keys(b) or not keys(a):
                raise ValueError("received keyframe latents differ or are missing")
            if align is None and any(v["actual_ref_start"] != v["requested_ref_start"] or
                    v["actual_target_start"] != v["requested_target_start"] for v in b["active"]):
                raise ValueError("rounding remained active")
        for loop, (a, b) in enumerate(zip(baseline["references"], trace["references"]), 1):
            if b["loop"] != loop or a["reference_rng_sha256"] != b["reference_rng_sha256"]:
                raise ValueError("reference RNG or loop differs")
            if a["input_pixel_shape"] != b["input_pixel_shape"]:
                raise ValueError("previous generated length changed")
            if case in {"tail17", "combined"}:
                if b["encoded_pixel_shape"][2] != 17 or b["latent_shape"][1] != 5:
                    raise ValueError("tail reference length differs")
            elif b["encoded_pixel_shape"] != b["input_pixel_shape"]:
                raise ValueError("full reference was unexpectedly cropped")


def verify_inputs(root):
    values = {}
    for name in WINDOWS:
        base = root / name / "baseline"
        paths = ["data/normalized.mp4", "keyframes.json", "receiver/metadata.csv", "receiver/decoder_inputs.json",
                 "receiver/frames"]
        expected = snapshot(base, paths)
        for case in CASES:
            if snapshot(root / name / case, paths) != expected:
                raise ValueError(f"received/source inputs differ: {name}/{case}")
        values[name] = expected
    return values


def summarize(root, source):
    inputs = verify_inputs(root)
    rows = []
    previous = Path(read_json(source / "protocol.json")["source"])
    for name, keys in WINDOWS.items():
        traces = {c: read_json(root / name / c / "receiver/conditioning_trace.json") for c in CASES}
        validate_conditions(traces, len(keys))
        values = {c: read_json(root / name / c / "quality.json")["delivered_mp4"] for c in CASES}
        rows.append({"window": name, "metrics": values,
            "noise_sampling_rng_reference_rng_and_received_latents_matched": True,
            "replayed_no_rounding_matches_previous_mp4": sha256(root/name/"no_rounding/receiver/reconstruction/sample_0000.mp4")
                == sha256(previous/name/"no_rounding/receiver/reconstruction/sample_0000.mp4"),
            "interaction_combined_minus_tail_minus_rounding_plus_baseline": {
                m: values["combined"][m]-values["tail17"][m]-values["no_rounding"][m]+values["baseline"][m]
                for m in ("psnr_db", "ssim", "lpips_vgg", "clip", "dists")}})
    write_json(root / "input_verification.json", {"status": "PASSED", "identical_inputs": inputs})
    write_json(root / "RESULT.json", {"status": "PAIRED_COMBINED_REFERENCE_COMPLETE", "windows": rows,
        "cases": {c: {"align": None if c in NEW_CASES else 5,
                      "reference": "tail17" if c in {"tail17", "combined"} else "full"} for c in CASES},
        "random_control": "Saved baseline noise and pre-reference/post-noise RNG replayed in all new cases.",
        "scope": "Three restarted short windows from one development video/seed; not full60s mitigation.",
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
    if root == source or source in root.parents or root in source.parents:
        raise ValueError("output must be separate from source")
    root.mkdir(parents=True, exist_ok=True)
    with lock(REPO / ".local/etri_60s_check.lock"), lock(root / ".lock"):
        receipts = verify_source(source)
        cfg = read_json(source / "people_10s/baseline/run_config.json")
        execution = execution_identity(REPO, cfg)
        if execution != read_json(source / "protocol.json")["execution"]:
            raise ValueError("decoder/model/environment changed; cannot reuse the saved controls")
        identity = {"source": str(source), "source_receipts": receipts,
            "source_protocol_sha256": sha256(source / "protocol.json"),
            "windows": WINDOWS, "cases": CASES, "execution": execution,
            "scripts": {p: sha256(REPO / p) for p in SCRIPTS}}
        signature = fingerprint(identity)
        protocol = root / "protocol.json"
        if protocol.exists() and read_json(protocol)["signature"] != signature:
            raise ValueError("experiment changed; choose a new output directory")
        if not protocol.exists():
            write_json(protocol, dict(identity, signature=signature))
        pipeline = Stages(root, signature)
        required = lambda: [str(p.relative_to(root)) for d in [root/'inputs', *[root/w for w in WINDOWS]]
                            for p in d.rglob('*') if p.is_file()] + ['pairs.json']
        pipeline.step("initialize", ["inputs", "pairs.json", *WINDOWS], required, lambda _: initialize(root, source))
        verify_inputs(root)
        local, env = settings(REPO), environment(cfg["seed"])
        for window in WINDOWS:
            for case in NEW_CASES:
                relative, run = f"{window}/{case}", root/window/case
                for stage in ("reconstruct", "evaluate"):
                    name = f"{window}.{case}.{stage}"
                    cmd = ([local['python'], str(Path(__file__).resolve()), '--worker', 'reconstruct', '--run-dir', str(run)]
                           if stage == 'reconstruct' else [local['python'], '-m', 'semantic_transmission.research_quality', 'evaluate', str(run)])
                    files = (["receiver/reconstruction", "receiver/decoder_config.py", "receiver/conditioning_trace.json"]
                             if stage == 'reconstruct' else ["quality.json", "quality_lossless_frames.csv", "quality_delivered_mp4.csv"])
                    paths = [f"{relative}/{p}" for p in files]
                    metrics, timing, progress = f"resources/{name}.json", f"resources/{name}.time.txt", f"progress/{name}.json"
                    pipeline.step(name, paths+[metrics,timing,progress], paths+[metrics,timing],
                        lambda log, cmd=cmd, m=metrics, p=progress: run_command(REPO, cmd, log, env, root/m, root/p))
        summarize(root, source)
        from etri_combined_reference_evidence import build
        build(root)
        if verify_source(source) != receipts:
            raise ValueError("source receipts changed during experiment")
        print(f"PAIRED_COMBINED_REFERENCE_COMPLETE: {root}", flush=True)


if __name__ == "__main__":
    main()
